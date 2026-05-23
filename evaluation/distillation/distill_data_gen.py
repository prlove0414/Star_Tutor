#!/usr/bin/env python3
"""
Qwen3.5-9B → Qwen3.5-4B 知识蒸馏 · 数据生成
运行环境: AutoDL (vGPU-48GB)

流程:
  1. 加载 9B teacher（bf16），为每个知识点×难度×题型组合生成题目
  2. 每个组合生成 3 道变式题 → 共 6×3×2×3 = 108 道
  3. 保存为 LLaMAFactory Alpaca 格式 → /root/autodl-tmp/star_tutor_distill_data.json
  4. 同时生成蒸馏训练配置文件

用法:
    python distill_data_gen.py
"""
import json, os, sys, time, gc, re
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ══════════════════════════════════════════
#  配置
# ══════════════════════════════════════════
TEACHER_PATH = "/root/autodl-tmp/models/Qwen3.5-9B"
OUTPUT_DIR   = "/root/autodl-tmp/star_tutor_distill"

# 6 知识点 × 3 难度 × 2 题型
KNOWLEDGE_POINTS = ["勾股定理", "一元二次方程", "相似三角形", "一次函数", "概率", "分式方程"]
DIFFICULTIES = ["简单", "中等", "困难"]
QUESTION_TYPES = ["解答题", "填空题"]
VARIANTS_PER_COMBO = 3  # 每个组合生成 3 道变式题

SYSTEM_PROMPT = (
    "你是一个初中数学出题助手。根据用户指定的知识点和难度，生成一道完整的数学题目，"
    "包含题目描述、详细解题步骤和最终答案。"
    "直接输出JSON，不要有任何思考过程或解释。格式如下：\n"
    '{"question": "题目内容", "solution": "解题步骤", "answer": "最终答案"}'
)


# ══════════════════════════════════════════
#  工具
# ══════════════════════════════════════════
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def vram_used() -> str:
    try:
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{used:.1f}G/{total:.1f}G"
    except:
        return "N/A"


def generate_one(model, tokenizer, kp: str, diff: str, qtype: str, variant: int) -> dict:
    """用 teacher 生成一道题"""
    prompt = f"请生成一道关于「{kp}」的{diff}难度{qtype}。"
    if variant > 0:
        prompt += f"（第{variant+1}种变式，不要和之前重复）"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    # 解析 JSON
    try:
        raw_clean = raw.strip()
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw_clean, re.DOTALL)
        if json_match:
            raw_clean = json_match.group(1).strip()
        result = json.loads(raw_clean)
    except (json.JSONDecodeError, KeyError):
        result = {"question": raw, "solution": "", "answer": "", "_parse_error": True}

    return result


def encode_instruction(kp: str, diff: str, qtype: str) -> str:
    """生成 instruction 文本"""
    return f"请生成一道关于「{kp}」的{diff}难度{qtype}，包含题目、解题步骤和最终答案。"


def encode_output(result: dict) -> str:
    """将生成结果编码为 output 文本"""
    q = result.get("question", "")
    s = result.get("solution", "")
    a = result.get("answer", "")

    # 根据题型调整 output 格式
    if "填空题" in str(result.get("_raw_output", "")):
        return f"题目：{q}\n\n答案：{a}"
    else:
        parts = [f"题目：{q}"]
        if s:
            parts.append(f"\n解题步骤：\n{s}")
        if a:
            parts.append(f"\n答案：{a}")
        return "\n".join(parts)


# ══════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_combos = len(KNOWLEDGE_POINTS) * len(DIFFICULTIES) * len(QUESTION_TYPES)
    total_examples = total_combos * VARIANTS_PER_COMBO
    log(f"📋 知识点: {len(KNOWLEDGE_POINTS)} | 难度: {len(DIFFICULTIES)} | 题型: {len(QUESTION_TYPES)}")
    log(f"📊 组合数: {total_combos} | 变式: {VARIANTS_PER_COMBO} | 总计: {total_examples} 题")

    # 加载 teacher
    log(f"加载 teacher: {TEACHER_PATH}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)
    model.eval()
    log(f"  Teacher 就绪 ({time.time()-t0:.0f}s) | 显存: {vram_used()}")

    # 生成数据
    alpaca_data = []
    success_count = 0
    start_time = time.time()

    for kp in KNOWLEDGE_POINTS:
        for diff in DIFFICULTIES:
            for qtype in QUESTION_TYPES:
                for v in range(VARIANTS_PER_COMBO):
                    idx = len(alpaca_data) + 1
                    log(f"  [{idx}/{total_examples}] {kp} · {diff} · {qtype} · v{v+1} ...", end=" ")

                    try:
                        result = generate_one(model, tokenizer, kp, diff, qtype, v)
                        has_error = result.pop("_parse_error", False)
                        result.pop("_raw_output", None)  # 不存原始输出

                        if has_error:
                            log("⚠ 格式异常")
                        else:
                            log(f"✓")

                        instruction = encode_instruction(kp, diff, qtype)
                        output = encode_output(result)

                        alpaca_data.append({
                            "instruction": instruction,
                            "input": "",
                            "output": output,
                            "knowledge_point": kp,
                            "difficulty": diff,
                            "question_type": qtype,
                        })
                        success_count += 1

                    except Exception as e:
                        log(f"❌ {e}")

    elapsed = time.time() - start_time
    log(f"\n✅ 生成完成: {success_count}/{total_examples} ({elapsed:.0f}s)")

    # 保存 Alpaca 格式数据
    data_path = os.path.join(OUTPUT_DIR, "star_tutor_distill_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    log(f"📁 训练数据: {data_path} ({len(alpaca_data)} 条)")

    # 生成 LLaMAFactory 训练配置
    config = {
        "model_name_or_path": "/root/autodl-tmp/models/Qwen3.5-4B",
        "dataset": "star_tutor_distill_data",
        "template": "qwen",
        "finetuning_type": "lora",
        "lora_target": "all",
        "output_dir": "/root/autodl-tmp/output/star_tutor_distill_4b",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_steps": 100,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "max_samples": 500,
        "max_grad_norm": 1.0,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "optim": "adamw_torch",
        "packing": False,
        "upcast_layernorm": True,
        "use_quantization": True,  # QLoRA 4-bit
    }

    config_path = os.path.join(OUTPUT_DIR, "distill_train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 生成 YAML 配置给 LLaMAFactory
    yaml_path = os.path.join(OUTPUT_DIR, "star_tutor_distill.yaml")
    yaml_content = f"""### 星学伴 · 蒸馏训练配置
### Teacher: Qwen3.5-9B (baseline) → Student: Qwen3.5-4B (QLoRA)
### 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}

model_name_or_path: /root/autodl-tmp/models/Qwen3.5-4B
dataset: star_tutor_distill_data
dataset_dir: /root/autodl-tmp/LLaMA-Factory/data
template: qwen
finetuning_type: lora
lora_target: all

output_dir: /root/autodl-tmp/output/star_tutor_distill_4b
logging_dir: /root/autodl-tmp/output/star_tutor_distill_4b/logs

per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.05
max_samples: 500

lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05

bf16: true
use_quantization: true
quantization_bit: 4

save_strategy: steps
save_steps: 100
logging_steps: 10
max_grad_norm: 1.0
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # === 自动注册数据集到 LLaMAFactory ===
    llamafactory_dir = "/root/autodl-tmp/LLaMA-Factory"
    dataset_json_path = os.path.join(llamafactory_dir, "data", "dataset_info.json")
    dataset_name = "star_tutor_distill_data"

    # 复制数据文件
    import shutil
    dest_data = os.path.join(llamafactory_dir, "data", "star_tutor_distill_data.json")
    shutil.copy(data_path, dest_data)
    log(f"📁 数据已复制: {dest_data}")

    # 注册 dataset_info.json
    if os.path.exists(dataset_json_path):
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            ds_info = json.load(f)

        if dataset_name in ds_info:
            log(f"⚠ {dataset_name} 已注册，跳过")
        else:
            ds_info[dataset_name] = {"file_name": "star_tutor_distill_data.json"}
            with open(dataset_json_path, "w", encoding="utf-8") as f:
                json.dump(ds_info, f, ensure_ascii=False, indent=2)
            log(f"✅ 已注册数据集: {dataset_name}")
    else:
        log(f"⚠ 未找到 {dataset_json_path}，请手动注册")

    log(f"📁 训练配置: {config_path}")
    log(f"📁 YAML: {yaml_path}")
    log(f"\n{'='*60}")
    log(f"✅ 数据生成 + 注册完成！直接训练：")
    log(f"   llamafactory-cli train {yaml_path}")
    log(f"{'='*60}")

    # 卸载
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
