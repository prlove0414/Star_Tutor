#!/usr/bin/env python3
"""
星学伴 · 蒸馏数据生成（纯生成，多重解析）
运行环境: AutoDL

策略：
  1. 9B teacher 生成原始文本
  2. 五层 JSON 解析（从严格到宽松）
  3. 解析失败 → 自动重试（换 seed）
  4. 保存原始输出 + 解析结果，便于后续清洗
"""
import json, os, sys, time, gc, re, random
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ══════════════════════════════════════════
#  配置
# ══════════════════════════════════════════
TEACHER_PATH = "/root/autodl-tmp/models/Qwen3.5-9B"
OUTPUT_FILE  = "/root/autodl-tmp/star_tutor_distill/raw_data.jsonl"

KNOWLEDGE_POINTS = ["勾股定理", "一元二次方程", "相似三角形", "一次函数", "概率", "分式方程"]
DIFFICULTIES = ["简单", "中等", "困难"]
QUESTION_TYPES = ["解答题", "填空题"]
VARIANTS = 3  # 每个组合 3 道变式
MAX_RETRIES = 2  # 解析失败最多重试 2 次

SYSTEM_PROMPT = (
    "你是一个初中数学出题助手。根据用户指定的知识点和难度，生成一道完整的数学题目，"
    "包含题目描述、详细解题步骤和最终答案。\n"
    '直接输出JSON。格式：{"question":"题目","solution":"解题步骤","answer":"答案"}'
)

# ══════════════════════════════════════════
#  日志
# ══════════════════════════════════════════
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def vram():
    try:
        u = torch.cuda.memory_allocated()/1024**3
        t = torch.cuda.get_device_properties(0).total_memory/1024**3
        return f"{u:.1f}G/{t:.1f}G"
    except: return "N/A"


# ══════════════════════════════════════════
#  JSON 解析（五层回退）
# ══════════════════════════════════════════
def extract_json(raw: str) -> dict | None:
    """
    尝试 5 种策略提取合法 JSON，返回 dict 或 None
    """
    strategies = []

    # S1: 直接解析
    strategies.append(("S1直接", raw.strip()))

    # S2: 提取最后一个 ```json...``` 块
    fences = list(re.finditer(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL))
    if fences:
        strategies.append(("S2代码块", fences[-1].group(1).strip()))

    # S3: 查找第一个 { 到最后一个 } 
    first = raw.find("{")
    last = raw.rfind("}")
    if first >= 0 and last > first:
        strategies.append(("S3括号", raw[first:last+1].strip()))

    # S4: 尝试补全截断的 JSON（缺 } 补齐）
    if raw.strip().startswith("{") and not raw.strip().endswith("}"):
        fixed = raw.strip()
        # 数括号差
        diff = fixed.count("{") - fixed.count("}")
        if diff > 0:
            fixed += "}" * diff
            strategies.append(("S4补全", fixed))

    # S5: 去掉首尾非 JSON 字符
    cleaned = re.sub(r'^[^{]*', '', raw)
    cleaned = re.sub(r'[^}]*$', '', cleaned)
    if cleaned and cleaned != raw:
        strategies.append(("S5去噪", cleaned))

    for name, text in strategies:
        try:
            result = json.loads(text)
            if isinstance(result, dict) and "question" in result:
                return result
        except (json.JSONDecodeError, ValueError):
            continue

    return None


# ══════════════════════════════════════════
#  生成一道题（带重试）
# ══════════════════════════════════════════
def generate_one(model, tokenizer, kp, diff, qtype, variant) -> dict:
    """生成一道题，解析失败自动重试"""

    for attempt in range(MAX_RETRIES + 1):
        seed = random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

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
                max_new_tokens=1536,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        result = extract_json(raw)
        if result:
            return {
                "status": "ok",
                "attempts": attempt + 1,
                "raw": raw,
                **result,
            }

    # 全部重试失败
    return {
        "status": "parse_failed",
        "attempts": MAX_RETRIES + 1,
        "raw": raw,
        "question": raw.strip()[:200],
        "solution": "",
        "answer": "",
    }


# ══════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════
def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    total = len(KNOWLEDGE_POINTS) * len(DIFFICULTIES) * len(QUESTION_TYPES) * VARIANTS
    log(f"📊 总计 {total} 题 | 每个组合 {VARIANTS} 变式 | 最大重试 {MAX_RETRIES}")

    # 加载 teacher
    log(f"加载 teacher: {TEACHER_PATH}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)
    model.eval()
    log(f"  就绪 ({time.time()-t0:.0f}s) | 显存: {vram()}")

    ok_count = 0
    fail_count = 0
    start_time = time.time()
    fout = open(OUTPUT_FILE, "w", encoding="utf-8")

    idx = 0
    for kp in KNOWLEDGE_POINTS:
        for diff in DIFFICULTIES:
            for qtype in QUESTION_TYPES:
                for v in range(VARIANTS):
                    idx += 1
                    result = generate_one(model, tokenizer, kp, diff, qtype, v)

                    record = {
                        "id": idx,
                        "knowledge_point": kp,
                        "difficulty": diff,
                        "question_type": qtype,
                        "variant": v + 1,
                        **result,
                    }

                    status = "✅" if result["status"] == "ok" else "❌"
                    attempts = result.get("attempts", "?")
                    log(f"  [{idx:3d}/{total}] {status} {kp}·{diff}·{qtype}·v{v+1} "
                        f"(尝试{attempts}次)")

                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()

                    if result["status"] == "ok":
                        ok_count += 1
                    else:
                        fail_count += 1

    fout.close()
    elapsed = time.time() - start_time

    # 汇总
    log(f"\n{'='*60}")
    log(f"✅ 成功: {ok_count}/{total} ({ok_count/total*100:.0f}%)")
    log(f"❌ 失败: {fail_count}/{total} ({fail_count/total*100:.0f}%)")
    log(f"⏱  耗时: {elapsed:.0f}s ({elapsed/total:.1f}s/题)")
    log(f"📁 {OUTPUT_FILE}")
    log(f"{'='*60}")

    # 卸载
    del model; del tokenizer; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
