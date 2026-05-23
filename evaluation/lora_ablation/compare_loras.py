#!/usr/bin/env python3
"""
星学伴·出题模型 LoRA 消融实验 — 单文件对比脚本
运行环境: AutoDL (vGPU-48GB), Python 3.10+
输出: 4 份 JSON 报告 + 1 份 Markdown 对比报告

用法:
    python compare_loras.py                 # 跑全部 4 组（基座 + r32/r64/r128）
    python compare_loras.py --smoke         # 冒烟模式：每个模型只跑 3 题
    python compare_loras.py --models r32    # 只跑指定模型

依赖: torch, transformers, peft, openai, accelerate
LLaMAFactory 环境默认已安装，无需额外 pip install
"""
import json, os, sys, time, gc, re, argparse
from datetime import datetime
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

# ══════════════════════════════════════════
#  配置（按需修改路径）
# ══════════════════════════════════════════
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen3.5-9B"
SAVES_DIR        = "/root/autodl-tmp/output"  # 可改为 LLaMA-Factory/saves 等
OUTPUT_DIR       = "/root/autodl-tmp/star_tutor_eval"
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# 尝试从 config 加载（本地环境），失败则用上面的默认值
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE
    DEEPSEEK_KEY = DEEPSEEK_API_KEY
except ImportError:
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

# LoRA checkpoint 命名约定
LORA_RUNS = {
    "r32":  "star_tutor_lora_r32",
    "r64":  "star_tutor_lora_r64",
    "r128": "star_tutor_lora_r128",
}

# 测试用例（30 题：6知识点 × 3难度 × 2题型）
TEST_CASES = [
    # === 勾股定理 ===
    {"knowledge_point": "勾股定理", "difficulty": "简单", "question_type": "解答题"},
    {"knowledge_point": "勾股定理", "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "勾股定理", "difficulty": "困难", "question_type": "解答题"},
    {"knowledge_point": "勾股定理", "difficulty": "中等", "question_type": "选择题"},
    {"knowledge_point": "勾股定理", "difficulty": "简单", "question_type": "填空题"},
    # === 一元二次方程 ===
    {"knowledge_point": "一元二次方程", "difficulty": "简单", "question_type": "解答题"},
    {"knowledge_point": "一元二次方程", "difficulty": "中等", "question_type": "选择题"},
    {"knowledge_point": "一元二次方程", "difficulty": "困难", "question_type": "解答题"},
    {"knowledge_point": "一元二次方程", "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "一元二次方程", "difficulty": "简单", "question_type": "填空题"},
    # === 相似三角形 ===
    {"knowledge_point": "相似三角形", "difficulty": "简单", "question_type": "选择题"},
    {"knowledge_point": "相似三角形", "difficulty": "中等", "question_type": "解答题"},
    {"knowledge_point": "相似三角形", "difficulty": "困难", "question_type": "解答题"},
    {"knowledge_point": "相似三角形", "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "相似三角形", "difficulty": "简单", "question_type": "解答题"},
    # === 一次函数 ===
    {"knowledge_point": "一次函数", "difficulty": "简单", "question_type": "填空题"},
    {"knowledge_point": "一次函数", "difficulty": "中等", "question_type": "解答题"},
    {"knowledge_point": "一次函数", "difficulty": "困难", "question_type": "选择题"},
    {"knowledge_point": "一次函数", "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "一次函数", "difficulty": "简单", "question_type": "选择题"},
    # === 概率 ===
    {"knowledge_point": "概率", "difficulty": "简单", "question_type": "选择题"},
    {"knowledge_point": "概率", "difficulty": "中等", "question_type": "解答题"},
    {"knowledge_point": "概率", "difficulty": "困难", "question_type": "填空题"},
    {"knowledge_point": "概率", "difficulty": "中等", "question_type": "选择题"},
    {"knowledge_point": "概率", "difficulty": "简单", "question_type": "解答题"},
    # === 分式方程 ===
    {"knowledge_point": "分式方程", "difficulty": "简单", "question_type": "填空题"},
    {"knowledge_point": "分式方程", "difficulty": "中等", "question_type": "解答题"},
    {"knowledge_point": "分式方程", "difficulty": "困难", "question_type": "解答题"},
    {"knowledge_point": "分式方程", "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "分式方程", "difficulty": "简单", "question_type": "选择题"},
]

SYSTEM_PROMPT = (
    "你是一个初中数学出题助手。根据用户指定的知识点和难度，生成一道完整的数学题目，"
    "包含题目描述、详细解题步骤和最终答案。"
    "不要有任何思考过程或解释，直接输出JSON。格式如下：\n"
    '{"question": "题目内容", "solution": "解题步骤", "answer": "最终答案"}'
)

JUDGE_PROMPT = """你是初中数学出题质量评估专家。根据给定的出题要求和模型生成的题目，从三个维度评分。

## 评分规则

1. **知识点匹配度 (1-5)**：生成的题目是否真正考察了指定的知识点？无关内容是否引入？
   - 5: 精准命中，所有考察点都属于该知识点
   - 3: 部分相关，混入了一些其他知识
   - 1: 完全不匹配或考察了错误的知识点

2. **难度校准度 (1-5)**：生成的题目难度是否与要求的难度一致？
   - 5: 难度精准，完全符合要求等级
   - 3: 有偏差但仍可接受
   - 1: 难度完全不符（简单题变难题 或 难题变简单题）

3. **可解性 (是/否)**：题目本身是否可以解答？提供的答案是否与题目一致且正确？
   - 是: 题目完整、有解、答案正确
   - 否: 题目有歧义、缺少条件、答案错误或无答案

## 输入

出题要求：
- 知识点：{knowledge_point}
- 目标难度：{target_difficulty}
- 题型：{question_type}

模型输出：
{model_output}

## 输出格式
严格按 JSON 输出，不要输出其他内容：
{{
  "知识点匹配度": 1-5,
  "知识点匹配分析": "一句话分析",
  "难度校准度": 1-5,
  "难度校准分析": "一句话分析",
  "可解性": "是/否",
  "可解性分析": "一句话分析",
  "综合评分": 0-100,
  "备注": "补充说明（可选）"
}}"""


# ══════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════

def log(msg: str, **kwargs):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True, **kwargs)


def find_lora_path(run_name: str) -> str | None:
    """自动查找 LLaMAFactory LoRA checkpoint 路径，兼容多种目录结构"""
    base = os.path.join(SAVES_DIR, run_name)
    if not os.path.isdir(base):
        log(f"⚠ 目录不存在: {base}")
        return None

    # 优先级：adapter_model.safetensors 直接存在 > checkpoint-xxx/adapter_model.safetensors
    for candidate in [base] + sorted(
        [os.path.join(base, d) for d in os.listdir(base) if d.startswith("checkpoint-")],
        reverse=True,
    ):
        if os.path.exists(os.path.join(candidate, "adapter_model.safetensors")):
            log(f"  找到 LoRA checkpoint: {candidate}")
            return candidate

    log(f"⚠ 未找到 adapter_model.safetensors: {base}")
    return None


def load_model(lora_path: str | None = None) -> tuple:
    """加载基座模型 + 可选 LoRA adapter"""
    log(f"加载基座模型: {BASE_MODEL_PATH}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    log(f"  基座加载完成 ({time.time()-t0:.0f}s), 显存: {_vram_used()}")

    if lora_path:
        from peft import PeftModel
        log(f"加载 LoRA adapter: {lora_path}")
        t0 = time.time()
        model = PeftModel.from_pretrained(model, lora_path)
        log(f"  LoRA 加载完成 ({time.time()-t0:.0f}s), 显存: {_vram_used()}")

    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer) -> None:
    """释放模型显存"""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log(f"  模型已卸载, 显存: {_vram_used()}")


def _vram_used() -> str:
    try:
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{used:.1f}G / {total:.1f}G"
    except Exception:
        return "N/A"


def generate_one(model, tokenizer, kp: str, diff: str, qtype: str) -> dict:
    """生成一道题，返回 {question, solution, answer} 或原始文本（JSON 解析失败时）"""
    prompt_text = f"请生成一道关于「{kp}」的{diff}难度{qtype}。"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    raw = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    # 尝试解析 JSON — 处理 Qwen3.5 思考型输出：&lt;/think&gt; → ```json...```
    try:
        raw_clean = raw.strip()

        # Qwen3.5 思考型：提取 &lt;/think&gt; 之后的内容
        THINK_CLOSE = "\x3c/think\x3e"  # &lt;/think&gt; escaped for safety
        if THINK_CLOSE in raw_clean:
            after_think = raw_clean.split(THINK_CLOSE)[-1].strip()
            if after_think:
                raw_clean = after_think

        # 提取 ```json ... ``` 代码块
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw_clean, re.DOTALL)
        if json_match:
            raw_clean = json_match.group(1).strip()

        result = json.loads(raw_clean)
    except (json.JSONDecodeError, KeyError):
        result = {"question": raw, "solution": "", "answer": "", "_parse_error": True}

    result["_raw_output"] = raw
    result["_gen_time"] = round(elapsed, 2)
    result["_gen_tokens"] = int(outputs.shape[1] - inputs.input_ids.shape[1])
    return result


def judge_one(client: OpenAI, kp: str, diff: str, qtype: str, model_output: dict) -> dict:
    """用 DeepSeek V4 裁判一道题"""
    prompt = JUDGE_PROMPT.format(
        knowledge_point=kp,
        target_difficulty=diff,
        question_type=qtype,
        model_output=json.dumps(model_output, ensure_ascii=False, indent=2),
    )
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=30,
        )
        raw = resp.choices[0].message.content
        # 清理 markdown 代码块
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = re.sub(r"```\w*\n?", "", cleaned).replace("```", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        return {"综合评分": 0, "错误": str(e)}


def aggregate_scores(scores: list[dict]) -> dict:
    """汇总评分"""
    if not scores:
        return {}
    kp_scores       = [s.get("知识点匹配度", 0) or 0 for s in scores]
    diff_scores     = [s.get("难度校准度", 0) or 0 for s in scores]
    solvable        = sum(1 for s in scores if s.get("可解性") == "是")
    composite       = [s.get("综合评分", 0) or 0 for s in scores]
    format_pass     = sum(1 for r in [s.get("_raw", {}) for s in scores]
                          if "_parse_error" not in r)
    gen_times       = [s.get("_raw", {}).get("_gen_time", 0) for s in scores]
    gen_tokens      = [s.get("_raw", {}).get("_gen_tokens", 0) for s in scores]

    # 按知识点分组
    by_kp = {}
    for s in scores:
        kp = s.get("test_case", {}).get("knowledge_point", "unknown")
        by_kp.setdefault(kp, []).append(s.get("综合评分", 0) or 0)

    # 按难度分组
    by_diff = {"简单": [], "中等": [], "困难": []}
    for s in scores:
        d = s.get("test_case", {}).get("difficulty", "")
        if d in by_diff:
            by_diff[d].append(s.get("综合评分", 0) or 0)

    return {
        "样本数":           len(scores),
        "格式合规率":       f"{format_pass}/{len(scores)} ({format_pass/len(scores)*100:.0f}%)",
        "知识点匹配度_平均": round(sum(kp_scores)/len(kp_scores), 2) if kp_scores else 0,
        "难度校准度_平均":   round(sum(diff_scores)/len(diff_scores), 2) if diff_scores else 0,
        "可解率":           f"{solvable}/{len(scores)} ({solvable/len(scores)*100:.0f}%)" if scores else "0/0",
        "综合评分_平均":     round(sum(composite)/len(composite), 2) if composite else 0,
        "平均生成耗时_秒":   round(sum(gen_times)/len(gen_times), 2) if gen_times else 0,
        "平均输出token数":   round(sum(gen_tokens)/len(gen_tokens), 0) if gen_tokens else 0,
        "按知识点":          {k: round(sum(v)/len(v), 2) if v else 0 for k, v in by_kp.items()},
        "按难度":            {k: round(sum(v)/len(v), 2) if v else 0 for k, v in by_diff.items()},
        "详细评分":          scores,
    }


# ══════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════

def run_one_model(label: str, lora_path: str | None, test_cases: list, judge_client: OpenAI) -> dict:
    """运行单个模型的完整评估"""
    log(f"\n{'='*60}")
    log(f"  {label} — {len(test_cases)} 题")
    log(f"{'='*60}")

    model, tokenizer = load_model(lora_path)

    scores = []
    for i, case in enumerate(test_cases):
        kp, diff, qtype = case["knowledge_point"], case["difficulty"], case.get("question_type", "解答题")

        # Step 1: 生成
        log(f"  [{i+1}/{len(test_cases)}] 出题: {kp} {diff} {qtype} ...", end=" ")
        try:
            gen = generate_one(model, tokenizer, kp, diff, qtype)
            log(f"OK ({gen.get('_gen_time', 0):.1f}s, {gen.get('_gen_tokens', 0)}t)")
        except Exception as e:
            log(f"❌ 生成失败: {e}")
            scores.append({"综合评分": 0, "错误": f"生成失败: {e}", "test_case": case, "_raw": {}})
            continue

        # Step 2: 裁判
        log(f"      裁判中 ...", end=" ")
        try:
            score = judge_one(judge_client, kp, diff, qtype, gen)
            score["test_case"] = case
            score["_raw"] = gen
            scores.append(score)
            log(f"综合={score.get('综合评分','?')} 匹配={score.get('知识点匹配度','?')} "
                f"难度={score.get('难度校准度','?')} 可解={score.get('可解性','?')}")
        except Exception as e:
            log(f"❌ 裁判失败: {e}")
            scores.append({"综合评分": 0, "错误": f"裁判失败: {e}", "test_case": case, "_raw": gen})

    unload_model(model, tokenizer)

    report = aggregate_scores(scores)
    report["label"] = label
    report["lora_path"] = lora_path

    # 存 JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, f"harness_report_{label}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"  📁 {json_path}")
    return report


def generate_markdown_report(reports: dict) -> str:
    """生成 Markdown 对比报告"""
    models = list(reports.keys())
    if not models:
        return "# 无数据\n"

    lines = []
    lines.append("# 星学伴 · 出题模型 LoRA 消融实验报告")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## 实验配置")
    lines.append("")
    lines.append("| 项目 | 配置 |")
    lines.append("|------|------|")
    lines.append(f"| 基座模型 | Qwen3.5-9B |")
    lines.append(f"| 训练数据 | 2,986 题 (8:1:1) |")
    lines.append(f"| 框架 | LLaMAFactory v0.9.5 |")
    lines.append(f"| 硬件 | RTX 4090 48GB (vGPU) |")
    lines.append(f"| 测试集 | {len(TEST_CASES)} 题 (6知识点 × 3难度 × 2题型) |")
    lines.append(f"| 裁判模型 | DeepSeek V4 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 总览对比")
    lines.append("")

    # 表头
    header = ["指标"] + models
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["------"] * len(header)) + "|")

    metrics = [
        ("综合评分", "综合评分_平均"),
        ("知识点匹配度", "知识点匹配度_平均"),
        ("难度校准度", "难度校准度_平均"),
        ("可解率", "可解率"),
        ("格式合规率", "格式合规率"),
        ("平均生成耗时", "平均生成耗时_秒"),
        ("平均输出 token", "平均输出token数"),
    ]

    for display_name, key in metrics:
        row = [display_name]
        for m in models:
            val = reports[m].get(key, "-")
            if isinstance(val, float):
                val = f"{val:.1f}" if key != "平均输出token数" else f"{int(val)}"
            row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    # 消融分析
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 消融分析")
    lines.append("")

    # 提取 rank 值用于排序
    def _rank_of(label: str) -> int:
        match = re.search(r'r(\d+)', label)
        return int(match.group(1)) if match else 0

    lora_labels = [m for m in models if m != "baseline"]
    lora_labels.sort(key=_rank_of)

    baseline_score = reports.get("baseline", {}).get("综合评分_平均", 0)
    lines.append(f"**基座 9B 综合评分**: {baseline_score}")
    lines.append("")

    for i, label in enumerate(lora_labels):
        report = reports[label]
        rank = _rank_of(label)
        score = report.get("综合评分_平均", 0)
        delta = score - baseline_score
        improvement = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        lines.append(f"- **r={rank}**: 综合评分 {score} ({improvement}{abs(delta):.1f} vs 基座)")

        if i > 0:
            prev_label = lora_labels[i-1]
            prev_score = reports[prev_label].get("综合评分_平均", 0)
            marginal = score - prev_score
            lines.append(f"  - 边际收益 (vs r={_rank_of(prev_label)}): {marginal:+.1f}")

    # 边际收益递减判断
    if len(lora_labels) >= 3:
        s32 = reports[lora_labels[0]].get("综合评分_平均", 0)
        s64 = reports[lora_labels[1]].get("综合评分_平均", 0)
        s128 = reports[lora_labels[2]].get("综合评分_平均", 0)
        gain_32_64 = s64 - s32
        gain_64_128 = s128 - s64
        lines.append("")
        lines.append("**边际收益趋势**:")
        lines.append(f"- r=32 → r=64: {gain_32_64:+.1f}")
        lines.append(f"- r=64 → r=128: {gain_64_128:+.1f}")
        if gain_64_128 < gain_32_64:
            lines.append("- ⚠ 边际收益递减，r=128 可能过拟合或收益饱和")
        lines.append("")

    # 推荐
    best_label = max(lora_labels, key=lambda l: reports[l].get("综合评分_平均", 0))
    best_rank = _rank_of(best_label)
    lines.append(f"**推荐 rank**: r={best_rank}（综合评分最优）")

    # 按知识点分组
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 按知识点分组")
    lines.append("")

    all_kps = set()
    for r in reports.values():
        all_kps.update(r.get("按知识点", {}).keys())
    all_kps = sorted(all_kps)

    kp_header = ["知识点"] + models
    lines.append("| " + " | ".join(kp_header) + " |")
    lines.append("|" + "|".join(["------"] * len(kp_header)) + "|")
    for kp in all_kps:
        row = [kp]
        for m in models:
            val = reports[m].get("按知识点", {}).get(kp, "-")
            row.append(f"{val:.1f}" if isinstance(val, (int, float)) else str(val))
        lines.append("| " + " | ".join(row) + " |")

    # 按难度分组
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 按难度分组")
    lines.append("")

    diff_header = ["难度"] + models
    lines.append("| " + " | ".join(diff_header) + " |")
    lines.append("|" + "|".join(["------"] * len(diff_header)) + "|")
    for diff in ["简单", "中等", "困难"]:
        row = [diff]
        for m in models:
            val = reports[m].get("按难度", {}).get(diff, "-")
            row.append(f"{val:.1f}" if isinstance(val, (int, float)) else str(val))
        lines.append("| " + " | ".join(row) + " |")

    # JSON 报告索引
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 原始数据")
    lines.append("")
    for m in models:
        lines.append(f"- `harness_report_{m}.json` — {m} 详细评分（30 题逐题数据）")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="LoRA 消融实验对比")
    parser.add_argument("--smoke", action="store_true", help="冒烟模式：只跑 3 题")
    parser.add_argument("--models", nargs="*", help="指定模型（如 --models r32 r64），默认全部")
    args = parser.parse_args()

    test_cases = TEST_CASES[:3] if args.smoke else TEST_CASES
    log(f"📋 测试用例: {len(test_cases)} 题{' (冒烟模式)' if args.smoke else ''}")

    # 初始化 DeepSeek 裁判客户端
    judge_client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_API_BASE)

    # 确定要跑的模型列表
    runs = {"baseline": None}  # 基座无 LoRA
    for rank, run_name in LORA_RUNS.items():
        path = find_lora_path(run_name)
        if path:
            runs[rank] = path
        else:
            log(f"⚠ LoRA r={rank} checkpoint 未找到，跳过")

    if args.models:
        runs = {k: v for k, v in runs.items() if k in args.models or k == "baseline"}

    log(f"🚀 开始评估: {', '.join(runs.keys())}")

    reports = {}
    for label, lora_path in runs.items():
        try:
            reports[label] = run_one_model(label, lora_path, test_cases, judge_client)
        except Exception as e:
            log(f"❌ {label} 评估中断: {e}")
            import traceback
            traceback.print_exc()

    # 生成 Markdown 报告
    if reports:
        md = generate_markdown_report(reports)
        md_path = os.path.join(OUTPUT_DIR, "lora_comparison_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)

        log(f"\n{'='*60}")
        log(f"✅ 完成！")
        log(f"  Markdown 报告: {md_path}")
        log(f"  JSON 报告目录: {OUTPUT_DIR}/")
        log(f"{'='*60}")

        # 打印摘要
        print("\n" + md[:3000])
    else:
        log("❌ 没有成功完成任何评估")


if __name__ == "__main__":
    main()
