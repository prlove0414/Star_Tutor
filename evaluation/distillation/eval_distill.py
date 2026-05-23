#!/usr/bin/env python3
"""
星学伴 · 蒸馏效果评估：4B Student (LoRA) vs 9B Teacher Baseline
运行环境: AutoDL (vGPU-48GB), Python 3.10+

用法:
    python eval_distill.py           # 全量 30 题
    python eval_distill.py --smoke   # 冒烟测试 3 题

输出:
    /root/autodl-tmp/star_tutor_distill/eval/
    ├── harness_report_student.json   # 逐题评分详情
    └── distill_comparison.md         # 对比报告
"""
import json, os, time, gc, re, argparse
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from openai import OpenAI

# ============================================
# 配置
# ============================================
BASE_4B_PATH    = "/root/autodl-tmp/models/Qwen3.5-4B"
LORA_DIR        = "/root/autodl-tmp/star_tutor_distill/output"
OUTPUT_DIR      = "/root/autodl-tmp/star_tutor_distill/eval"
DEEPSEEK_KEY    = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE   = "https://api.deepseek.com/v1"

# 9B Teacher 基线 (L1 全量 30 题)
BASELINE_9B = {
    "综合评分": 90.3,
    "可解率":   "27/30 (90%)",
    "知识点匹配": 4.8,
    "难度校准":   4.5,
}

# 测试用例 (30题: 6知识点 * 3难度, 混合题型)
TEST_CASES = [
    {"knowledge_point": "勾股定理",   "difficulty": "简单", "question_type": "解答题"},
    {"knowledge_point": "勾股定理",   "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "勾股定理",   "difficulty": "困难", "question_type": "解答题"},
    {"knowledge_point": "勾股定理",   "difficulty": "中等", "question_type": "选择题"},
    {"knowledge_point": "勾股定理",   "difficulty": "简单", "question_type": "填空题"},
    {"knowledge_point": "一元二次方程", "difficulty": "简单", "question_type": "解答题"},
    {"knowledge_point": "一元二次方程", "difficulty": "中等", "question_type": "选择题"},
    {"knowledge_point": "一元二次方程", "difficulty": "困难", "question_type": "解答题"},
    {"knowledge_point": "一元二次方程", "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "一元二次方程", "difficulty": "简单", "question_type": "填空题"},
    {"knowledge_point": "相似三角形",  "difficulty": "简单", "question_type": "选择题"},
    {"knowledge_point": "相似三角形",  "difficulty": "中等", "question_type": "解答题"},
    {"knowledge_point": "相似三角形",  "difficulty": "困难", "question_type": "解答题"},
    {"knowledge_point": "相似三角形",  "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "相似三角形",  "difficulty": "简单", "question_type": "解答题"},
    {"knowledge_point": "一次函数",    "difficulty": "简单", "question_type": "填空题"},
    {"knowledge_point": "一次函数",    "difficulty": "中等", "question_type": "解答题"},
    {"knowledge_point": "一次函数",    "difficulty": "困难", "question_type": "选择题"},
    {"knowledge_point": "一次函数",    "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "一次函数",    "difficulty": "简单", "question_type": "选择题"},
    {"knowledge_point": "概率",       "difficulty": "简单", "question_type": "选择题"},
    {"knowledge_point": "概率",       "difficulty": "中等", "question_type": "解答题"},
    {"knowledge_point": "概率",       "difficulty": "困难", "question_type": "填空题"},
    {"knowledge_point": "概率",       "difficulty": "中等", "question_type": "选择题"},
    {"knowledge_point": "概率",       "difficulty": "简单", "question_type": "解答题"},
    {"knowledge_point": "分式方程",    "difficulty": "简单", "question_type": "填空题"},
    {"knowledge_point": "分式方程",    "difficulty": "中等", "question_type": "解答题"},
    {"knowledge_point": "分式方程",    "difficulty": "困难", "question_type": "解答题"},
    {"knowledge_point": "分式方程",    "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "分式方程",    "difficulty": "简单", "question_type": "选择题"},
]

SYSTEM_PROMPT = (
    "你是一个初中数学出题助手。根据用户指定的知识点和难度，生成一道完整的数学题目，"
    "包含题目描述、详细解题步骤和最终答案。"
    "不要有任何思考过程或解释，直接输出JSON。格式如下：\n"
    "{\"question\": \"题目内容\", \"solution\": \"解题步骤\", \"answer\": \"最终答案\"}"
)

JUDGE_PROMPT = (
    "你是初中数学出题质量评估专家。根据给定的出题要求和模型生成的题目，从三个维度评分。\n\n"
    "## 评分规则\n\n"
    "1. **知识点匹配度 (1-5)**：生成的题目是否真正考察了指定的知识点？\n"
    "   - 5: 精准命中  3: 部分相关  1: 完全不匹配\n\n"
    "2. **难度校准度 (1-5)**：生成的题目难度是否与要求一致？\n"
    "   - 5: 精准匹配  3: 有偏差但可接受  1: 完全不符\n\n"
    "3. **可解性 (是/否)**：题目是否完整可解？答案是否正确？\n\n"
    "## 输入\n\n"
    "出题要求：知识点={knowledge_point} 目标难度={target_difficulty} 题型={question_type}\n\n"
    "模型输出：\n{model_output}\n\n"
    "## 输出格式\n"
    "严格按JSON输出，不要其他内容：\n"
    '{{"知识点匹配度":1-5,"知识点匹配分析":"...","难度校准度":1-5,'
    '"难度校准分析":"...","可解性":"是/否","可解性分析":"...","综合评分":0-100,"备注":"..."}}'
)


# ============================================
# 工具函数
# ============================================

def log(msg: str, **kw):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True, **kw)


def vram() -> str:
    try:
        u = torch.cuda.memory_allocated() / 1024**3
        t = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{u:.1f}G/{t:.1f}G"
    except:
        return "N/A"


def find_lora_ckpt(lora_dir: str) -> str:
    """自动找 LoRA checkpoint"""
    if os.path.isfile(os.path.join(lora_dir, "adapter_model.safetensors")):
        return lora_dir
    ckpts = sorted(
        [d for d in os.listdir(lora_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]), reverse=True
    )
    for ckpt in ckpts:
        p = os.path.join(lora_dir, ckpt)
        if os.path.isfile(os.path.join(p, "adapter_model.safetensors")):
            return p
    raise FileNotFoundError(f"找不到 adapter_model.safetensors 在 {lora_dir}")


def load_student():
    """加载 4B base + LoRA adapter"""
    log(f"加载基座: {BASE_4B_PATH}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_4B_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_4B_PATH, trust_remote_code=True)
    log(f"  基座 OK ({time.time()-t0:.0f}s) | 显存: {vram()}")

    ckpt = find_lora_ckpt(LORA_DIR)
    log(f"加载 LoRA: {ckpt}")
    t0 = time.time()
    model = PeftModel.from_pretrained(model, ckpt)
    model.eval()
    log(f"  LoRA OK ({time.time()-t0:.0f}s) | 显存: {vram()}")
    return model, tok


def unload(model, tok):
    del model; del tok; gc.collect(); torch.cuda.empty_cache()
    log(f"  卸载完成 | 显存: {vram()}")


def generate(model, tok, kp: str, diff: str, qtype: str) -> dict:
    """生成一道题"""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请生成一道关于「{kp}」的{diff}难度{qtype}。"},
    ]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tok(text, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=2048, temperature=0.7, do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
    elapsed = time.time() - t0
    raw = tok.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    # JSON 解析
    try:
        clean = raw.strip()
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", clean, re.DOTALL)
        if m:
            clean = m.group(1).strip()
        result = json.loads(clean)
    except (json.JSONDecodeError, KeyError):
        result = {"question": raw, "solution": "", "answer": "", "_parse_error": True}

    result["_raw"] = raw
    result["_gen_time"] = round(elapsed, 2)
    result["_gen_tokens"] = int(out.shape[1] - inputs.input_ids.shape[1])
    return result


def judge(client: OpenAI, kp: str, diff: str, qtype: str, gen: dict) -> dict:
    """DeepSeek V4 裁判"""
    prompt = JUDGE_PROMPT.format(
        knowledge_point=kp, target_difficulty=diff,
        question_type=qtype,
        model_output=json.dumps(gen, ensure_ascii=False, indent=2),
    )
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat", messages=[{"role": "user", "content": prompt}],
            temperature=0.1, timeout=30,
        )
        raw = resp.choices[0].message.content.strip()
        log(f"      [judge raw] {raw[:200]}...")

        # 多重解析策略
        result = None
        # S1: 直接 JSON
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            pass
        # S2: 去掉 markdown 代码块
        if result is None:
            cleaned = re.sub(r"```\w*\n?", "", raw).replace("```", "").strip()
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        # S3: 提取 ```json...``` 块
        if result is None:
            m = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    pass
        # S4: 从 { 到 }
        if result is None:
            first = raw.find("{")
            last = raw.rfind("}")
            if first >= 0 and last > first:
                try:
                    result = json.loads(raw[first:last+1])
                except json.JSONDecodeError:
                    pass

        if result is None:
            log(f"      [judge FAIL] 无法解析: {raw[:300]}")
            return {"综合评分": 0, "错误": f"JSON解析失败", "_raw": raw[:500]}

        return result

    except Exception as e:
        import traceback
        log(f"      [judge EXCEPTION] {e}")
        return {"综合评分": 0, "错误": str(e), "_traceback": traceback.format_exc()[:300]}


def agg(scores: list) -> dict:
    """汇总指标"""
    if not scores:
        return {}
    kps = [s.get("知识点匹配度", 0) or 0 for s in scores]
    diffs = [s.get("难度校准度", 0) or 0 for s in scores]
    solv = sum(1 for s in scores if s.get("可解性") == "是")
    comp = [s.get("综合评分", 0) or 0 for s in scores]

    by_kp = {}
    for s in scores:
        k = s.get("test_case", {}).get("knowledge_point", "?")
        by_kp.setdefault(k, []).append(s.get("综合评分", 0) or 0)

    by_diff = {"简单": [], "中等": [], "困难": []}
    for s in scores:
        d = s.get("test_case", {}).get("difficulty", "")
        if d in by_diff:
            by_diff[d].append(s.get("综合评分", 0) or 0)

    return {
        "样本数": len(scores),
        "知识点匹配_平均": round(sum(kps)/len(kps), 2) if kps else 0,
        "难度校准_平均":   round(sum(diffs)/len(diffs), 2) if diffs else 0,
        "可解率": f"{solv}/{len(scores)} ({solv/len(scores)*100:.0f}%)" if scores else "0/0",
        "综合评分_平均":   round(sum(comp)/len(comp), 2) if comp else 0,
        "按知识点": {k: round(sum(v)/len(v), 2) if v else 0 for k, v in by_kp.items()},
        "按难度":   {k: round(sum(v)/len(v), 2) if v else 0 for k, v in by_diff.items()},
        "详细评分": scores,
    }


def md_report(student: dict, baseline: dict) -> str:
    """生成 Markdown 对比报告"""
    s = student
    b = baseline
    delta_score = round(s.get("综合评分_平均", 0) - b.get("综合评分", 0), 1)
    delta_kp = round(s.get("知识点匹配_平均", 0) - b.get("知识点匹配", 0), 1)
    delta_diff = round(s.get("难度校准_平均", 0) - b.get("难度校准", 0), 1)

    lines = [
        "# 星学伴 · 蒸馏效果评估报告",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 实验配置",
        "",
        "| 项目 | Teacher (9B) | Student (4B+LoRA) |",
        "|------|-------------|-------------------|",
        "| 基座模型 | Qwen3.5-9B | Qwen3.5-4B |",
        "| 训练方式 | 无（基座推理） | QLoRA (r=32, alpha=64) |",
        "| 训练数据 | - | 100 题（9B 教师蒸馏） |",
        "| 测试集 | 30 题 | 30 题（相同） |",
        "| 裁判模型 | DeepSeek V4 | DeepSeek V4 |",
        "",
        "---",
        "",
        "## 总览对比",
        "",
        "| 指标 | Teacher (9B) | Student (4B) | 差距 |",
        "|------|-------------|-------------|------|",
        f"| 综合评分 | {b.get('综合评分', '-')} | {s.get('综合评分_平均', '-')} | {delta_score} |",
        f"| 知识点匹配 | {b.get('知识点匹配', '-')} | {s.get('知识点匹配_平均', '-')} | {delta_kp} |",
        f"| 难度校准 | {b.get('难度校准', '-')} | {s.get('难度校准_平均', '-')} | {delta_diff} |",
        f"| 可解率 | {b.get('可解率', '-')} | {s.get('可解率', '-')} | - |",
        "",
        "---",
        "",
        "## Student 按知识点",
        "",
        "| 知识点 | 综合评分 |",
        "|--------|---------|",
    ]
    for k, v in s.get("按知识点", {}).items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "---",
        "",
        "## Student 按难度",
        "",
        "| 难度 | 综合评分 |",
        "|------|---------|",
    ]
    for k, v in s.get("按难度", {}).items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "---",
        "",
        "## 结论",
        "",
        f"- Student 可解率：{s.get('可解率', '-')} vs Teacher：{b.get('可解率', '-')}",
        f"- Student 综合评分：{s.get('综合评分_平均', '-')} vs Teacher：{b.get('综合评分', '-')}",
        f"- 差距（综合评分）：{delta_score}",
    ]

    return "\n".join(lines)


# ============================================
# 主流程
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="冒烟测试（3题）")
    args = parser.parse_args()

    cases = TEST_CASES[:3] if args.smoke else TEST_CASES
    log(f"📋 测试用例: {len(cases)} 题{' (冒烟)' if args.smoke else ''}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    judge_client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE)
    model, tok = load_student()

    scores = []
    for i, c in enumerate(cases):
        kp = c["knowledge_point"]
        diff = c["difficulty"]
        qtype = c.get("question_type", "解答题")

        # Step 1: 生成
        log(f"[{i+1}/{len(cases)}] {kp} {diff} {qtype} ...", end=" ")
        try:
            gen = generate(model, tok, kp, diff, qtype)
            log(f"OK ({gen.get('_gen_time', 0):.1f}s, {gen.get('_gen_tokens', 0)}t)")
        except Exception as e:
            log(f"ERROR gen: {e}")
            scores.append({"综合评分": 0, "错误": str(e), "test_case": c})
            continue

        # Step 2: 裁判
        log(f"      裁判...", end=" ")
        try:
            score = judge(judge_client, kp, diff, qtype, gen)
            score["test_case"] = c
            score["_gen"] = gen
            scores.append(score)
            log(f"综合={score.get('综合评分','?')} 匹配={score.get('知识点匹配度','?')} "
                f"难度={score.get('难度校准度','?')} 可解={score.get('可解性','?')}")
        except Exception as e:
            log(f"ERROR judge: {e}")
            scores.append({"综合评分": 0, "错误": str(e), "test_case": c, "_gen": gen})

    unload(model, tok)

    # 汇总
    report = agg(scores)
    report["label"] = "student_4b_lora"

    json_path = os.path.join(OUTPUT_DIR, "harness_report_student.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"📁 JSON: {json_path}")

    md_path = os.path.join(OUTPUT_DIR, "distill_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report(report, BASELINE_9B))
    log(f"📁 Markdown: {md_path}")

    print(f"\n{'='*60}")
    print(f"DISTILL EVAL COMPLETE")
    print(f"  Student solvable:  {report.get('可解率')}")
    print(f"  Student score:     {report.get('综合评分_平均')}")
    print(f"  Teacher solvable:  {BASELINE_9B['可解率']}")
    print(f"  Teacher score:     {BASELINE_9B['综合评分']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
