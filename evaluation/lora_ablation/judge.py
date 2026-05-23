"""
Harness 裁判：DeepSeek V4 三维度评分
- 知识点匹配度 (1-5)
- 难度校准度 (1-5)
- 可解性 (是/否)
"""
import json, os, sys
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEEPSEEK_API_KEY as DEEPSEEK_KEY, DEEPSEEK_API_BASE

llm = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_API_BASE)

# === 三维度 Judge Prompt ===
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
}}
"""


def judge(knowledge_point: str, target_difficulty: str, question_type: str, model_output: dict) -> dict:
    """用 DeepSeek V4 裁判一道题的三维度质量"""
    prompt = JUDGE_PROMPT.format(
        knowledge_point=knowledge_point,
        target_difficulty=target_difficulty,
        question_type=question_type,
        model_output=json.dumps(model_output, ensure_ascii=False, indent=2),
    )

    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw = response.choices[0].message.content

    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        return {"解析失败": raw}


def batch_judge(test_cases: list[dict], results: list[dict]) -> list[dict]:
    """批量裁判"""
    scores = []
    for case, result in zip(test_cases, results):
        score = judge(
            knowledge_point=case["knowledge_point"],
            target_difficulty=case["difficulty"],
            question_type=case.get("question_type", "解答题"),
            model_output=result,
        )
        score["test_case"] = case
        scores.append(score)
    return scores


def aggregate(scores: list[dict]) -> dict:
    """汇总指标"""
    if not scores:
        return {}
    kp_scores = [s.get("知识点匹配度", 0) for s in scores]
    diff_scores = [s.get("难度校准度", 0) for s in scores]
    solvable = sum(1 for s in scores if s.get("可解性") == "是")
    total_scores = [s.get("综合评分", 0) for s in scores]

    return {
        "样本数": len(scores),
        "知识点匹配度_平均": round(sum(kp_scores) / len(kp_scores), 2),
        "难度校准度_平均": round(sum(diff_scores) / len(diff_scores), 2),
        "可解率": f"{solvable}/{len(scores)} ({solvable/len(scores)*100:.0f}%)",
        "综合评分_平均": round(sum(total_scores) / len(total_scores), 2),
        "详细评分": scores,
    }


if __name__ == "__main__":
    # 测试
    result = judge(
        knowledge_point="勾股定理",
        target_difficulty="中等",
        question_type="解答题",
        model_output={
            "question": "已知直角三角形两条直角边分别为3和4，求斜边长度。",
            "solution": "根据勾股定理，c²=3²+4²=9+16=25，所以c=5。",
            "answer": "5"
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
