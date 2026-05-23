"""
Harness 评估框架
- 遍历测试用例 → 调出题 Agent → 调裁判 → 汇总报告
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
import time
from mcp import ClientSession
from mcp.client.sse import sse_client
from evaluation.judge import judge, aggregate

QUESTION_SERVER = "http://127.0.0.1:8765/sse"


async def generate_question(kp: str, difficulty: str, qtype: str) -> dict:
    """通过 MCP 调用出题 Agent"""
    async with sse_client(QUESTION_SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("generate_question", {
                "knowledge_point": kp,
                "difficulty": difficulty,
                "question_type": qtype,
            })
            return json.loads(result.content[0].text)


async def run_eval(test_cases: list[dict], label: str = "eval") -> dict:
    """
    运行完整评估管线：
    1. 逐题生成
    2. 逐题裁判
    3. 汇总指标
    """
    results = []
    scores = []
    start = time.time()

    print(f"\n{'='*60}")
    print(f"🧪 Harness 评估: {label} ({len(test_cases)} 题)")
    print(f"{'='*60}\n")

    for i, case in enumerate(test_cases):
        kp = case["knowledge_point"]
        diff = case["difficulty"]
        qtype = case.get("question_type", "解答题")

        # Step 1: 生成
        print(f"[{i+1}/{len(test_cases)}] 出题: {kp} {diff} {qtype} ...", end=" ", flush=True)
        try:
            gen_result = await generate_question(kp, diff, qtype)
            results.append(gen_result)
            print("✅", flush=True)
        except Exception as e:
            print(f"❌ {e}", flush=True)
            results.append({"error": str(e)})
            scores.append({"综合评分": 0, "错误": str(e)})
            continue

        # Step 2: 裁判
        print(f"        裁判中...", end=" ", flush=True)
        try:
            score = judge(kp, diff, qtype, gen_result)
            score["test_case"] = case
            scores.append(score)
            print(f"综合={score.get('综合评分', '?')} 匹配={score.get('知识点匹配度','?')} 难度={score.get('难度校准度','?')} 可解={score.get('可解性','?')}", flush=True)
        except Exception as e:
            print(f"❌ {e}", flush=True)
            scores.append({"综合评分": 0, "错误": str(e)})

        await asyncio.sleep(0.5)  # API 频率控制

    elapsed = time.time() - start

    # Step 3: 汇总
    report = aggregate(scores)
    report["label"] = label
    report["耗时_秒"] = round(elapsed, 1)
    report["生成速率_题分钟"] = round(len(test_cases) / elapsed * 60, 1)

    # 打印报告
    print(f"\n{'='*60}")
    print(f"📊 Harness 评估报告: {label}")
    print(f"{'='*60}")
    print(f"  样本数:          {report.get('样本数')}")
    print(f"  知识点匹配度:    {report.get('知识点匹配度_平均')}/5")
    print(f"  难度校准度:      {report.get('难度校准度_平均')}/5")
    print(f"  可解率:          {report.get('可解率')}")
    print(f"  综合评分:        {report.get('综合评分_平均')}/100")
    print(f"  耗时:            {report.get('耗时_秒')}秒")
    print(f"  生成速率:        {report.get('生成速率_题分钟')} 题/分钟")

    # 按知识点分组
    by_kp = {}
    for s in scores:
        kp = s.get("test_case", {}).get("knowledge_point", "unknown")
        by_kp.setdefault(kp, []).append(s.get("综合评分", 0))
    print(f"\n  按知识点:")
    for kp, kp_scores in sorted(by_kp.items()):
        avg = sum(kp_scores) / len(kp_scores) if kp_scores else 0
        print(f"    {kp}: {avg:.0f}/100 ({len(kp_scores)}题)")

    # 保存 JSON 报告
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"harness_report_{label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  📁 报告已保存: {out_path}")

    return report


async def main():
    from evaluation.tasks import TEST_CASES, SMOKE_TEST

    # 冒烟测试（6题）
    await run_eval(SMOKE_TEST, label="smoke")

    # 正式评估（30题）
    # await run_eval(TEST_CASES, label="full")


if __name__ == "__main__":
    asyncio.run(main())
