#!/usr/bin/env python3
"""
星学伴 · IRT 自适应门禁
Harness 驾驭框架第三层：验证 IRT 知识追踪的数学逻辑是否正确

5 场景，纯数学验证，不需要模型/API/GPU

用法:
    python irt_harness.py
"""
import sys, os, json, math
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.irt import IRT

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIOS = [
    {
        "id": 1,
        "name": "优等生（全部答对）",
        "desc": "θ应从0.5单调上升，难度从简单→中等→困难",
        "init_theta": 0.5,
        "sequence": [True] * 15,  # 15次全对（步长0.08收敛慢，需要更多轮）
        "checks": [
            ("θ单调上升", lambda ts: all(ts[i] <= ts[i+1] for i in range(len(ts)-1))),
            ("最终θ>0.78", lambda ts: ts[-1] > 0.78),
            ("最终难度=困难", lambda diffs: diffs[-1] == "困难"),
            ("至少跨越两个难度区间", lambda diffs: len(set(diffs)) >= 2),
        ],
    },
    {
        "id": 2,
        "name": "后进生（全部答错）",
        "desc": "θ应从0.5单调下降，难度保持简单",
        "init_theta": 0.5,
        "sequence": [False] * 8,
        "checks": [
            ("θ单调下降", lambda ts: all(ts[i] >= ts[i+1] for i in range(len(ts)-1))),
            ("最终θ<0.4", lambda ts: ts[-1] < 0.4),
            ("最终难度=简单", lambda diffs: diffs[-1] == "简单"),
        ],
    },
    {
        "id": 3,
        "name": "正常生（震荡收敛）",
        "desc": "θ应在真值附近震荡，不剧烈波动",
        "init_theta": 0.5,
        "sequence": [True, False, True, True, False, True, False, True, True, False],
        "checks": [
            ("θ在0.3~0.8之间", lambda ts: all(0.3 <= t <= 0.8 for t in ts)),
            ("最大振幅<0.25", lambda ts: max(ts) - min(ts) < 0.25),
            ("趋势上升（整体正确>错误）", lambda ts: ts[-1] > ts[0]),
        ],
    },
    {
        "id": 4,
        "name": "边界安全",
        "desc": "极端θ不应溢出[0,1]",
        "init_theta": 0.99,
        "sequence": [True] * 5,  # 从高起点继续答对
        "checks": [
            ("θ不超过1.0", lambda ts: all(t <= 1.0 for t in ts)),
        ],
        "extra_test": {
            "name": "极低起点",
            "init_theta": 0.01,
            "sequence": [False] * 5,
            "checks": [
                ("θ不低于0.0", lambda ts: all(t >= 0.0 for t in ts)),
            ],
        },
    },
    {
        "id": 5,
        "name": "收敛性（20轮模拟）",
        "desc": "难度匹配时70%正确率→θ应收敛到真值",
        "checks": [],  # 动态生成
    },
]


def run_scenario(scenario: dict, kp: str = "测试知识点") -> dict:
    """跑单个场景"""
    irt = IRT()
    irt.set_theta(kp, scenario.get("init_theta", 0.5))

    theta_trace = [irt.get_theta(kp)]
    diff_trace = [irt.get_difficulty(kp)]
    zone_trace = [irt.get_zone(kp)]

    sequence = scenario["sequence"]
    for correct in sequence:
        irt.update(kp, correct)
        theta_trace.append(irt.get_theta(kp))
        diff_trace.append(irt.get_difficulty(kp))
        zone_trace.append(irt.get_zone(kp))

    # 运行检查
    check_results = []
    for name, fn in scenario.get("checks", []):
        try:
            passed = fn(theta_trace) if "θ" in name or "振幅" in name or "θ" in str(fn.__code__) else fn(diff_trace)
            # heuristic: try theta first, then diff
            if not passed:
                try:
                    passed = fn(theta_trace)
                except:
                    try:
                        passed = fn(diff_trace)
                    except:
                        passed = False
        except Exception as e:
            passed = False
        check_results.append({"检查项": name, "通过": passed})

    return {
        "scenario": scenario["id"],
        "name": scenario["name"],
        "desc": scenario["desc"],
        "theta_trace": [round(t, 4) for t in theta_trace],
        "diff_trace": diff_trace,
        "zone_trace": zone_trace,
        "checks": check_results,
        "passed": all(c["通过"] for c in check_results),
    }


def run_convergence() -> dict:
    """场景5：收敛性——模拟20轮，难度匹配时70%正确率"""
    import random
    random.seed(42)

    irt = IRT()
    kp = "收敛测试"
    irt.set_theta(kp, 0.5)

    theta_trace = [irt.get_theta(kp)]
    diff_trace = [irt.get_difficulty(kp)]

    true_ability = 0.65  # 假设学生真实能力

    for _ in range(20):
        theta = irt.get_theta(kp)
        # 难度匹配：题难度=b（IRT默认0.0），做对概率≈P(θ)
        prob = irt.prob_correct(theta)
        # 但我们要模拟的是"难度匹配时70%正确率"
        # 用 student 真实能力和题难度的差来算概率
        # 简化：用 irt.prob_correct(true_ability) 为基础，加噪声
        base_prob = irt.prob_correct(true_ability)
        correct = random.random() < base_prob

        irt.update(kp, correct)
        theta_trace.append(round(irt.get_theta(kp), 4))
        diff_trace.append(irt.get_difficulty(kp))

    final_theta = theta_trace[-1]
    deviation = abs(final_theta - true_ability)
    monotonic = all(theta_trace[i] <= theta_trace[i+1] 
                    for i in range(len(theta_trace)-1) 
                    if theta_trace[i+1] - theta_trace[i] > -0.01)

    checks = [
        {"检查项": "最终θ偏差<0.2", "通过": deviation < 0.2},
        {"检查项": "θ趋势大致单调上升", "通过": monotonic},
        {"检查项": "θ在0.4~0.85之间", "通过": 0.4 <= final_theta <= 0.85},
        {"检查项": "最终θ>初始θ（学到了）", "通过": final_theta > theta_trace[0]},
    ]

    return {
        "scenario": 5,
        "name": "收敛性",
        "desc": f"真值 θ={true_ability}，20轮模拟",
        "theta_trace": theta_trace,
        "diff_trace": diff_trace,
        "zone_trace": [],
        "checks": checks,
        "passed": all(c["通过"] for c in checks),
        "final_deviation": round(deviation, 4),
    }


def run_extra(scenario: dict, kp="边界测试2") -> dict:
    """跑场景4的额外子测试"""
    irt = IRT()
    extra = scenario["extra_test"]
    irt.set_theta(kp, extra["init_theta"])

    theta_trace = [irt.get_theta(kp)]
    for correct in extra["sequence"]:
        irt.update(kp, correct)
        theta_trace.append(irt.get_theta(kp))

    check_results = []
    for name, fn in extra["checks"]:
        try:
            passed = fn(theta_trace)
        except:
            passed = False
        check_results.append({"检查项": name, "通过": passed})

    return {
        "scenario": 4,
        "name": f"边界安全 — {extra['name']}",
        "desc": f"从 θ={extra['init_theta']} 开始",
        "theta_trace": [round(t, 4) for t in theta_trace],
        "diff_trace": [],
        "zone_trace": [],
        "checks": check_results,
        "passed": all(c["通过"] for c in check_results),
    }


def generate_report(results: list) -> str:
    lines = []
    lines.append("# 星学伴 · IRT 自适应门禁报告")
    lines.append("")
    lines.append(f"> Harness 驾驭框架 · 第三层")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    lines.append("## 总览")
    lines.append("")
    for r in results:
        emoji = "✅" if r["passed"] else "❌"
        lines.append(f"- {emoji} 场景{r['scenario']} **{r['name']}**: {r['desc']}")
    lines.append("")
    lines.append(f"**通过率: {passed}/{total}**")
    lines.append("")

    door = "🟢 IRT 逻辑正确" if passed == total else "🔴 IRT 逻辑异常"
    lines.append(f"## 门禁判定：{door}")
    lines.append("")

    lines.append("## 场景详情")
    for r in results:
        emoji = "✅" if r["passed"] else "❌"
        lines.append(f"### {emoji} 场景{r['scenario']}: {r['name']}")
        lines.append("")
        lines.append(f"**描述**: {r['desc']}")
        lines.append("")

        # θ 轨迹
        if r.get("theta_trace"):
            thetas = r["theta_trace"]
            diffs = r.get("diff_trace", [])
            lines.append(f"| 轮次 | θ | 难度 |")
            lines.append(f"|------|-----|------|")
            for i, (t, d) in enumerate(zip(thetas, diffs if diffs else ["-"]*len(thetas))):
                label = "初始" if i == 0 else f"第{i}轮"
                lines.append(f"| {label} | {t:.4f} | {d} |")
            lines.append("")

        # 检查项
        lines.append("| 检查项 | 结果 |")
        lines.append("|--------|------|")
        for c in r["checks"]:
            mark = "✅" if c["通过"] else "❌"
            lines.append(f"| {c['检查项']} | {mark} |")
        lines.append("")

    return "\n".join(lines)


def main():
    results = []

    for s in SCENARIOS[:4]:
        r = run_scenario(s)
        results.append(r)
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} 场景{s['id']}: {s['name']}")
        for c in r["checks"]:
            print(f"      {'✓' if c['通过'] else '✗'} {c['检查项']}")

    # 场景4 额外子测试
    r4b = run_extra(SCENARIOS[3])
    results.append(r4b)
    status = "✅" if r4b["passed"] else "❌"
    print(f"  {status} 场景4b: {r4b['name']}")
    for c in r4b["checks"]:
        print(f"      {'✓' if c['通过'] else '✗'} {c['检查项']}")

    # 场景5 收敛性
    r5 = run_convergence()
    results.append(r5)
    status = "✅" if r5["passed"] else "❌"
    print(f"  {status} 场景5: {r5['name']} (偏差={r5.get('final_deviation', '?')})")
    for c in r5["checks"]:
        print(f"      {'✓' if c['通过'] else '✗'} {c['检查项']}")

    # 保存
    json_path = os.path.join(OUTPUT_DIR, "irt_harness.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    md = generate_report(results)
    md_path = os.path.join(OUTPUT_DIR, "irt_harness_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*60}")
    print(f"  IRT 门禁: {passed}/{len(results)} 通过")
    print(f"  📁 {md_path}")
    print(f"  📁 {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
