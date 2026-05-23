#!/usr/bin/env python3
"""
星学伴 · L2 Agent 决策门禁
Harness 驾驭框架第二层：验证教师 Agent 在关键教学场景下的决策是否正确

5 个场景，每个场景：
  → 注入预设学生消息
  → 获取教师回复（无工具模式，纯文本决策）
  → DeepSeek V4 双重判断（符合/不符合 + 详细分析）
  → 如未通过，AI 自动生成 Prompt 修改建议

用法:
    python l2_agent_harness.py          # 跑全部 5 场景
    python l2_agent_harness.py --scene 2  # 只跑场景2
"""
import json, os, sys, time
from datetime import datetime
from openai import OpenAI

# ══════════════════════════════════════════
#  配置
# ══════════════════════════════════════════
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE

DEEPSEEK_KEY = DEEPSEEK_API_KEY
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

judge_client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_API_BASE)
teacher_client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_API_BASE)

# ══════════════════════════════════════════
#  教师 System Prompt（与 teachers_agent.py 一致）
#  去掉 IRT 状态注入，L2 测的是纯教学决策能力
# ══════════════════════════════════════════
TEACHER_SYSTEM = """你是星学伴AI教师，进行苏格拉底式数学引导。

## ⚠️ 铁律：绝不直接解题

- 学生问任何题，你只能用提问来引导，禁止给出答案或解题步骤
- 禁止说「这道题考的是…」「解题思路是…」「首先…然后…」等直接讲解的开场白
- 禁止说「答案是…」「结果是…」「等于…」等任何形式的直接结论
- 你要做的唯一一件事：让学生自己思考、自己解出来

## 你的六个行动

1. **引导（默认）**：当学生展示一道题目，你的第一反应永远是提问，不是解答：
   - 开场提问：「这道题你目前有什么思路吗？」「你先试试看，卡在哪里了告诉我」
   - 学生卡住或迷茫时，用以下三种策略之一：
     - 反问启发：「你觉得下一步该怎么做？」「如果换成另一种方法呢？」
     - 子问题拆解：把综合题拆成 2-3 个小步骤，逐个引导
     - 类比迁移：「这题和你上次做过的 X 思路类似，能想起什么？」
   - **禁止**：直接分析题目结构、直接点出考点、直接讲解思路

2. **诊断**：学生连续 2 次答错或明显困惑 → **立即停止追问当前问题**，告诉学生"你的问题可能出在更早的 X 上，我们先复习一下"，并**主动提供一个更简单的子问题或基础练习**

3. **评估**：学生给出答案后 → 判断掌握程度，给予反馈

4. **确认**：学生答对后，先肯定，然后**仅邀请**学生选择是否继续挑战。禁止直接出下一题。例如：「你掌握得很好！想试试更难的变式吗？请告诉我。」

5. **出题**：确认学生掌握当前知识点后，生成变式题巩固

6. **换题**：连续引导无进展 → 主动建议换个知识点或休息

## 对话风格
- 温暖鼓励，但坚定引导：先肯定努力「你的思路方向是对的！」，再指出问题
- 禁止说「你错了」「不对」等打击性语言，改为「再想想」「换个角度看看」
- 禁止直接给出完整答案和解题步骤
- 每轮对话控制在 2-3 句话以内

## 当前场景
你正在教初中数学「{knowledge_point}」。学生之前的对话是为了引出下面的教学场景。"""

# ══════════════════════════════════════════
#  5 个教学场景
# ══════════════════════════════════════════
SCENARIOS = [
    {
        "id": 1,
        "name": "学生答对",
        "knowledge_point": "勾股定理",
        "context": "你刚出了一道题：直角三角形两直角边分别为3和4，求斜边长度。",
        "student_message": "根据勾股定理，3²+4²=9+16=25，所以斜边是5。",
        "expected": (
            "确认学生正确，简要总结要点（如'对的，勾股定理a²+b²=c²'），"
            "不继续追问、不质疑、不出新题。可以问'要不要来一道变式题？'但不能强制执行。"
        ),
        "pass_conditions": [
            "回复中确认了学生答案正确",
            "没有说'你再想想'、'你确定吗'、'还有呢'等质疑或追问",
            "没有直接给出新的完整题目（可以邀请但不强制）",
        ],
        "fail_indicators": [
            "质疑已正确的答案",
            "出了新题并要求学生立刻做",
            "完全忽略学生回答",
        ],
    },
    {
        "id": 2,
        "name": "学生答错",
        "knowledge_point": "勾股定理",
        "context": "你出了一道题：直角三角形两直角边分别为3和4，求斜边长度。",
        "student_message": "我觉得斜边是7，因为3+4=7。",
        "expected": (
            "不直接给出正确答案5。用引导方式让学生自己发现错误："
            "比如'勾股定理说的是平方和，不是直接相加哦。3²+4²等于多少？'"
            "或者'回忆一下勾股定理的公式，a²+b²=c²，试试代入？'"
        ),
        "pass_conditions": [
            "没有直接给出正确答案'5'或'斜边是5'",
            "使用了引导性语言（提示、反问、拆解步骤）",
            "指向了勾股定理的正确公式（平方和而非直接相加）",
        ],
        "fail_indicators": [
            "直接给出了正确答案",
            "说了'你错了'或'不对'等打击性语言",
            "没有给出任何引导，只说了'再想想'敷衍",
        ],
    },
    {
        "id": 3,
        "name": "连续两次答错",
        "knowledge_point": "一元二次方程",
        "context": (
            "你出了一道题：解方程 x²-5x+6=0。"
            "学生第一次答：x=5。你说：'再试试，代入验证一下？'"
            "学生第二次答：x=1和x=6？"
        ),
        "student_message": "x=1和x=6？",
        "expected": (
            "连续两次答错后，教师应降低难度：不再直接追问原题，"
            "而是给出一个更简单的基础练习或换个角度。"
            "比如'我们先从简单的开始，x²-4=0 你试试？'"
            "或者'你记得因式分解的方法吗？我们先复习一下'"
        ),
        "pass_conditions": [
            "没有继续追问原题答案",
            "降低了难度（给了更简单的子问题或换了角度）",
            "或主动提出先复习基础知识",
        ],
        "fail_indicators": [
            "继续追问'再想想？'而不给新的引导",
            "直接给出原题正确答案",
            "完全跳过、说'我们做下一题'而不解决困惑",
        ],
    },
    {
        "id": 4,
        "name": "学生说不会",
        "knowledge_point": "相似三角形",
        "context": "你出了一道题：已知△ABC∽△DEF，AB=6，DE=9，BC=4，求EF的长度。",
        "student_message": "老师，我不会做这道题……",
        "expected": (
            "不给答案，不给完整解题步骤。用启发式追问或子问题拆解引导："
            "比如'还记得相似三角形的对应边成比例吗？你能写出比例式吗？'"
            "或者'我们先找到对应边，AB对应DE，那BC对应谁？'"
        ),
        "pass_conditions": [
            "没有直接给出答案或完整解题步骤",
            "使用了启发式追问（>=1个问句）",
            "给出了至少一步具体的引导方向",
        ],
        "fail_indicators": [
            "直接给出了完整解题过程和答案",
            "只说'没关系，我再讲一遍'然后重讲（教学而非引导）",
            "放弃引导，说'那我们换一题'",
        ],
    },
    {
        "id": 5,
        "name": "学生部分正确",
        "knowledge_point": "概率",
        "context": "你出了一道题：袋中有3红2白，连续取出2个球不放回，求两次都取到红球的概率。",
        "student_message": "第一次取红球是3/5，第二次取红球也是3/5，所以乘起来是9/25。",
        "expected": (
            "先肯定思路方向（'你的思路是对的，乘法原理！'），"
            "再指出具体错误点（'但第二次取的时候，不放回，袋子里的球数变了吗？'），"
            "让学生自己修正。"
        ),
        "pass_conditions": [
            "先肯定或鼓励了学生的思路",
            "然后指出了具体错误位置",
            "没有直接给出正确答案'3/10'",
        ],
        "fail_indicators": [
            "直接给出正确答案",
            "只说'不对'或'你再想想'，没有指出具体问题",
            "只表扬不纠错（忽略了错误）",
        ],
    },
]


# ══════════════════════════════════════════
#  裁判 Prompt
# ══════════════════════════════════════════
JUDGE_PROMPT = """你是AI教学行为评估专家。给定一个教学场景、学生消息和教师回复，判断教师行为是否符合期望。

## 场景信息
- 场景名称：{scene_name}
- 知识点：{knowledge_point}
- 背景：{context}

## 学生消息
{student_message}

## 教师回复
{teacher_reply}

## 期望行为
{expected}

## 通过条件
{pass_conditions}

## 失败标志
{fail_indicators}

## 输出（严格 JSON）
{{
  "判定": "通过 / 未通过 / 部分通过",
  "分析": "2-3句话分析教师行为，具体指出哪个通过条件满足/未满足",
  "关键证据": "引用教师回复中的具体语句",
  "得分": 1-5 的整数（5=完美符合，1=完全不符合）
}}"""


FIX_PROMPT = """你是AI教学Prompt优化专家。一个教师Agent在以下教学场景中行为不符合期望。

## 当前 System Prompt
```\n{system_prompt}\n```

## 失败场景
- 场景名称：{scene_name}
- 学生消息：{student_message}
- 教师实际回复：{teacher_reply}
- 判定：{judgment}
- 问题分析：{analysis}

## 任务
请给出**精确、最小化**的 Prompt 修改建议。格式如下：

1. **要改的位置**：指出 System Prompt 中哪一句/哪一段需要修改（引用原文）
2. **建议改成**：给出新的措辞
3. **修改理由**：一句话解释为什么这样改能解决当前问题

注意：
- 只改动最少文字，不重写整个 Prompt
- 改动要能精确防止当前错误，同时不影响其他场景
- 用中文
"""


# ══════════════════════════════════════════
#  核心逻辑
# ══════════════════════════════════════════

def get_teacher_reply(scenario: dict) -> str:
    """调用教师 Agent 获取回复（无工具模式，纯文本决策）"""
    system = TEACHER_SYSTEM.format(knowledge_point=scenario["knowledge_point"])

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"[教学背景] {scenario['context']}"},
        {"role": "assistant", "content": "好的，我准备好了。请学生回答。"},
        {"role": "user", "content": scenario["student_message"]},
    ]

    resp = teacher_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.5,
        max_tokens=512,
    )
    return resp.choices[0].message.content


def judge(scenario: dict, teacher_reply: str) -> dict:
    """裁判教师行为"""
    prompt = JUDGE_PROMPT.format(
        scene_name=scenario["name"],
        knowledge_point=scenario["knowledge_point"],
        context=scenario["context"],
        student_message=scenario["student_message"],
        teacher_reply=teacher_reply,
        expected=scenario["expected"],
        pass_conditions="\n".join(f"- {c}" for c in scenario["pass_conditions"]),
        fail_indicators="\n".join(f"- {c}" for c in scenario["fail_indicators"]),
    )

    resp = judge_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content
    # 清理 markdown
    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"判定": "解析失败", "分析": raw, "关键证据": "", "得分": 0}


def suggest_fix(scenario: dict, teacher_reply: str, judgment: dict) -> str:
    """AI 自动生成 Prompt 修改建议"""
    prompt = FIX_PROMPT.format(
        system_prompt=TEACHER_SYSTEM,
        scene_name=scenario["name"],
        student_message=scenario["student_message"],
        teacher_reply=teacher_reply,
        judgment=judgment.get("判定", "未通过"),
        analysis=judgment.get("分析", ""),
    )

    resp = judge_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def run_scenario(scenario: dict) -> dict:
    """跑单个场景"""
    print(f"\n{'='*60}")
    print(f"  场景 {scenario['id']}: {scenario['name']}")
    print(f"  知识点: {scenario['knowledge_point']}")
    print(f"{'='*60}")

    # Step 1: 获取教师回复
    print(f"  🎓 学生: {scenario['student_message'][:80]}...")
    reply = get_teacher_reply(scenario)
    print(f"  👩‍🏫 教师: {reply[:120]}...")

    # Step 2: 裁判
    print(f"  ⚖️  裁判中...", end=" ")
    judgment = judge(scenario, reply)
    verdict = judgment.get("判定", "?")
    score = judgment.get("得分", 0)
    print(f"{verdict} (得分={score})")

    result = {
        "scenario": scenario,
        "teacher_reply": reply,
        "judgment": judgment,
    }

    # Step 3: 如果未通过，生成 Prompt 修改建议
    if verdict != "通过":
        print(f"  🔧 生成 Prompt 修改建议...", end=" ")
        fix = suggest_fix(scenario, reply, judgment)
        result["fix_suggestion"] = fix
        print("✓")
        print(f"\n  📝 修改建议:\n{fix[:500]}...")
    else:
        print(f"  ✅ 通过！无需修改")

    return result


def generate_report(results: list) -> str:
    """生成 Markdown 报告"""
    lines = []
    lines.append("# 星学伴 · L2 Agent 决策门禁报告")
    lines.append("")
    lines.append(f"> Harness 驾驭框架 · 第二层")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> 教学模型：DeepSeek V4")
    lines.append("")

    total = len(results)
    passed = sum(1 for r in results if r["judgment"].get("判定") == "通过")
    partial = sum(1 for r in results if r["judgment"].get("判定") == "部分通过")
    failed = total - passed - partial
    avg_score = sum(r["judgment"].get("得分", 0) for r in results) / total

    lines.append("## 总览")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 场景总数 | {total} |")
    lines.append(f"| 通过 | {passed} |")
    lines.append(f"| 部分通过 | {partial} |")
    lines.append(f"| 未通过 | {failed} |")
    lines.append(f"| 通过率 | {passed}/{total} ({passed/total*100:.0f}%) |")
    lines.append(f"| 平均得分 | {avg_score:.1f}/5 |")
    lines.append("")

    door = "🟢 批准上线" if passed >= total else "🔴 门禁拦截" if passed < 3 else "🟡 需修复后重测"
    lines.append(f"## 门禁判定：{door}")
    lines.append("")

    lines.append("## 场景详情")
    lines.append("")

    for r in results:
        s = r["scenario"]
        j = r["judgment"]
        emoji = "✅" if j.get("判定") == "通过" else "⚠️" if j.get("判定") == "部分通过" else "❌"
        lines.append(f"### {emoji} 场景 {s['id']}: {s['name']}")
        lines.append("")
        lines.append(f"| 项目 | 内容 |")
        lines.append(f"|------|------|")
        lines.append(f"| 知识点 | {s['knowledge_point']} |")
        lines.append(f"| 学生消息 | {s['student_message']} |")
        lines.append(f"| 教师回复 | {r['teacher_reply'][:200]} |")
        lines.append(f"| 判定 | {j.get('判定', '?')} |")
        lines.append(f"| 得分 | {j.get('得分', '?')}/5 |")
        lines.append(f"| 分析 | {j.get('分析', '')} |")
        lines.append(f"| 证据 | {j.get('关键证据', '')} |")
        lines.append("")

        if "fix_suggestion" in r:
            lines.append(f"**📝 Prompt 修改建议：**")
            lines.append("")
            lines.append(r["fix_suggestion"])
            lines.append("")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="L2 Agent 决策门禁")
    parser.add_argument("--scene", type=int, help="只跑指定场景")
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.scene:
        scenarios = [s for s in SCENARIOS if s["id"] == args.scene]
        if not scenarios:
            print(f"❌ 场景 {args.scene} 不存在")
            return

    results = []
    for s in scenarios:
        r = run_scenario(s)
        results.append(r)

    # 存 JSON
    json_path = os.path.join(OUTPUT_DIR, "l2_agent_harness.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📁 JSON: {json_path}")

    # 存 Markdown
    md = generate_report(results)
    md_path = os.path.join(OUTPUT_DIR, "l2_agent_harness_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"📁 报告: {md_path}")

    # 摘要
    passed = sum(1 for r in results if r["judgment"].get("判定") == "通过")
    print(f"\n{'='*60}")
    print(f"  L2 门禁: {passed}/{len(results)} 通过")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
