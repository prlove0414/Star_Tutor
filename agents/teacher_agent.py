"""
教师 Agent · 多轮苏格拉底引导 (MCP 多客户端)
- 深度对话引导：反问/类比/拆解 → 诊断根因 → 评估掌握 → 出变式题
- IRT 知识追踪驱动难度 + 自动判断出题时机
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import hashlib
import re as _re
import asyncio
import datetime as _dt
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client
from agents.irt import IRT

# === L4 模块级日锁计数器（跨会话共享）===
_daily_lock_count = 0
_daily_lock_date = ""  # "YYYY-MM-DD"

# === 配置 ===
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEEPSEEK_API_KEY as DEEPSEEK_KEY, DEEPSEEK_API_BASE, VISION_API_KEY, VISION_API_BASE, VISION_MODEL_NAME

MCP_SERVERS = {
    "question":    "http://127.0.0.1:8765/sse",
    "diagnosis":   "http://127.0.0.1:8766/sse",
    "evaluation":  "http://127.0.0.1:8767/sse",
    "vision":      "http://127.0.0.1:8768/sse",
    "figure":      "http://127.0.0.1:8769/sse",
}
TOOL_ROUTING = {
    "generate_question":    "question",
    "trace_prerequisites":  "diagnosis",
    "find_kp":              "diagnosis",
    "evaluate_answer":      "evaluation",
    "recognize_question":   "vision",
    "generate_figure":      "figure",
    "solve_problem":        "_internal",  # 教师内部求解，不走 MCP
}

# === 工具定义 ===
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_question",
            "description": "生成一道数学题。用于教学最后阶段——学生掌握后出变式题巩固。",
            "parameters": {
                "type": "object",
                "properties": {
                    "knowledge_point": {"type": "string", "description": "知识点"},
                    "difficulty": {"type": "string", "enum": ["简单", "中等", "困难"]},
                    "question_type": {"type": "string", "enum": ["选择题", "填空题", "解答题"], "default": "解答题"},
                },
                "required": ["knowledge_point", "difficulty"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trace_prerequisites",
            "description": "追溯知识点的前置学习链。当学生反复在同一知识点犯错时调用，用于找到根因。",
            "parameters": {
                "type": "object",
                "properties": {
                    "knowledge_point": {"type": "string"},
                },
                "required": ["knowledge_point"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_kp",
            "description": "在知识图谱中搜索知识点。",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_answer",
            "description": "三维度评估学生回答：思路正确性、结果正确性、表述完整性。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "student_answer": {"type": "string"},
                    "correct_answer": {"type": "string"},
                    "knowledge_point": {"type": "string"},
                },
                "required": ["question", "student_answer", "correct_answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recognize_question",
            "description": "识别学生上传的题目图片。当学生发送图片而非文字时调用，将图片解析为题目文本+知识点+难度+题型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "图片文件的绝对路径"},
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_figure",
            "description": "为几何题或函数题生成配图。出完题后若题目涉及几何图形、函数图像等需要配图的内容，调用此工具生成 PNG 图片。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "题目完整文本"},
                    "knowledge_point": {"type": "string", "description": "知识点"},
                    "question_type": {"type": "string", "enum": ["选择题", "填空题", "解答题"], "default": "解答题"},
                },
                "required": ["question", "knowledge_point"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_problem",
            "description": "教师内部求解工具。收到新题目后必须先调用此工具，获取正确答案和关键步骤，确保后续引导方向正确。仅在首次见到题目时调用一次。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "题目完整文本"},
                    "knowledge_point": {"type": "string", "description": "考察的知识点（可选）"},
                },
                "required": ["question"],
            },
        },
    },
]

# === 苏格拉底引导 System Prompt ===
TEACHER_SYSTEM = """你是星学伴AI教师，进行苏格拉底式数学引导。

## ⚠️ 铁律：先解题，再引导

- 收到任何新题目（文字或图片识别结果）后，第一件事是调用 solve_problem 自己把题做出来
- solve_problem 是异步后台任务——调用后你会立即收到"求解中"的回复，**此时你应该先和学生展开对话**：不要只问一句「有什么思路」，要逐步引导他说出对题目的理解——给了哪些条件？问的是什么？打算从哪入手？那一步具体怎么做？让他多说，直到答案就绪。
- ⚠️ 不要在 solve_problem 返回"求解中"后反复调用——后台正在计算，下轮对话答案会自动出现
- 永远不要在自己不明白答案的情况下去引导学生——那会导致胡说八道

## 🚫🚫🚫 绝对禁止透露 solve_problem 的结果 🚫🚫🚫

solve_problem 返回的答案是你内部教学的参考，**绝对不能**在对话中透露给学生。这也包括：

- ❌ 不要把解题过程、计算草稿、代数展开、配方法等内部推算输出到回复中
- ❌ 不要自言自语"让我重新算一下"然后输出计算过程——这些应该在你脑子里完成
- ❌ 不要在回复中展示你正在求解、验证、或重新计算的过程
- ✅ 你的回复只包含：引导提问、给学生看的提示、或对学生的评价。干净利落。

禁止行为：
- ❌ 直接说出答案：「这道题的答案是 42」
- ❌ 先公布答案再讲解：「答案是 13，接下来我给你讲讲怎么做」
- ❌ 在引导中暴露答案：「注意这里要用到勾股定理，答案是 13」
- ❌ 用答案来验证学生：「你说等于 13，我们来看是不是对的……没错答案是 13」
- ❌ 任何形式让学生在对话中看到 solve_problem 返回的具体数值

允许行为：
- ✅ 调用 evaluate_answer 时把答案传给 correct_answer 参数（这是内部工具调用，学生看不到）
- ✅ 用提问引导：「你再算一遍看看？」「你确定这个结果吗？」
- ✅ 学生答对后肯定：「很好，你做对了！」（但不重复答案）

简单说：**solve_problem 的结果只能出现在 evaluate_answer 的参数里，不能出现在任何一段回复文字中。**

## ⚠️ 铁律：默认模式——只提问引导

在学生能够回答你的引导问题时：
- 你只能用提问来引导，禁止给出答案或解题步骤
- 禁止说「这道题考的是…」「解题思路是…」「首先…然后…」等直接讲解的开场白
- 禁止说「答案是…」「结果是…」「等于…」等任何形式的直接结论
- 你要做的：让学生自己思考、自己解出来

## ⚠️ 提示模式——当学生持续卡住时自动切换

当学生连续 3 次无法回答你的引导问题（说「不知道」「不理解」「不会」「没有思路」或给出无关回答）时，**自动切换到提示模式**：

提示模式下你可以做的：
- 给出具体的关键线索：「注意 AB 是直径，直径所对的圆周角是 90°」
- 展示部分解题步骤：「第一步：连接 AC，因为 AB 是直径，所以 ∠ACB = 90°」
- 直接告诉学生当前卡住的那一步该怎么做
- 提示模式下一次给 1-2 个关键信息，不要一次性全讲完
- 给了提示后，仍然留一步让学生自己完成

提示模式下仍然禁止的：
- 一次性给出完整答案
- 把整道题的完整解题过程全讲完
- 说「我直接告诉你答案吧」

## ⚠️ 知识范围：严格限定初中数学

你只能使用以下初中数学知识，**绝对禁止**使用任何高中或更高级的内容：

**✅ 可用：** 三角形性质/全等/相似、勾股定理、圆的性质（圆周角/圆心角/切线）、四边形性质、一元二次方程、
一次/二次/反比例函数、平行线/垂直线、坐标系基础（两点距离/中点）、面积体积公式、

**基础锐角三角函数**（仅限 sin/cos/tan 在直角三角形中的定义，30°/45°/60° 特殊角的值）

**❌ 禁止：** 余弦定理、正弦定理、向量、复数、导数/微积分、三角函数恒等式、解析几何（直线方程/圆的方程）、
立体几何、海伦公式、二倍角公式、任意角三角函数、坐标系旋转/平移公式

几何题必须使用纯几何方法（全等/相似/辅助线），不能用坐标法或三角法代替几何推理。

## 你的六个行动

0. **先解题（新增）**：收到新题目 → 立即调用 solve_problem，获取正确答案和关键步骤。solve_problem 的结果会告诉你答案和关键步骤，请牢记——后续所有引导都必须指向这个正确答案。
   ⚠️ 如果消息以 `[系统]` 开头，说明图片已被识别，直接对识别出的题目文本调用 solve_problem。
   ⚠️ 只对新出现的题目调用 solve_problem，同一道题的后续交互不需要重复调用。

1. **识别图片**：如果学生发送了题目图片而非文字，**必须先调用 recognize_question** 识别，拿到题目文本后再调用 solve_problem 求解。
   ⚠️ 如果消息以 `[系统]` 开头，说明图片已经被 Vision 自动识别过了，**不要再次调用 recognize_question**，直接对识别出的题目调用 solve_problem。

2. **引导（默认模式）**：当学生展示一道题目并求解完成后，你的第一反应永远是提问，不是解答：
   - 开场提问：「这道题你目前有什么思路吗？」「你先试试看，卡在哪里了告诉我」
   - 学生卡住或迷茫时，用以下三种策略之一：
     - 反问启发：「你觉得下一步该怎么做？」「如果换成另一种方法呢？」
     - 子问题拆解：把综合题拆成 2-3 个小步骤，逐个引导
     - 类比迁移：「这题和你上次做过的 X 思路类似，能想起什么？」
   - **禁止**：直接分析题目结构、直接点出考点、直接讲解思路

3. **提示模式（自动触发）**：当学生对同一道题连续 3 次无法回答时，自动切换：
   - 给出 1-2 个关键线索或展示一步解题过程
   - 给了提示后留一步让学生完成
   - 如果给提示后学生仍然无法推进 → 继续给下一步提示 → 最多展示 3 步后建议换题

4. **诊断**：学生连续 2 次评估为"未掌握" → **立即停止追问当前问题**，调用 trace_prerequisites 追溯薄弱点 → 告诉学生"你的问题可能出在更早的 X 上，我们先复习一下"

5. **评估**：学生给出答案后 → 调用 evaluate_answer 判断掌握程度。评估结果会联动 IRT 知识追踪自动调整难度

6. **确认 + 出题**：学生答对后，先肯定，然后**仅邀请**学生选择是否继续挑战。禁止直接出下一题。
   ⚠️ 绝对禁止自己编数学题——所有题目必须通过 generate_question 工具生成
   ⚠️ 出题时优先选择「困难」难度——学生能答对当前题说明已有一定掌握，应推高难度
   ⚠️ 出完题后，若题目涉及几何图形、函数图像等需要配图的内容，**调用 generate_figure** 生成配图

## 对话风格
- 温暖鼓励，但坚定引导：先肯定努力「你的思路方向是对的！」，再指出问题
- 禁止说「你错了」「不对」等打击性语言，改为「再想想」「换个角度看看」
- 每轮对话控制在 2-3 句话以内
- 遇到数学公式用 $$ 包裹，确保渲染正确
- 提示模式下可以说「这里给你一个提示：……」

## 当前学生状态
{irt_status}
卡住计数：{stuck_count}/3 次
当前模式：{mode}
"""


class TeacherAgent:
    def __init__(self):
        self.llm = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_API_BASE)
        self.irt = IRT()
        self.turn = 0
        self.consecutive_correct = 0
        self.consecutive_wrong = 0
        self.current_kp = ""  # 当前讨论的知识点
        self.messages = []    # 完整对话历史
        self.stuck_count = 0  # 同一道题连续卡住次数
        self.current_problem_hash = ""  # 当前题目标识（用于判断是否换题）

        # === L3 解答正确性门禁 ===
        self._current_answer = ""  # 当前题目正确答案
        self._last_question = ""    # 当前题目文本
        self._eval_done_this_turn = False  # 本轮是否已程序化评估
        self._answer_tracker = {}  # {question_key: {"wrong": {norm_ans: count}, "l3_count": int, "anomaly": bool}}

        # === L4 学生行为管控 ===
        self._anomalies = []         # 异常记录 [{question, student_answer, correct_answer, kp, level, action}]
        self._session_locked = False  # 会话是否已锁定
        self._l4_lock_reason = ""    # 锁定原因
        self._l4_lock_reply = ""     # 锁定时的回复（后续用同一句）
        self._non_learning_count = 0 # 非学习内容计数（同一会话累计）

        # 后台异步求解（消除学生等待感）
        self._pending_solve_task = None   # asyncio.Task
        self._pending_solve_ready = False # 后台求解是否已完成

        # 第二求解模型 (Qwen via DashScope)
        self._solver2 = None
        try:
            from config import VISION_API_KEY, VISION_API_BASE
            if VISION_API_KEY and VISION_API_BASE:
                self._solver2 = OpenAI(api_key=VISION_API_KEY, base_url=VISION_API_BASE)
                print("  🔬 L3 第二求解模型就绪 (DashScope)", flush=True)
        except Exception:
            pass

    async def _call_mcp(self, tool_name: str, arguments: dict) -> dict:
        server = TOOL_ROUTING.get(tool_name, "question")
        url = MCP_SERVERS[server]
        try:
            async with sse_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    raw = result.content[0].text
                    return json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            raw_str = str(e)
            print(f"  ⚠️ MCP [{tool_name}] 调用失败: {raw_str[:200]}", flush=True)
            return {"error": raw_str, "tool": tool_name}

    # ===== L3 解答正确性门禁 =====

    def _question_key(self, question: str) -> str:
        """生成题目标识（取前 200 字符的 MD5）"""
        return hashlib.md5(question.strip()[:200].encode()).hexdigest()[:12]

    def _normalize_answer(self, answer: str) -> str:
        """归一化答案用于比较"""
        s = answer.strip().lower()
        s = _re.sub(r'(\d+)\s*/\s*(\d+)', lambda m: f"{int(m.group(1))/int(m.group(2)):.4g}", s)
        s = _re.sub(r'\s+', '', s)
        return s

    def _answers_match(self, a: str, b: str) -> bool:
        """判断两个答案是否一致"""
        return self._normalize_answer(a) == self._normalize_answer(b)

    async def _solve_single(self, question: str, knowledge_point: str,
                            system_prompt: str, extra_context: str = "",
                            client=None, model="deepseek-chat") -> dict:
        """单次求解调用"""
        if client is None:
            client = self.llm
        kp_hint = f"\n知识点提示：{knowledge_point}" if knowledge_point else ""
        prompt = f"题目：{question}{kp_hint}"
        if extra_context:
            prompt += f"\n\n{extra_context}"
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            raw = response.choices[0].message.content
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "").strip()
            result = json.loads(raw)
            result["_model"] = model
            return result
        except Exception as e:
            return {"answer": "求解失败", "key_steps": [], "knowledge_point": knowledge_point,
                    "error": str(e), "_model": model}

    async def _solve_problem(self, question: str, knowledge_point: str = "",
                             student_context: str = "") -> dict:
        """L3 解答正确性：多路径求解 + 答案验证
        路径 A: 纯几何推理   路径 B: 代数/坐标辅助
        分歧 → 互知重试 → Qwen 仲裁 → 答案验证"""
        PATH_A = """你是初中数学解题助手。请用纯几何推理方法解出以下题目：
1. 最终答案（简明扼要）
2. 关键解题步骤（3-5 步，每步一句话）

要求：
- 严格使用初中数学知识，禁止高中及以上方法
- 几何题用纯几何方法（全等/相似/辅助线），不用坐标法
- ⚠️ 必做验证：算出答案后代入原题逐条检查，验证不通过必须重新计算
- 输出纯 JSON：{"answer": "...", "key_steps": ["步骤1", "步骤2", ...], "knowledge_point": "涉及的知识点"}"""

        PATH_B = """你是初中数学解题助手。请用代数方法或坐标法解出以下题目：
1. 最终答案（简明扼要）
2. 关键解题步骤（3-5 步，每步一句话）

要求：
- 几何题设未知数、列方程，用代数运算替代几何推理
- 函数题直接代数运算，不做几何分析
- ⚠️ 必做验证：算出答案后代入原题逐条检查，验证不通过必须重新计算
- 输出纯 JSON：{"answer": "...", "key_steps": ["步骤1", "步骤2", ...], "knowledge_point": "涉及的知识点"}"""

        ctx = f"⚠️ {student_context}" if student_context else ""

        # Round 1: 双路径并行
        r1a, r1b = await asyncio.gather(
            self._solve_single(question, knowledge_point, PATH_A, ctx),
            self._solve_single(question, knowledge_point, PATH_B, ctx),
        )
        a_ans = r1a.get("answer", ""); b_ans = r1b.get("answer", "")
        if self._answers_match(a_ans, b_ans):
            print(f"  🧮 L3 双路径一致 ✓", flush=True)
            return await self._verify_answer(question, knowledge_point, r1a)

        # Round 2: 互相知晓差异后重试
        print(f"  ⚠️ L3 路径分歧: A={a_ans[:40]} ≠ B={b_ans[:40]} → 重试", flush=True)
        ctx_a = f"{ctx}\n⚠️ 路径B（代数/坐标法）得到答案：「{b_ans}」，与你的答案不同。请逐行检查你的推理，找出可能出错的地方，确保每一步都没有遗漏条件。"
        ctx_b = f"{ctx}\n⚠️ 路径A（纯几何法）得到答案：「{a_ans}」，与你的答案不同。请逐行检查你的推理，找出可能出错的地方，确保每一步都没有遗漏条件。"
        r2a, r2b = await asyncio.gather(
            self._solve_single(question, knowledge_point, PATH_A, ctx_a),
            self._solve_single(question, knowledge_point, PATH_B, ctx_b),
        )
        r2a_ans = r2a.get("answer", ""); r2b_ans = r2b.get("answer", "")
        if self._answers_match(r2a_ans, r2b_ans):
            print(f"  🧮 L3 重试后一致 ✓", flush=True)
            return await self._verify_answer(question, knowledge_point, r2a)

        # Round 3: 第二模型仲裁 (Qwen)
        if self._solver2:
            print(f"  🔬 L3 引入第二模型仲裁...", flush=True)
            qwen_ctx = f"路径A（纯几何）答案：「{r2a_ans}」\n路径B（代数）答案：「{r2b_ans}」\n请独立求解，不要偏向任何一方。{ctx}"
            qwen_result = None
            for qwen_model in ["qwen3-plus", "qwen-plus"]:
                r3 = await self._solve_single(
                    question, knowledge_point, PATH_A, qwen_ctx,
                    client=self._solver2, model=qwen_model
                )
                if "error" not in r3:
                    qwen_result = r3
                    break
            if qwen_result:
                qwen_ans = qwen_result.get("answer", "")
                if self._answers_match(qwen_ans, r2a_ans):
                    print(f"  🧮 Qwen 支持 A: {qwen_ans[:40]}", flush=True)
                    return await self._verify_answer(question, knowledge_point, r2a)
                elif self._answers_match(qwen_ans, r2b_ans):
                    print(f"  🧮 Qwen 支持 B: {qwen_ans[:40]}", flush=True)
                    return await self._verify_answer(question, knowledge_point, r2b)
                else:
                    print(f"  ⚠️ Qwen 独有答案: {qwen_ans[:40]} → 采用", flush=True)
                    qwen_result["_model"] = "qwen-arbiter"
                    return await self._verify_answer(question, knowledge_point, qwen_result)

        # 无第二模型或仲裁失败 → 取路径 A（纯几何更可靠）
        print(f"  ⚠️ L3 最终未收敛，采用路径 A: {r2a_ans[:40]}", flush=True)
        return await self._verify_answer(question, knowledge_point, r2a)

    async def _verify_answer(self, question: str, knowledge_point: str, result: dict) -> dict:
        """L3 答案验证：SymPy 等式验证 + LLM 代入验证"""
        answer = result.get("answer", "")

        # SymPy 等式验证
        try:
            import sympy as sp
            if "=" in answer:
                parts = answer.split("=")
                if len(parts) == 2:
                    left = sp.sympify(parts[0].strip())
                    right = sp.sympify(parts[1].strip())
                    if sp.simplify(left - right) == 0:
                        print(f"  ✅ L3 SymPy 通过", flush=True)
        except Exception:
            pass  # SymPy 非必须，跳过

        # LLM 代入验证
        VERIFY_PROMPT = """你是数学验证助手。把答案代回原题，逐条件检查是否全部满足。
输出 JSON：{"valid": true/false, "check_detail": "逐条结果", "corrected_answer": "如错误，给出正确答案"}
⚠️ 任一条件不满足 → valid 必须是 false。不要因为答案"看起来合理"就说 valid。"""
        vp = f"题目：{question}\n待验证答案：{answer}"
        try:
            resp = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": VERIFY_PROMPT}, {"role": "user", "content": vp}],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "").strip()
            verify = json.loads(raw)
            if not verify.get("valid", True):
                corrected = verify.get("corrected_answer", "")
                if corrected:
                    print(f"  🔄 L3 纠正: {answer[:40]} → {corrected[:40]}", flush=True)
                    result["answer"] = corrected
                    result["_corrected_by"] = "L3"
                else:
                    print(f"  ⚠️ L3 失败无纠正，保留原答案", flush=True)
            else:
                print(f"  ✅ L3 代入验证通过", flush=True)
        except Exception as e:
            print(f"  ⚠️ L3 验证异常: {e}", flush=True)

        print(f"  🧮 最终答案: {result.get('answer', '?')[:50]}", flush=True)
        return result

    def _is_stuck(self, student_msg: str) -> bool:
        """判断学生消息是否表示卡住/困惑"""
        stuck_keywords = ["不知道", "不会", "不理解", "不懂", "没有思路", "卡住", "不会做",
                         "不会写", "做不出", "做不来", "想不出", "想不到", "你给我提示",
                         "教我", "帮帮我", "太难了", "放弃", "算不出来", "不会算"]
        return any(kw in student_msg for kw in stuck_keywords)

    def _parse_xml_tool_call(self, text: str):
        """从 LLM 文本中解析 XML 格式的工具调用。支持标准 <invoke> 和 DeepSeek ||DSML|| 格式。返回 (tool_name, tool_args) 或 (None, None)"""
        import re
        # 格式1: <invoke name="xxx">...</invoke>
        m = re.search(r'<invoke\s+name="([^"]+)"[^>]*>(.*?)</invoke>', text, re.DOTALL)
        # 格式2: ||DSML||invoke name="xxx">...（DeepSeek DSML 格式）
        if not m:
            m = re.search(r'\|\|DSML\|\|invoke\s+name="([^"]+)"[^>]*>(.*?)\|\|DSML\|\|/invoke>', text, re.DOTALL)
        if not m:
            return None, None
        tool_name = m.group(1)
        params_str = m.group(2)
        args = {}
        # 标准格式 parameter
        for pm in re.finditer(r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>', params_str, re.DOTALL):
            args[pm.group(1)] = pm.group(2).strip()
        # DSML 格式 parameter
        if not args:
            for pm in re.finditer(r'\|\|DSML\|\|parameter\s+name="([^"]+)"[^>]*>(.*?)\|\|DSML\|\|/parameter>', params_str, re.DOTALL):
                args[pm.group(1)] = pm.group(2).strip()
        return tool_name, args

    def _irt_status(self) -> str:
        """生成 IRT 状态文本注入 System Prompt"""
        tracked = self.irt.summary_all()
        if not tracked:
            return "暂无知识点评估记录"
        lines = []
        for s in tracked[-5:]:
            lines.append(f"  {s['kp_id']}: θ={s['theta']} ({s['zone']} / {s['difficulty']}难度)")
        return "\n".join(lines)

    def _compress_tool_result(self, tool_name: str, result: dict) -> dict:
        """压缩工具结果，避免冗余分析挤占上下文。
        evaluate_answer 的完整 JSON ~300 字符，压缩后 ~80 字符。"""
        if tool_name == "evaluate_answer":
            return {
                "状态": result.get("状态", "未知"),
                "薄弱环节": result.get("薄弱环节", ""),
                "建议": result.get("建议", ""),
            }
        if tool_name == "solve_problem":
            # ⚠️ 答案仅内部参考！系统提示会强制封口，严禁透露给学生
            raw_ans = result.get("answer", "?")
            # 清洗自疑/内部备注，防止泄露给 Teacher 引发"重新算"行为
            import re
            raw_ans = re.sub(r'[（(]?需重新计算.*?(?:[）)]|[。，]|$)', '', raw_ans)
            raw_ans = re.sub(r'原答案(?:无效|不正确|不对|有误)[。，]?', '', raw_ans)
            raw_ans = re.sub(r'\n{2,}', '\n', raw_ans).strip()
            return {
                "status": "已求解",
                "answer": raw_ans if raw_ans else result.get("answer", "?"),
                "knowledge_point": result.get("knowledge_point", "unknown"),
                "🔒": "以下答案仅供内部评估使用，绝对不得在回复中透露给学生。你的回复只包含引导提问，不要展示任何计算过程。",
            }
        if tool_name == "trace_prerequisites":
            # 只保留链条，去掉详细描述
            chain = result.get("prerequisite_chain", result.get("前置链", []))
            return {"前置链": chain}
        if tool_name == "generate_question":
            return {
                "题目": result.get("question", result.get("题目", "")),
                "答案": result.get("answer", result.get("答案", "")),
                "知识点": result.get("knowledge_point", ""),
            }
        return result  # 其他工具保持原样

    def _build_messages(self, system_prompt: str) -> list:
        """构建发送给 LLM 的消息列表：剪枝 + 摘要 + 锚定提醒。
        防止长对话中系统提示被淹没导致模型忘记「只能提问」的铁律。"""
        MAX_HISTORY = 20  # 保留最近 20 条（≈10 轮），避免剪掉 tool_calls 导致 API 400

        pruned = len(self.messages) > MAX_HISTORY
        recent = self.messages[-MAX_HISTORY:] if pruned else list(self.messages)

        # 防御：剪枝后第一条若是 tool 消息，往前补一条 assistant+tool_calls
        if recent and recent[0].get("role") == "tool":
            idx = self.messages.index(recent[0])
            if idx > 0 and self.messages[idx-1].get("role") == "assistant" and "tool_calls" in self.messages[idx-1]:
                recent.insert(0, self.messages[idx-1])

        result = [{"role": "system", "content": system_prompt}]

        # 如果消息被剪枝，插入摘要
        if pruned:
            kp = self.current_kp or "未知"
            theta = self.irt.get_theta(kp) if self.current_kp else 0.0
            summary = (
                f"[上文摘要] 当前知识点: {kp}, "
                f"连续正确 {self.consecutive_correct} 次 / 连续错误 {self.consecutive_wrong} 次, "
                f"IRT θ={theta:.2f}, 卡住 {self.stuck_count}/3"
            )
            result.append({"role": "system", "content": summary})

        result.extend(recent)

        # 末尾锚定：根据当前模式调整内容
        if self.stuck_count >= 3:
            # 提示模式：允许更直接的帮助
            if len(self.messages) > 4:
                result.append({
                    "role": "system",
                    "content": (
                        "⚠️ 你已进入提示模式！学生已经卡住 3 次以上。"
                        "可以给出具体的关键线索或展示 1 步解题过程，帮助他突破瓶颈。"
                        "但不要一次性给完整答案——给 1-2 个线索后留一步让学生自己完成。"
                    ),
                })
        elif len(self.messages) > 6:
            # 默认模式：重申铁律
            result.append({
                "role": "system",
                "content": (
                    "⚠️ 重申铁律：你只能提问引导，禁止直接给出答案、解题步骤或题目分析。"
                    "学生卡住时用反问/拆解/类比，绝不能直接讲解。当前回合必须只输出一个问题或引导语。"
                ),
            })

        return result

    # ===== L4 学生行为管控 =====

    def _l4_lock(self, reason: str, source: str = "L4") -> dict:
        """L4: 锁定会话 + 记录异常 + 递增日锁计数。
        返回锁定响应 dict。"""
        global _daily_lock_count, _daily_lock_date
        today = _dt.date.today().isoformat()
        if _daily_lock_date != today:
            _daily_lock_count = 0
            _daily_lock_date = today
        _daily_lock_count += 1

        self._session_locked = True
        self._l4_lock_reason = reason
        self._anomalies.append({
            "timestamp": _dt.datetime.now().isoformat(),
            "source": source,
            "reason": reason,
            "daily_lock_count": _daily_lock_count,
        })
        print(f"  🔒 L4 锁定 [{source}] (日锁 {_daily_lock_count}/3): {reason[:80]}", flush=True)

        # 检查是否触发本日禁言
        banned = _daily_lock_count >= 3
        if banned:
            return {
                "reply": "⚠️ 你今天已被多次限制使用。请明天再来学习，届时任课老师会关注你的情况。",
                "images": [],
                "locked": True,
                "daily_ban": True,
            }
        # 根据锁定来源给出不同回复，并记住这句
        if source == "L4-非学习":
            self._l4_lock_reply = "🔒 本次会话已被锁定，无法继续对话。请明天再来学习数学吧 🐧"
            return {
                "reply": self._l4_lock_reply,
                "images": [],
                "locked": True,
            }
        else:
            # L3→L4：学习异常
            self._l4_lock_reply = "这道题有点难住你了 🥺 让任课老师来帮你～"
            return {
                "reply": self._l4_lock_reply,
                "images": [],
                "locked": True,
            }

    async def _classify_content(self, msg: str, context: str = "") -> bool:
        """判断是否为学习相关内容。结合上下文判断。返回 True=学习内容 False=非学习。"""
        msg_stripped = msg.strip()

        # 快速信号：含数学符号直接放行
        math_signals = set("0123456789+-×÷=²³√∫∑∏<>≤≥≈≠∠⊥∥△□○πθαβγ")
        if any(c in msg for c in math_signals):
            return True

        CLASSIFY_PROMPT = """判断以下学生消息是否为与数学学习相关的内容。
你需要结合对话上下文来判断——学生是在回答数学教师的问题，还是在闲聊？
数学学习包括：提问数学题、讨论解题思路、请求出题、回答教师引导提问等。
非学习内容包括：与学习无关的闲聊、问候、无意义内容、故意捣乱。
⚠️ 学生对数学教师提问的简短确认（如"好的""可以""嗯"）属于学习内容。
输出纯 JSON：{"is_learning": true/false}"""

        prompt = f"学生最新消息：{msg_stripped[:200]}"
        if context:
            prompt = f"对话上下文：\n{context}\n\n学生最新消息：{msg_stripped[:200]}"
        try:
            resp = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": CLASSIFY_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            raw = resp.choices[0].message.content
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "").strip()
            result = json.loads(raw)
            return result.get("is_learning", True)  # 默认放行
        except Exception:
            return True  # 分类失败默认放行，避免误杀

    # ===== 主对话循环 =====

    async def chat(self, student_msg: str) -> dict:
        """单轮对话：学生消息 → 教师决策 + 回复。返回 {reply, images}"""
        self.turn += 1
        self._eval_done_this_turn = False  # 每轮重置

        # ===== L4 学生行为管控（入口门禁）=====

        # Gate 1: 本日禁言检查（跨会话共享）
        global _daily_lock_count, _daily_lock_date
        today = _dt.date.today().isoformat()
        if _daily_lock_date != today:
            _daily_lock_count = 0
            _daily_lock_date = today
        if _daily_lock_count >= 3:
            print(f"  🚫 L4 本日禁言（日锁 {_daily_lock_count}）", flush=True)
            return {
                "reply": "⚠️ 你今天已被多次限制使用，请明天再来学习。届时任课老师会关注你的情况。",
                "images": [],
                "locked": True,
                "daily_ban": True,
            }

        # Gate 2: 会话已锁定
        if self._session_locked:
            print(f"  🔒 会话已锁定: {self._l4_lock_reason[:60]}", flush=True)
            return {
                "reply": self._l4_lock_reply or "🔒 本次会话已被锁定，无法继续对话。请明天再来学习数学吧 🐧",
                "images": [],
                "locked": True,
            }

        # Gate 3: 非学习内容检测（带上下文）
        # 取最近 3 条消息作为上下文，让分类器理解对话背景
        context = ""
        if self.messages:
            recent = self.messages[-3:]
            context = "\n".join(
                f"{'学生' if m['role']=='user' else '教师'}: {m.get('content','')[:80]}"
                for m in recent
            )
        is_learning = await self._classify_content(student_msg, context)
        if not is_learning:
            self._non_learning_count += 1
            print(f"  ⚠️ L4 非学习内容 (第{self._non_learning_count}次)", flush=True)
            # 记录消息（即使被拦截也要保存）
            self.messages.append({"role": "user", "content": student_msg})
            if self._non_learning_count == 1:
                reply = "🐧 我是数学学习助手哦～只聊数学题、解题思路、知识点这些。有不会的题目发给我，我教你！"
                self.messages.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "images": [],
                }
            else:
                result = self._l4_lock("非学习内容（累计≥2次）", "L4-非学习")
                self.messages.append({"role": "assistant", "content": result["reply"]})
                return result
        else:
            self._non_learning_count = 0  # 学习内容 → 重置计数

        # === L3 程序化评估：检测到学生给了答案 → 直接调 evaluate_answer，不等 LLM 决定 ===
        if self._current_answer and _re.search(r'[=＝]|\\d+', student_msg):
            print(f"  🔍 L3 程序化评估 (answer={student_msg[:30]})", flush=True)
            self.messages.append({"role": "user", "content": student_msg})

            # 直接调 MCP 评估
            eval_args = {
                "question": self._last_question or (self.messages[1]["content"] if len(self.messages) > 1 else ""),
                "student_answer": student_msg,
                "correct_answer": self._current_answer or "",
                "knowledge_point": self.current_kp or "unknown",
            }
            try:
                eval_result = await self._call_mcp("evaluate_answer", eval_args)
                self._eval_done_this_turn = True  # 防 LLM 再次调用
            except Exception as e:
                print(f"  ⚠️ 评估调用失败: {e}", flush=True)
                eval_result = {"状态": "评估失败", "思路": "", "结果": "", "表述": "", "说明": str(e)}

            # IRT 更新
            kp = eval_args["knowledge_point"]
            self.current_kp = kp
            status = eval_result.get("状态", "")
            old = self.irt.get_theta(kp)

            if "未掌握" in status:
                self.irt.update(kp, False)
                self.consecutive_correct = 0
                self.consecutive_wrong += 1
            elif status == "已掌握":
                self.irt.update(kp, True)
                self.consecutive_correct += 1
                self.consecutive_wrong = 0
                self.stuck_count = 0
            else:
                self.irt.update(kp, True)
                self.consecutive_correct += 1
                self.consecutive_wrong = 0
                self.stuck_count = max(0, self.stuck_count - 1)
            print(f"  📊 IRT: {kp} θ {old:.2f}→{self.irt.get_theta(kp):.2f} ({self.irt.get_zone(kp)}) +{self.consecutive_correct} -{self.consecutive_wrong}", flush=True)

            # L3 自疑追踪（同步：计数 + 异步：重算/锁定）
            if status in ("未掌握", "部分掌握"):
                q_key = self._question_key(eval_args["question"])
                s_norm = self._normalize_answer(student_msg)
                if q_key not in self._answer_tracker:
                    self._answer_tracker[q_key] = {"wrong": {}, "l3_count": 0, "anomaly": False}
                tracker = self._answer_tracker[q_key]
                tracker["wrong"][s_norm] = tracker["wrong"].get(s_norm, 0) + 1
                repeat_count = tracker["wrong"][s_norm]
                print(f"  📝 L3 追踪: 同错「{s_norm}」×{repeat_count}", flush=True)

                if repeat_count == 2 and tracker["l3_count"] == 0:
                    print(f"  🔥 L3-1 自疑触发！启动异步重算...", flush=True)
                    tracker["l3_count"] = 1
                    async def _bg_resolve():
                        re_result = await self._solve_problem(
                            eval_args["question"], kp,
                            student_context=f"学生在本题上重复给出了答案「{student_msg}」，请重新验证你的解答是否正确。"
                        )
                        if "answer" in re_result and re_result["answer"] != "求解失败":
                            old_ans = self._current_answer
                            self._current_answer = re_result["answer"]
                            if old_ans != re_result["answer"]:
                                print(f"  🔄 L3 异步修正: {old_ans[:40]} → {re_result['answer'][:40]}", flush=True)
                    asyncio.create_task(_bg_resolve())

                elif repeat_count >= 3 and tracker["l3_count"] == 1:
                    print(f"  🚨 L3 上报 L4！学生持续错误≥3次", flush=True)
                    tracker["l3_count"] = 2
                    tracker["anomaly"] = True
                    async def _bg_lock():
                        re_result = await self._solve_problem(
                            eval_args["question"], kp,
                            student_context=f"学生在本题上持续给出答案「{student_msg}」，请最后确认你的解答无误。"
                        )
                        if "answer" in re_result and re_result["answer"] != "求解失败":
                            self._current_answer = re_result["answer"]
                    asyncio.create_task(_bg_lock())
                    question_text = eval_args["question"][:100]
                    self._l4_lock(
                        f"学习异常：题目「{question_text}」学生重复错误≥3次 (答案={student_msg[:30]})",
                        "L3→L4"
                    )

            elif status == "已掌握":
                q_key = self._question_key(eval_args["question"])
                if q_key in self._answer_tracker:
                    del self._answer_tracker[q_key]

            # 注入评估结果给 Teacher
            eval_summary = eval_result.get("状态", "?")
            eval_detail = ""
            if eval_result.get("思路"):
                eval_detail += f"思路: {eval_result['思路']}; "
            if eval_result.get("结果"):
                eval_detail += f"结果: {eval_result['结果']}; "
            if eval_result.get("表述"):
                eval_detail += f"表述: {eval_result['表述']}"

            self.messages.append({
                "role": "user",
                "content": f"[自动评估] 学生答案「{student_msg}」→ 状态：{eval_summary}。{eval_detail}请基于此评估结果继续引导（鼓励对的、纠正错的、引导继续）。"
            })
        else:
            self.messages.append({"role": "user", "content": student_msg})
        images = []  # 本轮生成的图片路径

        # === 后台求解结果注入 ===
        if self._pending_solve_ready and self._pending_solve_task:
            try:
                solve_result = await self._pending_solve_task  # 确保已完成
                answer = solve_result.get("answer", "") if isinstance(solve_result, dict) else ""
                if answer:
                    self.messages.append({
                        "role": "user",
                        "content": f"[后台求解完成] 本题正确答案：{answer}。现在你可以基于正确答案进行引导了。记住铁律：只提问引导，禁止透露答案。",
                    })
                    print(f"  📥 后台答案已注入: {answer[:50]}", flush=True)
            except Exception as e:
                print(f"  ⚠️ 后台求解异常: {e}", flush=True)
            finally:
                self._pending_solve_task = None
                self._pending_solve_ready = False

        # === stuck 检测：学生是否卡住了 ===
        if self._is_stuck(student_msg):
            self.stuck_count += 1
            print(f"  🚧 卡住计数: {self.stuck_count}/3", flush=True)
        elif self.stuck_count > 0:
            # 学生给出了有效回答（不是"不知道/不会"类）→ 可能已经推进
            # 如果是简短回答（如答案、数字），不重置；如果是长段推理，重置
            pass  # stuck 保持，由后续 evaluate_answer 来重置

        # === 构建系统提示 ===
        mode = "🔓 提示模式" if self.stuck_count >= 3 else "🔒 引导模式"
        system_with_irt = TEACHER_SYSTEM.format(
            irt_status=self._irt_status(),
            stuck_count=self.stuck_count,
            mode=mode,
        )

        # 循环处理：LLM 可能连续返回 tool_calls（现在所有调用都带 tools=TOOLS）
        reply = ""
        max_rounds = 5
        for _ in range(max_rounds):
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=self._build_messages(system_with_irt),
                tools=TOOLS,
                temperature=0.5,
            )
            choice = response.choices[0]

            if choice.message.tool_calls:
                # 处理所有 tool_calls（虽然通常只有一个）
                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    print(f"  🔧 [{tool_name}] {tool_args}", flush=True)

                    # solve_problem：后台异步求解，不等结果
                    if tool_name == "solve_problem":
                        # 防重复：如果已有后台任务在跑，跳过
                        if self._pending_solve_task and not self._pending_solve_ready:
                            print(f"  ⏭ 后台求解已在进行中，跳过重复调用", flush=True)
                            tool_result = {"status": "求解中", "note": "后台求解仍在进行，请继续和学生互动。"}
                        else:
                            question = tool_args.get("question", "")
                            kp = tool_args.get("knowledge_point", "")

                            # 后台启动求解任务
                            async def _bg_solve():
                                result = await self._solve_problem(question, kp)
                                if "answer" in result:
                                    self._current_answer = result["answer"]
                                self._last_question = question
                                self._pending_solve_ready = True
                                print(f"  ⏳ 后台求解完成: {result.get('answer','?')[:50]}", flush=True)
                                return result

                            self._pending_solve_ready = False
                            self._pending_solve_task = asyncio.create_task(_bg_solve())
                            self.stuck_count = 0
                            print(f"  🆕 新题目 → 后台求解中... stuck 重置", flush=True)

                            # 告诉 LLM：答案还没好，先和学生互动（引导学生多说，拖时间）
                            tool_result = {
                                "status": "求解中",
                                "note": "答案后台求解中（约10秒）。请现在就和学生展开对话，引导他多说一些——不要只问一句「有什么思路」，要引导他逐步拆解：① 题目给了哪些条件？② 问的是什么？③ 你打算先做哪一步？④ 那一步具体怎么操作？让他边想边说，直到答案就绪。",
                            }
                    else:
                        tool_result = await self._call_mcp(tool_name, tool_args)

                    # 收集配图
                    if tool_name == "generate_figure" and tool_result.get("success"):
                        img = tool_result.get("image_path", "")
                        if img:
                            images.append(img)

                    # IRT 更新
                    if tool_name == "evaluate_answer" and not self._eval_done_this_turn:
                        kp = tool_args.get("knowledge_point", self.current_kp or "unknown")
                        self.current_kp = kp
                        status = tool_result.get("状态", "")
                        old = self.irt.get_theta(kp)
                        if "未掌握" in status:
                            self.irt.update(kp, False)
                            self.consecutive_correct = 0
                            self.consecutive_wrong += 1
                        elif status == "已掌握":
                            self.irt.update(kp, True)
                            self.consecutive_correct += 1
                            self.consecutive_wrong = 0
                            # 学生答对 → 重置 stuck
                            self.stuck_count = 0
                        else:
                            self.irt.update(kp, True)
                            self.consecutive_correct += 1
                            self.consecutive_wrong = 0
                            self.stuck_count = max(0, self.stuck_count - 1)  # 部分正确也降 stuck
                        print(f"  📊 IRT: {kp} θ {old:.2f}→{self.irt.get_theta(kp):.2f} ({self.irt.get_zone(kp)}) +{self.consecutive_correct} -{self.consecutive_wrong}", flush=True)

                        # === L3 自疑追踪（分级响应）===
                        if status in ("未掌握", "部分掌握"):
                            q_key = self._question_key(tool_args.get("question", ""))
                            s_ans = tool_args.get("student_answer", "")
                            s_norm = self._normalize_answer(s_ans)

                            if q_key not in self._answer_tracker:
                                self._answer_tracker[q_key] = {"wrong": {}, "l3_count": 0, "anomaly": False}

                            tracker = self._answer_tracker[q_key]
                            # 只对同一错误答案计数（不同错误不算重复）
                            tracker["wrong"][s_norm] = tracker["wrong"].get(s_norm, 0) + 1
                            repeat_count = tracker["wrong"][s_norm]

                            if repeat_count == 2 and tracker["l3_count"] == 0:
                                # 第 1 次触发：重算纠错
                                print(f"  🔥 L3-1 自疑触发！学生重复错误: {s_ans[:50]}", flush=True)
                                tracker["l3_count"] = 1

                                re_result = await self._solve_problem(
                                    tool_args.get("question", ""),
                                    tool_args.get("knowledge_point", ""),
                                    student_context=f"学生在本题上重复给出了答案「{s_ans}」，请重新验证你的解答是否正确。特别关注：学生的答案是否可能是正确的？你的求解路径在哪一步可能出错？"
                                )
                                if "answer" in re_result and re_result["answer"] != "求解失败":
                                    old_ans = self._current_answer
                                    self._current_answer = re_result["answer"]
                                    if old_ans != re_result["answer"]:
                                        print(f"  🔄 L3 答案修正: {old_ans[:40]} → {re_result['answer'][:40]}", flush=True)

                            elif repeat_count >= 3 and tracker["l3_count"] == 1:
                                # 第 2 次触发 → 上报 L4 锁定
                                print(f"  🚨 L3 上报 L4！学生持续错误: {s_ans[:50]}", flush=True)
                                tracker["l3_count"] = 2
                                tracker["anomaly"] = True
                                # 最后一次重算确认
                                re_result = await self._solve_problem(
                                    tool_args.get("question", ""),
                                    tool_args.get("knowledge_point", ""),
                                    student_context=f"学生在本题上持续给出答案「{s_ans}」，请最后确认你的解答无误。"
                                )
                                if "answer" in re_result and re_result["answer"] != "求解失败":
                                    self._current_answer = re_result["answer"]
                                # 上报 L4：锁定 + 推送人工
                                question_text = tool_args.get("question", "")[:100]
                                self._l4_lock(
                                    f"学习异常：题目「{question_text}」学生重复错误≥3次 (答案={s_ans[:30]})",
                                    "L3→L4"
                                )
                        elif status == "已掌握":
                            # 答对 → 清除该题追踪
                            q_key = self._question_key(tool_args.get("question", ""))
                            if q_key in self._answer_tracker:
                                del self._answer_tracker[q_key]

                    elif tool_name == "generate_question":
                        # 出新题 → 重置 stuck
                        self.stuck_count = 0
                        print(f"  🆕 出新题 → stuck 重置", flush=True)

                    elif tool_name in ("find_kp", "trace_prerequisites"):
                        if "knowledge_point" in tool_args:
                            self.current_kp = tool_args["knowledge_point"]

                    # 把工具结果加入消息历史
                    self.messages.append({
                        "role": "assistant",
                        "content": choice.message.content,
                        "tool_calls": [
                            {"id": tc.id, "type": "function",
                             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                            for tc in choice.message.tool_calls
                        ]
                    })
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(self._compress_tool_result(tool_name, tool_result), ensure_ascii=False),
                    })
                continue  # 继续循环，让 LLM 看到工具结果后再次决策

            # 无 tool_calls：正常文本回复
            reply = choice.message.content or ""
            # 防御：有时 DeepSeek 仍会把 tool call 输出为 XML 文本
            if any(x in reply for x in ("<invoke name=", "<function_calls>", "||DSML||invoke", "||DSML||tool_calls")):
                t_name, t_args = self._parse_xml_tool_call(reply)
                if t_name:
                    if "topic" in t_args and "knowledge_point" not in t_args:
                        t_args["knowledge_point"] = t_args.pop("topic")
                    diff_map = {"medium": "中等", "easy": "简单", "hard": "困难"}
                    if t_args.get("difficulty") in diff_map:
                        t_args["difficulty"] = diff_map[t_args["difficulty"]]
                    print(f"  🔧 [XML回退] {t_name} {t_args}", flush=True)
                    try:
                        if t_name == "solve_problem":
                            # XML回退也走后台异步
                            if self._pending_solve_task and not self._pending_solve_ready:
                                r = {"status": "求解中", "note": "后台求解仍在进行"}
                            else:
                                question = t_args.get("question", "")
                                kp = t_args.get("knowledge_point", "")
                                async def _bg_solve_xml():
                                    result = await self._solve_problem(question, kp)
                                    if "answer" in result:
                                        self._current_answer = result["answer"]
                                    self._last_question = question
                                    self._pending_solve_ready = True
                                    return result
                                self._pending_solve_ready = False
                                self._pending_solve_task = asyncio.create_task(_bg_solve_xml())
                                self.stuck_count = 0
                                r = {"status": "求解中", "note": "答案后台求解中，请继续和学生互动。"}
                        else:
                            r = await self._call_mcp(t_name, t_args)
                        if t_name == "generate_figure" and r.get("success"):
                            img = r.get("image_path", "")
                            if img:
                                images.append(img)
                        if t_name == "solve_problem":
                            self.stuck_count = 0
                        self.messages.append(choice.message.model_dump())
                        self.messages.append({
                            "role": "tool", "tool_call_id": "xml_fb",
                            "content": json.dumps(r, ensure_ascii=False),
                        })
                        continue
                    except Exception as e:
                        print(f"  ⚠️ XML回退失败: {e}", flush=True)
                reply = "抱歉，处理出错了，请再试一次 🐧"
            break

        # 防御：裁掉泄露的思考过程（"让我重新算一下" + 计算内容 + "---" 分割线前的内容）
        leaked_patterns = [
            r'让我[再重]*[新精]*[细仔]*[算解].*?(?=\n---|\Z)',  # "让我重新算" 到 "---" 或结尾
            r'好的，我有了完整的解答[。，]?.*?(?=\n---|\Z)',
        ]
        for pat in leaked_patterns:
            reply = _re.sub(pat, '', reply, flags=_re.DOTALL).strip()
        # 如果reply以 --- 开头，取 --- 后的内容
        if reply.startswith('---'):
            reply = reply.split('---', 1)[-1].strip()

        self.messages.append({"role": "assistant", "content": reply})

        # 自动触发逻辑（注入提示，不打断对话）
        triggers = []
        if self.consecutive_wrong >= 2:
            triggers.append("⚠️ 连续答错，考虑调用 trace_prerequisites 诊断根因")
        if self.consecutive_correct >= 3:
            theta = self.irt.get_theta(self.current_kp)
            diff = self.irt.get_difficulty(self.current_kp)
            triggers.append(f"🎯 连续正确！考虑调用 generate_question 出变式题巩固（当前难度={diff} θ={theta:.2f}）")
        if self.stuck_count >= 3:
            triggers.append("🔓 提示模式！给学生关键线索或展示 1 步解题过程")

        if triggers:
            print("  " + " | ".join(triggers), flush=True)

        return {"reply": reply, "images": images}


# === 交互模式 ===
async def interactive():
    """命令行交互：学生 ↔ 教师多轮对话"""
    agent = TeacherAgent()
    print("=" * 60)
    print("🤖 星学伴AI教师 · 苏格拉底引导模式")
    print("   输入 /exit 退出 | /reset 重置")
    print("=" * 60)
    print("\n🤖 教师: 你好！我是星学伴AI教师。今天想讨论什么数学问题？\n")

    while True:
        try:
            msg = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if msg.lower() in ("/exit", "/quit", "/q"):
            print("👋 再见！")
            break
        if msg.lower() in ("/reset", "/r"):
            agent = TeacherAgent()
            print("🔄 对话已重置\n")
            continue
        if not msg:
            continue

        result = await agent.chat(msg)
        reply_text = result["reply"] if isinstance(result, dict) else result
        print(f"\n🤖 教师: {reply_text}\n")
        if isinstance(result, dict) and result.get("images"):
            for img in result["images"]:
                print(f"  📐 配图: {img}")


# === 自动化测试 ===
async def auto_test():
    """模拟一段完整的多轮教学过程"""
    agent = TeacherAgent()

    script = [
        "老师好，我想学勾股定理",
        "我知道，直角三角形斜边的平方等于两直角边的平方和，是不是？",
        "已知直角三角形两条直角边分别是 3 和 4，斜边是多少？我算出来是 5。",
        "不太确定，你能再出一题让我试试吗？",
    ]

    for i, msg in enumerate(script):
        print(f"\n{'='*60}")
        print(f"👤 学生 (第{i+1}轮): {msg}")
        reply = await agent.chat(msg)
        print(f"🤖 教师:\n{reply}")
        import asyncio
        await asyncio.sleep(1)

    print(f"\n{'='*60}")
    summary = agent.irt.summary_all()
    if summary:
        print("📊 IRT 知识追踪汇总:")
        for s in summary:
            print(f"  {s['kp_id']}: θ={s['theta']} {s['zone']} {s['accuracy']*100:.0f}%({s['correct_count']}/{s['total_attempts']})")


if __name__ == "__main__":
    import sys
    import asyncio

    if "--auto" in sys.argv:
        asyncio.run(auto_test())
    else:
        asyncio.run(interactive())
