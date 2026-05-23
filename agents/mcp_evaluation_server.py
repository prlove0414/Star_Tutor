"""
评估 Agent · MCP Server (HTTP)
- 三维度评估学生答题：思路正确性 / 结果正确性 / 表述完整性
- 使用 DeepSeek V4 API 做评估推理
"""
import json
import sys
from openai import OpenAI
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn

# === 配置 ===
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
HOST = "127.0.0.1"
PORT = 8767


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# === DeepSeek 客户端 ===
llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")

EVAL_SYSTEM_PROMPT = """你是初中数学评估专家。根据题目、标准答案和学生回答，从三个维度给出结构化评价。

评估规则：
1. 思路正确性：学生解题方向是否正确（即便结果算错）
2. 结果正确性：最终答案是否准确
3. 表述完整性：步骤是否完整、逻辑是否清晰

⚠️ 数学表达式等价判定（重要）：
判断"结果正确性"时必须识别等价表达式，不能做字面字符串匹配。以下为常见等价形式：
- 乘法：2*根号13 = 2倍根号13 = 2×根号13 = 2√13 = 二倍的根号十三
- 分数：1/2 = 二分之一 = 0.5 = ½
- 指数：x^2 = x² = x的平方
- 括号等价：(a+b)c = c(a+b) = ac+bc
- 不同书写顺序：x+y=z 等价于 z=x+y
评估时应以数学含义为准，学生使用编程符号（* ^ /）、中文表达、或不同书写顺序但数值/逻辑正确，都应判定为正确。

状态判定：
- 已掌握：三维度都达标
- 部分掌握：存在 1-2 个维度不达标
- 未掌握：三维度都不达标，或思路完全错误

严格按以下 JSON 格式输出，不要输出其他内容：
{
  "状态": "已掌握/部分掌握/未掌握",
  "思路正确性": {"结果": "正确/部分正确/错误", "分析": "一句话分析"},
  "结果正确性": {"结果": "正确/部分正确/错误", "分析": "一句话分析"},
  "表述完整性": {"结果": "完整/不完整", "分析": "一句话分析"},
  "薄弱环节": "如果有问题，具体是哪里薄弱",
  "建议": "下一步学习建议"
}"""


def evaluate(question: str, student_answer: str, correct_answer: str, knowledge_point: str = "") -> dict:
    """调用 DeepSeek V4 做三维度评估"""
    prompt = f"""知识点：{knowledge_point or "未指定"}

题目：{question}

标准答案：{correct_answer}

学生回答：{student_answer}"""
    
    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content
    
    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        return {"状态": "评估失败", "原始输出": raw}


# === MCP Server ===
server = Server("star-tutor-evaluation")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="evaluate_answer",
            description="评估学生的答题质量。从思路正确性、结果正确性、表述完整性三个维度打分，判定掌握状态并给出建议。",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "题目内容"},
                    "student_answer": {"type": "string", "description": "学生的回答"},
                    "correct_answer": {"type": "string", "description": "标准答案"},
                    "knowledge_point": {"type": "string", "description": "考察的知识点（可选）"},
                },
                "required": ["question", "student_answer", "correct_answer"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "evaluate_answer":
        log(f"📊 评估: {arguments.get('knowledge_point', '未指定')}")
        result = evaluate(
            question=arguments["question"],
            student_answer=arguments["student_answer"],
            correct_answer=arguments["correct_answer"],
            knowledge_point=arguments.get("knowledge_point", ""),
        )
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    raise ValueError(f"Unknown tool: {name}")


# === HTTP 路由 ===
sse = SseServerTransport("/messages/")


async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


async def health(request):
    return JSONResponse({"status": "ok", "agent": "evaluation"})


app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Route("/health", endpoint=health),
    Mount("/messages/", app=sse.handle_post_message),
])

if __name__ == "__main__":
    log(f"🚀 评估 Agent: http://{HOST}:{PORT}/sse")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
