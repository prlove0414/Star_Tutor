"""
出题 Agent · MCP Server (HTTP 持久版)
- 本地 GPU 推理（QUESTION_PROVIDER=local，默认）
- 远程 API 调用（QUESTION_PROVIDER=api，无需 GPU）
- 通过 HTTP/SSE 暴露 MCP 工具
"""
import json
import sys
import torch
from openai import OpenAI
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn

# === 配置 ===
from config import MODEL_PATH, QUESTION_PROVIDER, DEEPSEEK_API_KEY, DEEPSEEK_API_BASE
HOST = "127.0.0.1"
PORT = 8765

SYSTEM_PROMPT = (
    "你是初中数学出题助手。根据用户指定的知识点和难度，生成一道完整的数学题目并解答。\n"
    "必须输出合法JSON，包含三个字段：question（题目）、solution（详细解题过程）、answer（最终答案，含单位）。\n"
    "\n"
    "## 难度标准\n"
    "- 简单：直接套公式/定理即可，一步到位，适合基础薄弱学生建立信心\n"
    "- 中等：需要 2-3 步推理，涉及一个知识点的中等应用\n"
    "- 困难：必须满足以下至少 2 项——\n"
    "  ① 涉及 2 个以上知识点的综合运用\n"
    "  ② 需要辅助线/设元/分类讨论等解题技巧\n"
    "  ③ 题目含隐藏条件，需要学生自己挖掘\n"
    "  ④ 有陷阱或易错点（如单位换算、多解情况、排除干扰条件）\n"
    "  ⑤ 是中考压轴题难度级别的综合题\n"
    "\n"
    "## 输出要求\n"
    "直接输出JSON，不要加```包裹，不要输出解释。\n"
    "困难题不设字数上限——题目可以长、条件可以多，确保真正有挑战性。"
)


def log(msg: str):
    """所有日志走 stderr，不污染 MCP 协议"""
    print(msg, file=sys.stderr, flush=True)


# === 模型 / API 客户端加载 ===
model = None
tokenizer = None
api_client = None
used_vram = 0
total_vram = 0

if QUESTION_PROVIDER == "local":
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    log("\u23f3 加载出题模型 (4-bit 本地)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    used_vram = torch.cuda.memory_allocated() / 1024**3
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log(f"\u2705 出题 Agent 就绪 (本地) | 显存: {used_vram:.1f}G/{total_vram:.1f}G | http://{HOST}:{PORT}")

elif QUESTION_PROVIDER == "api":
    api_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
    log(f"\u2705 出题 Agent 就绪 (API) | {DEEPSEEK_API_BASE} | http://{HOST}:{PORT}")

else:
    raise ValueError(f"未知的 QUESTION_PROVIDER: {QUESTION_PROVIDER}，请设为 local 或 api")


# === 工具实现 ===
def generate_question(knowledge_point: str, difficulty: str, question_type: str = "解答题") -> dict:
    prompt = f"请生成一道关于「{knowledge_point}」的{difficulty}难度{question_type}。"

    if QUESTION_PROVIDER == "local" and model is not None:
        # 本地模型推理
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs, max_new_tokens=512, temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    else:
        # API 调用
        response = api_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,  # 扩容以支持困难题的长输出
        )
        raw = response.choices[0].message.content

    # 从模型输出中提取 JSON（可能前面有 CoT 思考过程）
    import re
    # 先尝试直接解析
    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        result = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        # 回退：找文本中最后一个 { ... } JSON 块（本地模型 CoT 后通常以 JSON 结尾）
        result = None
        last_brace = raw.rfind("{")
        if last_brace >= 0:
            # 从最后一个 { 开始，找匹配的 }
            depth = 0
            end = -1
            for i in range(last_brace, len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end > last_brace:
                try:
                    result = json.loads(raw[last_brace:end + 1])
                except json.JSONDecodeError:
                    pass
        if not result:
            result = {"question": raw, "solution": "", "answer": ""}

    # 清洗：本地模型可能在 question 中输出思考过程，只保留实际题目
    q = result.get("question", "")
    for marker in ["**Final Question:**", "最终题目：", "## 题目", "**题目：**", "Thinking Process:"]:
        if marker in q:
            after = q.split(marker, 1)[1].strip()
            # 如果切出来是空的，说明 marker 后面是 JSON 或空行，跳过
            if len(after) > 5:
                q = after
                result["question"] = q
                break
    # 二次清洗：去掉行首的 CoT 角色标记
    for prefix in ["1.  **Analyze", "2.  **Determine", "3.  **"]:
        idx = q.find(prefix)
        if idx > 0:
            # 回头看前一个换行，截断到那
            prev_nl = q.rfind("\n", 0, idx)
            q = q[:prev_nl if prev_nl > 0 else idx].strip()
            result["question"] = q
            break

    return result


# === MCP Server ===
server = Server("star-tutor-question-generator")


@server.list_tools()
async def list_tools():
    from mcp.types import Tool
    return [
        Tool(
            name="generate_question",
            description="生成一道初中数学题，返回题目、解题步骤和最终答案。",
            inputSchema={
                "type": "object",
                "properties": {
                    "knowledge_point": {"type": "string", "description": "知识点，例如：勾股定理"},
                    "difficulty": {"type": "string", "enum": ["简单", "中等", "困难"]},
                    "question_type": {"type": "string", "enum": ["选择题", "填空题", "解答题"], "default": "解答题"},
                },
                "required": ["knowledge_point", "difficulty"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    from mcp.types import TextContent
    if name == "generate_question":
        log(f"\U0001f4dd 出题: {arguments.get('knowledge_point')} {arguments.get('difficulty')}")
        result = generate_question(**arguments)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    raise ValueError(f"Unknown tool: {name}")


# === HTTP 路由 ===
sse = SseServerTransport("/messages/")


async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


async def health(request):
    provider = "local" if model else "api"
    return JSONResponse({"status": "ok", "provider": provider})


app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Route("/health", endpoint=health),
    Mount("/messages/", app=sse.handle_post_message),
])

if __name__ == "__main__":
    log(f"\U0001f680 启动 MCP Server: http://{HOST}:{PORT}/sse")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
