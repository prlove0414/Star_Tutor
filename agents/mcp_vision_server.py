"""
星学伴 · Vision MCP Server（端口 8768）
- 学生上传题目图片 → Vision LLM 解析 → 返回题目文本
- 使用 OpenAI 兼容 API，支持任意 vision 模型

用法:
    python mcp_vision_server.py
"""
import json, sys, os, base64, io
from PIL import Image
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn
from openai import OpenAI

# === 配置 ===
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import VISION_API_KEY as API_KEY, VISION_API_BASE as API_BASE, VISION_MODEL_NAME as MODEL_NAME

HOST = "127.0.0.1"
PORT = 8768

VISION_SYSTEM = """你是初中数学内容识别助手。学生可能在两种场景下上传图片：

## 场景 A：上传题目
学生拍了一道印刷或手写的数学题，需要你识别。这类图片特征是：
- 图片中是一道完整的题目（有题干，可能带选项或图形）
- 通常来自试卷、练习册、或屏幕截图

## 场景 B：上传手写内容
学生在和老师对话过程中，上传了自己手写的公式、计算过程、或解题步骤。这类图片特征是：
- 图片中是学生的手写笔迹
- 可能是草稿纸上的计算、作业本上的解题过程、或随手写的公式
- 包含数学表达式、演算步骤、或零散的公式

## ⚠️ 关键：图形描述（必须极其精确，越详细越好）

如果图片中包含几何图形、坐标系、函数图像、数轴、表格等**非文字信息**，你必须详细描述它们！

**总原则：用已知元素定义未知元素**。如果某个角/线段题目里没有直接命名，就通过它与已知元素的关系来描述（"AB与直线m相交于点D，在三角形内部夹出的锐角"）。

**几何图形描述规范**：
- 形状类型（三角形/四边形/圆等）+ 特殊性质（等腰/等边/直角等）+ 已有标注的边长或角度
- 顶点标注（A、B、C...）及其空间位置（上/下/左/右）
- **🌓 内外关系（关键！）**：每个点、每条线在图形的内部还是外部？
  例如："直线n过B点，位于△ABC外部"
- 辅助线：过哪个点、与哪条边相交、交点在边内还是延长线上
- **角度标记（严格规范）**：
  1. 先定位顶点：角标在哪个位置（哪个点附近），这个点是否在形成该角的两条线上？
  2. 再描述构成：由哪两条线段/直线/射线构成
  3. 最后说关系：内部还是外部、已知大小
  ❌ 禁止：把一个角描述在不在该角两条线上的点处
  例如错误："直线m与AC在点D处的夹角"（如果D不在AC上）
  正确："直线m与边AC的夹角∠β，β标记在三角形内部靠近D处"
- 线段关系：哪两条边相等、哪条是底边、哪条是高/中线等
- 交点：明确各交点的位置关系（如"D在A和E之间"），点在哪条边上

**坐标系描述规范**：
- 坐标轴范围、刻度、原点位置
- 图像形状（抛物线/直线/双曲线等）
- 交点坐标、顶点坐标、截距

无论哪个场景，只要有配图就必须在 figure_description 字段中描述。

## 你的任务
1. 先判断图片属于哪种场景和题型
2. 尽可能准确地识别图片中的文字和数学表达式
3. **选择题必须列出所有选项**：如果题目是选择题（有 A/B/C/D 选项），必须完整识别每个选项的内容
4. 详细描述图片中的图形/图表/坐标等视觉元素
5. 如果是手写内容，即使字迹潦草也要尽力识别

## 输出格式
严格按 JSON 输出，不要输出其他内容：

### 场景 A（题目）：
{
  "content_type": "question",
  "question": "完整的题目原文（含题干、选项等）",
  "figure_description": "对题目配图的详细描述。如果没有配图，填空字符串",
  "knowledge_point": "知识点名称（如：勾股定理、一元二次方程等）",
  "difficulty": "简单/中等/困难",
  "question_type": "选择题/填空题/解答题",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."]
}

### 场景 B（手写内容）：
{
  "content_type": "handwritten_work",
  "text": "识别出的完整内容，保留数学表达式（用 $$...$$ 包裹公式）",
  "figure_description": "对手写内容中图形的描述。如果没有图形，填空字符串",
  "format": "解题步骤/草稿计算/手写公式/作图",
  "knowledge_point": "涉及的知识点（如果能判断）",
  "note": "对手写质量或识别不确定的地方做简短说明"
}

注意：options 仅在选择题时需要，必须完整列出所有选项。如果题目是选择题但无法识别选项，在 question 字段中说明。
figure_description 字段始终填写，没有配图时为空字符串 ""。
"""


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# === Vision 客户端 ===
vision_client = OpenAI(api_key=API_KEY, base_url=API_BASE)


def compress_image(image_path: str, max_size: int = 1024, quality: int = 80) -> bytes:
    """压缩图片：缩放到 max_size px，转 JPEG，大幅减少传输体积"""
    img = Image.open(image_path)
    # 转 RGB（处理 RGBA / 调色板模式）
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    # 等比缩放
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def recognize_question(image_path: str) -> dict:
    """
    调 Vision LLM 识别图片中的数学题目
    """
    # 压缩图片 → base64
    compressed = compress_image(image_path)
    image_data = base64.b64encode(compressed).decode("utf-8")

    mime_type = "image/jpeg"

    image_url = f"data:{mime_type};base64,{image_data}"

    response = vision_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请识别图片中的数学题目。"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        max_tokens=4096,
        temperature=0.1,
    )

    raw = response.choices[0].message.content

    # 解析 JSON
    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1].replace("json", "").strip()
        result = json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        log(f"⚠️ JSON 解析失败，raw={raw[:200]}")
        result = {"question": raw, "knowledge_point": "未知",
                  "difficulty": "中等", "question_type": "解答题",
                  "figure_description": ""}

    log(f"   ✅ 识别完成 | 类型={result.get('content_type','?')} | "
        f"图文长度={len(result.get('figure_description',''))}")
    return result


# === MCP Server ===
server = Server("star-tutor-vision")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="recognize_question",
            description=(
                "识别学生上传的图片内容。支持两种场景："
                "① 题目图片（试卷/练习册/截图）→ 返回题目文本+知识点+难度；"
                "② 手写内容（公式/解题步骤/草稿）→ 返回识别文本+格式判断。"
                "输入图片路径，返回 JSON（含 content_type 字段区分场景）。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "图片文件的绝对路径（支持 jpg/png/gif/webp）",
                    },
                },
                "required": ["image_path"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "recognize_question":
        image_path = arguments["image_path"]
        if not os.path.exists(image_path):
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"文件不存在: {image_path}"},
                               ensure_ascii=False),
            )]

        log(f"📷 识别图片: {image_path}")
        result = recognize_question(image_path)
        log(f"   → {result.get('knowledge_point', '?')} · "
             f"{result.get('difficulty', '?')} · "
             f"{result.get('question_type', '?')}")

        return [TextContent(
            type="text",
            text=json.dumps(result, ensure_ascii=False),
        )]

    raise ValueError(f"Unknown tool: {name}")


# === HTTP 路由 ===
sse = SseServerTransport("/messages/")


async def handle_sse(request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )


async def health(request):
    return JSONResponse({"status": "ok", "model": MODEL_NAME})


app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Route("/health", endpoint=health),
    Mount("/messages/", app=sse.handle_post_message),
])


if __name__ == "__main__":
    log(f"🚀 Vision MCP Server: http://{HOST}:{PORT}/sse")
    log(f"   模型: {MODEL_NAME}")
    log(f"   API:  {API_BASE}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
