#!/usr/bin/env python3
"""
星学伴 · Figure MCP Server (Port 8769)
- ① 路由决策：TikZ or Matplotlib
- ② DeepSeek V4 Pro 生成渲染代码
- ③ pdflatex / python 渲染 → PNG

依赖: texlive-{base,latex-base,pictures,latex-recommended}, imagemagick, matplotlib
"""
import json, os, sys, re, subprocess, tempfile, shutil
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn

# === 配置 ===
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE

HOST = "127.0.0.1"
PORT = 8769
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "figures")

# 路由规则：哪些知识点/题型 → 哪个引擎
# TikZ: 几何/函数/统计  Matplotlib: 等高线(地理)
TIKZ_KP_KEYWORDS = [
    "三角形", "四边形", "圆", "平行", "相似", "全等", "勾股",
    "函数", "方程", "不等式", "坐标", "概率", "统计",
    "图形", "几何", "角", "面积", "体积",
]
MPL_KP_KEYWORDS = ["等高线", "地形"]  # 未来扩展


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] FIGURE {msg}", file=sys.stderr, flush=True)


def route_engine(knowledge_point: str, question_type: str) -> str:
    """决定用 TikZ 还是 Matplotlib"""
    kp = knowledge_point.lower()
    # 等高线 → matplotlib
    for kw in MPL_KP_KEYWORDS:
        if kw in kp:
            return "matplotlib"
    # 其余全走 tikz
    return "tikz"


# ══════════════════════════════════════════
#  DeepSeek Prompt 模板
# ══════════════════════════════════════════

TIKZ_SYSTEM = """你是中学数学配图专家。根据题目描述，生成**完整、可编译**的 LaTeX standalone 文档，
用 TikZ + tkz-euclide 画出精确的配图。

## 规范
1. `\\documentclass[tikz,border=5pt]{standalone}`
2. 必须加载：`\\usepackage{tkz-euclide}`
   （注意：不需要 ctex，所有文字标注用 math mode `$...$`，如 `$A$`、`$B$`）
3. **几何构造用 tkz-euclide 命令，不要手算坐标**：
   - 定义点：`\\tkzDefPoint(0,0){C} \\tkzDefPoint(0,6){A} \\tkzDefPoint(8,0){B}`
   - 中点：`\\tkzDefMidPoint(A,B) \\tkzGetPoint{M}`
   - 垂足：`\\tkzDefPointBy[projection=onto A--B](C) \\tkzGetPoint{D}`
   - 直角标记：`\\tkzMarkRightAngle[size=0.4](C,D,B)`
   - 画多边形：`\\tkzDrawPolygon(A,B,C)`
   - 画线段：`\\tkzDrawSegment[dashed](C,D)`
   - 标点：`\\tkzLabelPoint[below](C){$C$}`
4. 虚线用 `dashed`

## ⚠️ 禁止
- 不要手算交点/垂足坐标，用 tkz-euclide 自动算
- 不要标注边长、角度数值
- 所有线条不带箭头
- 不要加载 ctex 或任何中文宏包

## 输出
只输出完整 LaTeX 代码，从 `\\documentclass` 到 `\\end{document}`。"""

MPL_SYSTEM = """你是中学数学配图专家。根据题目描述，生成 Python matplotlib 脚本画出配图。

## 规范
1. `import matplotlib.pyplot as plt`
2. 设置中文字体：`plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']`
3. `plt.axis('equal')` 保证几何图形比例
4. 最后 `plt.savefig('/tmp/fig.png', dpi=150, bbox_inches='tight')`
5. 不要 `plt.show()`

## ⚠️ 禁止
- **绝对不要在图上标注任何数值**（边长、角度、坐标值等）
- 只需要画图形本身和必要的标签
- 所有数值信息由题目文字提供

## 输出
只输出完整 Python 代码，不要任何解释文字。"""


# ══════════════════════════════════════════
#  代码生成
# ══════════════════════════════════════════

llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)


def generate_code(question: str, knowledge_point: str, question_type: str, engine: str) -> str:
    """调 DeepSeek V4 生成 TikZ 或 Matplotlib 代码"""
    if engine == "tikz":
        system = TIKZ_SYSTEM
    else:
        system = MPL_SYSTEM

    user_prompt = (
        f"题目：{question}\n"
        f"知识点：{knowledge_point}\n"
        f"题型：{question_type}\n\n"
        f"请用 {'TikZ (LaTeX standalone 文档)' if engine == 'tikz' else 'matplotlib Python 脚本'} 画出这道题的配图。"
    )

    resp = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    raw = resp.choices[0].message.content.strip()

    # 清理 markdown 代码块
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw


# ══════════════════════════════════════════
#  渲染引擎
# ══════════════════════════════════════════

def render_tikz(code: str, output_path: str) -> bool:
    """pdflatex → pdfcrop → convert → PNG"""
    tmpdir = tempfile.mkdtemp(prefix="tikz_")
    tex_path = os.path.join(tmpdir, "fig.tex")
    pdf_path = os.path.join(tmpdir, "fig.pdf")

    try:
        # 写 .tex
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(code)

        # pdflatex (两次编译解决交叉引用)
        for i in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
                capture_output=True, text=True, timeout=30,
            )
            # pdflatex 即使有非致命警告也 returncode≠0，以 PDF 是否生成为准
            if not os.path.exists(pdf_path):
                err_text = result.stdout[-800:] if result.stdout else result.stderr[-800:]
                log(f"  pdflatex 第{i+1}次编译失败(无PDF):\n{err_text}")
                return False

        # pdfcrop 去白边
        cropped = os.path.join(tmpdir, "fig-crop.pdf")
        subprocess.run(["pdfcrop", pdf_path, cropped], capture_output=True, timeout=10)

        # convert PDF → PNG
        src = cropped if os.path.exists(cropped) else pdf_path
        subprocess.run(
            ["convert", "-density", "300", src, "-quality", "95", output_path],
            capture_output=True, timeout=15,
        )

        return os.path.exists(output_path)

    except subprocess.TimeoutExpired:
        log("  TikZ 编译超时")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def render_matplotlib(code: str, output_path: str) -> bool:
    """python 执行 Matplotlib 脚本 → PNG"""
    tmpdir = tempfile.mkdtemp(prefix="mpl_")
    py_path = os.path.join(tmpdir, "fig.py")
    tmp_png = os.path.join(tmpdir, "fig.png")

    try:
        # 替换 savefig 路径
        code = code.replace("/tmp/fig.png", tmp_png)
        if "savefig" not in code:
            code += f"\nplt.savefig('{tmp_png}', dpi=150, bbox_inches='tight')"

        with open(py_path, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            ["python", py_path], capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            log(f"  matplotlib 错误:\n{result.stderr[-300:]}")
            return False

        if os.path.exists(tmp_png):
            shutil.move(tmp_png, output_path)
            return True
        return False

    except subprocess.TimeoutExpired:
        log("  matplotlib 超时")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════
#  核心工具
# ══════════════════════════════════════════

def generate_figure(question: str, knowledge_point: str, question_type: str = "解答题") -> dict:
    """
    生成配图：
    ① 路由决策 → TikZ/Matplotlib
    ② DeepSeek V4 生成代码
    ③ 渲染 → PNG
    """
    engine = route_engine(knowledge_point, question_type)
    log(f"路由: {engine} | {knowledge_point} / {question_type}")

    # 生成代码
    log(f"  DeepSeek 生成 {engine} 代码 ...")
    code = generate_code(question, knowledge_point, question_type, engine)
    log(f"  代码长度: {len(code)} chars")

    # 渲染
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    png_path = os.path.join(OUTPUT_DIR, f"fig_{ts}.png")

    log(f"  渲染 {engine} → PNG ...")
    if engine == "tikz":
        success = render_tikz(code, png_path)
    else:
        success = render_matplotlib(code, png_path)

    if success:
        log(f"  ✅ {png_path}")
        return {
            "success": True,
            "image_path": png_path,
            "engine": engine,
            "code": code,
        }
    else:
        log(f"  ❌ 渲染失败")
        return {
            "success": False,
            "error": f"{engine} 渲染失败",
            "engine": engine,
            "code": code,
        }


# ══════════════════════════════════════════
#  MCP Server
# ══════════════════════════════════════════

server = Server("star-tutor-figure-generator")


@server.list_tools()
async def list_tools():
    from mcp.types import Tool
    return [
        Tool(
            name="generate_figure",
            description="为一道数学题生成配图（自动选择 TikZ 或 Matplotlib，调 DeepSeek 生成代码，渲染为 PNG）",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "题目文本，包含几何描述（如：Rt△ABC中，∠C=90°，AC=6，BC=8）",
                    },
                    "knowledge_point": {
                        "type": "string",
                        "description": "知识点（如：勾股定理、一次函数）",
                    },
                    "question_type": {
                        "type": "string",
                        "enum": ["选择题", "填空题", "解答题"],
                        "default": "解答题",
                    },
                },
                "required": ["question", "knowledge_point"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    from mcp.types import TextContent
    if name == "generate_figure":
        log(f"📐 {arguments.get('knowledge_point')} / {arguments.get('question_type', '?')}")
        result = generate_figure(**arguments)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    raise ValueError(f"Unknown tool: {name}")


# ══════════════════════════════════════════
#  HTTP 路由
# ══════════════════════════════════════════

sse = SseServerTransport("/messages/")


async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())
    return JSONResponse({"status": "disconnected"})


async def health(request):
    return JSONResponse({
        "status": "ok",
        "engines": {
            "pdflatex": shutil.which("pdflatex") is not None,
            "pdfcrop": shutil.which("pdfcrop") is not None,
            "convert": shutil.which("convert") is not None,
        }
    })


app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Route("/health", endpoint=health),
    Mount("/messages/", app=sse.handle_post_message),
])

if __name__ == "__main__":
    log(f"🚀 Figure MCP Server: http://{HOST}:{PORT}/sse")
    log(f"📁 图片输出: {OUTPUT_DIR}")
    log(f"  pdflatex: {'✅' if shutil.which('pdflatex') else '❌'}  "
        f"pdfcrop: {'✅' if shutil.which('pdfcrop') else '❌'}  "
        f"convert: {'✅' if shutil.which('convert') else '❌'}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
