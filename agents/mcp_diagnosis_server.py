"""
诊断 Agent · MCP Server (HTTP)
- 连接 AuraDB 知识图谱
- 暴露诊断工具：前置链追溯、知识点查找
"""
import json
import sys
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn

sys.path.insert(0, "/mnt/d/AI_0114/programs_engineer/Star_Tutor")
from agents.kg_client import KGClient

# === 配置 ===
HOST = "127.0.0.1"
PORT = 8766


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# === 初始化 KG 客户端 ===
log("⏳ 连接知识图谱 (AuraDB)...")
kg = KGClient()
log(f"✅ 诊断 Agent 就绪 | http://{HOST}:{PORT}")


# === MCP Server ===
server = Server("star-tutor-diagnosis")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="trace_prerequisites",
            description="追溯一个知识点的完整前置学习链，从基础知识到当前知识点。用于诊断学生错误的根因——顺着链路往前找最早薄弱环节。",
            inputSchema={
                "type": "object",
                "properties": {
                    "knowledge_point": {"type": "string", "description": "知识点名称，例如：勾股定理、二次函数"},
                },
                "required": ["knowledge_point"],
            },
        ),
        Tool(
            name="find_kp",
            description="在知识图谱中模糊搜索知识点。",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["keyword"],
            },
        ),
        Tool(
            name="get_kp_detail",
            description="通过 ID 获取知识点的详细信息（所属章节、领域、描述）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "kp_id": {"type": "string", "description": "知识点的结构化 ID"},
                },
                "required": ["kp_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "trace_prerequisites":
            kp_name = arguments["knowledge_point"]
            kp = kg.find_kp(kp_name)
            if not kp:
                suggestions = kg.search(kp_name) if kg else []
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"未找到知识点「{kp_name}」", "suggestions": [r["name"] for r in suggestions]},
                    ensure_ascii=False
                ))]
            chain = kg.get_prerequisites(kp["name"])
            return [TextContent(type="text", text=json.dumps({
                "kp": kp,
                "prerequisite_chain": chain,
                "chain_display": " → ".join(chain) if chain else "无前置链",
            }, ensure_ascii=False))]

        elif name == "find_kp":
            results = kg.search(arguments["keyword"])
            enriched = []
            for r in results[:5]:
                chain = kg.get_prerequisites(r["name"]) if r.get("type") == "knowledge_point" else []
                enriched.append({**r, "prerequisite_chain": chain, "chain_display": " → ".join(chain) if chain else "无前置链"})
            return [TextContent(type="text", text=json.dumps({"results": enriched, "count": len(results)}, ensure_ascii=False))]

        elif name == "get_kp_detail":
            detail = kg.get_kp_by_id(arguments["kp_id"])
            if not detail:
                return [TextContent(type="text", text=json.dumps({"error": f"未找到 ID: {arguments['kp_id']}"}))]
            return [TextContent(type="text", text=json.dumps(detail, ensure_ascii=False))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": f"知识图谱查询失败: {str(e)[:200]}", "tool": name}, ensure_ascii=False))]

    raise ValueError(f"Unknown tool: {name}")


# === HTTP 路由 ===
sse = SseServerTransport("/messages/")


async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


async def health(request):
    return JSONResponse({"status": "ok", "agent": "diagnosis"})


app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Route("/health", endpoint=health),
    Mount("/messages/", app=sse.handle_post_message),
])

if __name__ == "__main__":
    log(f"🚀 诊断 Agent: http://{HOST}:{PORT}/sse")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
