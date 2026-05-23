"""
星学伴 · FastAPI 统筹服务 (Port 8000)
启动时自动拉起 5 个 MCP Server，暴露 REST API + 简易聊天页面
"""
import os, sys, json, uuid, time, subprocess, signal, urllib.request, threading
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn, tempfile

# === 配置 ===
PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

HOST = "0.0.0.0"
PORT = 8000
MCP_PORTS = [8765, 8766, 8767, 8768, 8769]
SESSIONS_DIR = os.path.join(PROJECT, "data", "sessions")

server_proc = None
sessions: dict[str, "TeacherAgent"] = {}
# 会话元数据：{id: {title, created_at, updated_at, msg_count}}
sessions_meta: dict[str, dict] = {}


def log(msg: str):
    print(f"[APP] {msg}", flush=True)


# ══════════════════════════════════════════
#  生命周期
# ══════════════════════════════════════════

def start_mcp_servers():
    global server_proc
    log("🚀 启动 MCP Server...")
    server_proc = subprocess.Popen(
        [sys.executable, os.path.join(PROJECT, "servers.py")],
        cwd=PROJECT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for port in MCP_PORTS:
        url = f"http://127.0.0.1:{port}/health"
        timeout = 300 if port == 8765 else 90
        for _ in range(timeout):
            try:
                urllib.request.urlopen(url, timeout=2)
                log(f"  ✅ MCP :{port} 就绪")
                break
            except Exception:
                time.sleep(1)
        else:
            log(f"  ❌ MCP :{port} 超时！")


def stop_mcp_servers():
    global server_proc
    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        log("🛑 MCP 已关闭")


def load_sessions_meta():
    """从磁盘恢复会话列表"""
    global sessions_meta
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".json"):
            sid = fname[:-5]
            try:
                with open(os.path.join(SESSIONS_DIR, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions_meta[sid] = data.get("meta", {})
            except Exception:
                pass
    log(f"📂 加载 {len(sessions_meta)} 个历史会话")


def save_session(sid: str, agent):
    """保存会话到磁盘"""
    meta = {
        "id": sid,
        "title": sessions_meta.get(sid, {}).get("title", "新对话"),
        "created_at": sessions_meta.get(sid, {}).get("created_at", datetime.now().isoformat()),
        "updated_at": datetime.now().isoformat(),
        "msg_count": len(agent.messages),
    }
    sessions_meta[sid] = meta
    data = {"meta": meta, "messages": agent.messages}
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    with open(os.path.join(SESSIONS_DIR, f"{sid}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ══════════════════════════════════════════
#  FastAPI App
# ══════════════════════════════════════════

app = FastAPI(title="星学伴 API", version="1.0")


@app.on_event("startup")
async def startup():
    load_sessions_meta()
    # MCP 后台启动，不阻塞前端
    threading.Thread(target=start_mcp_servers, daemon=True).start()
    log("✅ 星学伴 前端就绪（MCP 后台加载中...")


@app.on_event("shutdown")
async def shutdown():
    stop_mcp_servers()


# ══════════════════════════════════════════
#  API
# ══════════════════════════════════════════

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    images: list[str] = []


class RenameRequest(BaseModel):
    title: str


@app.get("/api/sessions")
async def list_sessions():
    """返回所有会话列表（按更新时间倒序，过滤空会话）"""
    items = [m for m in sessions_meta.values() if m.get("msg_count", 0) > 0]
    items.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
    return {"sessions": items}


@app.post("/api/session/new")
async def new_session():
    """创建新会话"""
    from agents.teacher_agent import TeacherAgent

    sid = uuid.uuid4().hex[:12]
    sessions[sid] = TeacherAgent()
    now = datetime.now().isoformat()
    sessions_meta[sid] = {"id": sid, "title": "新对话", "created_at": now, "updated_at": now, "msg_count": 0}
    # 不立即存盘，等第一条消息发送后再存
    log(f"📝 新会话: {sid}")
    return {"session_id": sid, "message": "你好！我是星学伴AI教师，今天想讨论什么数学问题？"}


@app.get("/api/session/{sid}/history")
async def session_history(sid: str):
    """恢复历史会话的消息记录"""
    fpath = os.path.join(SESSIONS_DIR, f"{sid}.json")
    if not os.path.exists(fpath):
        raise HTTPException(404, "会话不存在")

    # 如果不在内存，重新加载 Agent 并回放消息
    if sid not in sessions:
        from agents.teacher_agent import TeacherAgent
        agent = TeacherAgent()
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        agent.messages = data.get("messages", [])
        sessions[sid] = agent

    return {"messages": sessions[sid].messages}


@app.delete("/api/session/{sid}")
async def delete_session(sid: str):
    """删除会话"""
    if sid not in sessions_meta:
        raise HTTPException(404, "会话不存在")
    sessions.pop(sid, None)
    sessions_meta.pop(sid, None)
    fpath = os.path.join(SESSIONS_DIR, f"{sid}.json")
    if os.path.exists(fpath):
        os.remove(fpath)
    log(f"🗑 删除会话: {sid}")
    return {"ok": True}


@app.patch("/api/session/{sid}/rename")
async def rename_session(sid: str, req: RenameRequest):
    """重命名会话"""
    if sid not in sessions_meta:
        raise HTTPException(404, "会话不存在")
    sessions_meta[sid]["title"] = req.title
    fpath = os.path.join(SESSIONS_DIR, f"{sid}.json")
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["meta"]["title"] = req.title
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    log(f"✏ 重命名会话 [{sid}]: {req.title}")
    return {"ok": True}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """学生发消息 → 教师回复 + 配图"""
    agent = sessions.get(req.session_id)
    if not agent:
        # 尝试从磁盘恢复
        fpath = os.path.join(SESSIONS_DIR, f"{req.session_id}.json")
        if os.path.exists(fpath):
            from agents.teacher_agent import TeacherAgent
            agent = TeacherAgent()
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            agent.messages = data.get("messages", [])
            agent.turn = len(agent.messages)
            sessions[req.session_id] = agent
        else:
            raise HTTPException(404, "会话不存在")

    log(f"💬 [{req.session_id}] {req.message[:60]}...")

    # 第一条用户消息设为会话标题
    if len(agent.messages) == 0:
        title = req.message[:30]
        sessions_meta[req.session_id]["title"] = title

    result = await agent.chat(req.message)
    reply = result["reply"] if isinstance(result, dict) else result
    images = result.get("images", []) if isinstance(result, dict) else []

    # 图片路径转相对 URL
    img_urls = []
    for img in images:
        fname = os.path.basename(img)
        img_urls.append(f"/api/figures/{fname}")

    save_session(req.session_id, agent)
    return ChatResponse(session_id=req.session_id, reply=reply, images=img_urls)


@app.get("/api/figures/{filename}")
async def serve_figure(filename: str):
    fig_path = os.path.join(PROJECT, "data", "figures", filename)
    if not os.path.isfile(fig_path):
        raise HTTPException(404, "图片不存在")
    return FileResponse(fig_path)


@app.post("/api/upload/{session_id}")
async def upload_image(session_id: str, image: UploadFile = File(...), message: str = Form(None)):
    """拍照上传（可选附带文字）→ Vision MCP 识别 → Teacher 引导"""
    content = await image.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(content)
    tmp.close()

    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client("http://127.0.0.1:8768/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("recognize_question", {"image_path": tmp.name})
            data = json.loads(result.content[0].text)

    os.unlink(tmp.name)

    # 用户附带的文字消息
    user_text = f'学生说："{message}"\n\n' if message and message.strip() else ""

    content_type = data.get("content_type", "question")
    figure_desc = data.get("figure_description", "")
    if content_type == "handwritten_work":
        text = data.get("text", "（图片识别失败）")
        fmt = data.get("format", "手写内容")
        kp = data.get("knowledge_point", "")
        note = data.get("note", "")
        msg = f"[系统] 学生上传了手写{fmt}。{user_text}Vision 已识别如下：\n\n{text}"
        if figure_desc:
            msg += f"\n\n【图片中的图形】{figure_desc}"
        if kp:
            msg += f"\n（知识点：{kp}）"
        if note:
            msg += f"\n（识别备注：{note}）"
    else:
        # 默认：题目识别
        question = data.get("question", "（图片识别失败）")
        kp = data.get("knowledge_point", "")
        options = data.get("options", [])
        question_type = data.get("question_type", "")
        msg = f"[系统] 学生上传了一道题的图片。{user_text}Vision 已识别如下：\n\n{question}"
        if options:
            msg += "\n\n选项：\n" + "\n".join(options)
        if figure_desc:
            msg += f"\n\n【题目配图描述】{figure_desc}"
        if kp:
            msg += f"\n（知识点：{kp}）"

    agent = sessions.get(session_id)
    if not agent:
        raise HTTPException(404, "会话不存在")

    if len(agent.messages) == 0:
        title = data.get("question", data.get("text", "新对话"))[:30]
        sessions_meta[session_id]["title"] = title

    result = await agent.chat(msg)
    reply = result["reply"] if isinstance(result, dict) else result
    images = result.get("images", []) if isinstance(result, dict) else []
    img_urls = [f"/api/figures/{os.path.basename(i)}" for i in images]

    save_session(session_id, agent)
    return ChatResponse(session_id=session_id, reply=reply, images=img_urls)


@app.get("/api/health")
async def health():
    status = {}
    for port in MCP_PORTS:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            status[str(port)] = "ok"
        except Exception:
            status[str(port)] = "down"
    return {"status": "ok" if all(v == "ok" for v in status.values()) else "degraded",
            "mcp": status, "sessions": len(sessions)}


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


# ══════════════════════════════════════════
#  静态文件
# ══════════════════════════════════════════

STATIC_DIR = os.path.join(PROJECT, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
