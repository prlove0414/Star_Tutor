"""
一键启动所有 MCP Server（出题 + 诊断 + 评估）
显示实时启动状态
"""
import os
import sys
import signal
import subprocess
import time
import threading
import urllib.request

PROJECT = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = sys.executable

# 仅在 WSL 环境需要显式指定 CUDA 库路径，Docker 中 CUDA 已自动配置
_cuda_env = {}
_wsl_lib = "/usr/lib/wsl/lib"
if os.path.exists(_wsl_lib):
    _cuda_env = {"LD_LIBRARY_PATH": _wsl_lib}

SERVERS = [
    ("出题",   "agents/mcp_question_server.py",   8765, _cuda_env),
    ("诊断",   "agents/mcp_diagnosis_server.py",  8766, {}),
    ("评估",   "agents/mcp_evaluation_server.py", 8767, {}),
    ("Vision", "agents/mcp_vision_server.py",      8768, {}),
    ("配图",   "agents/mcp_figure_server.py",      8769, {}),
]

processes = []
status = {name: "⏳" for name, _, _, _ in SERVERS}


def print_status():
    """打印当前状态条"""
    parts = [f"{s} {n}" for n, _, _, _ in SERVERS for s in [status[n]]]
    print(f"\r  {'  '.join(parts)}", end="", flush=True)


def wait_health(name: str, port: int):
    """轮询直到 /health 响应"""
    url = f"http://127.0.0.1:{port}/health"
    while True:
        try:
            urllib.request.urlopen(url, timeout=2)
            status[name] = "✅"
            print_status()
            return
        except Exception:
            time.sleep(0.5)


def main():
    print("🚀 启动 MCP Server...", flush=True)

    # 启动所有子进程
    for name, script, port, env_extra in SERVERS:
        env = {**os.environ, **env_extra}
        proc = subprocess.Popen(
            [VENV_PYTHON, script],
            cwd=PROJECT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        processes.append((name, proc))

    # 初始状态
    print_status()

    # 后台线程等每个 Server 就绪
    for name, _, port, _ in SERVERS:
        threading.Thread(target=wait_health, args=(name, port), daemon=True).start()

    print("\n\n按 Ctrl+C 全部关闭", flush=True)

    def shutdown(sig=None, frame=None):
        print("\n🛑 关闭...", flush=True)
        for name, proc in processes:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("👋 已关闭", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
