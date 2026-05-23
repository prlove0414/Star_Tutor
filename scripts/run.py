"""
端到端启动器：启动 Server → 健康检查 → 跑测试 → 清理
用法: python3 run.py
"""
import sys
import time
import signal
import asyncio
import subprocess
import urllib.request

PROJECT = "/mnt/d/AI_0114/programs_engineer/Star_Tutor"
VENV_PYTHON = "/mnt/d/star_tutor_venv/bin/python3"

server_proc = None


def start_servers():
    """启动 servers.py 子进程"""
    global server_proc
    print("🚀 启动 MCP Server...", flush=True)
    server_proc = subprocess.Popen(
        [VENV_PYTHON, f"{PROJECT}/servers.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    # 读启动日志
    for _ in range(20):
        line = server_proc.stdout.readline()
        print(f"  {line.rstrip()}", flush=True)
        if "全部就绪" in line:
            break


def wait_health():
    """等两个服务就绪"""
    for name, port in [("出题", 8765), ("诊断", 8766), ("评估", 8767)]:
        url = f"http://127.0.0.1:{port}/health"
        for _ in range(60):
            try:
                urllib.request.urlopen(url, timeout=2)
                print(f"  ✅ {name} 就绪", flush=True)
                break
            except Exception:
                time.sleep(1)
        else:
            print(f"  ❌ {name} 超时", flush=True)
            return False
    return True


async def run_tests():
    """跑测试"""
    sys.path.insert(0, PROJECT)
    from agents.teacher_agent import TeacherAgent

    agent = TeacherAgent()
    try:
        tests = [
            ("出题", "给我一道勾股定理的中等难度解答题"),
            ("诊断", "学生做勾股定理的题老错，帮我诊断一下根因"),
            ("搜索", "有哪些关于函数的知识点？"),
        ]
        for label, msg in tests:
            print(f"\n{'='*60}", flush=True)
            print(f"🧪 [{label}] 👤 {msg}", flush=True)
            try:
                reply = await agent.chat(msg)
                reply_text = reply["reply"] if isinstance(reply, dict) else reply
                print(f"🤖 教师:\n{reply_text}", flush=True)
            except Exception as e:
                print(f"❌ 失败: {e}", flush=True)
            await asyncio.sleep(2)
    finally:
        await agent.close()
    print(f"\n{'='*60}")
    print("🎉 全部完成", flush=True)


def cleanup():
    global server_proc
    if server_proc:
        print("🛑 关闭 Server...", flush=True)
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


def main():
    signal.signal(signal.SIGINT, lambda s, f: (cleanup(), sys.exit(0)))

    start_servers()
    if not wait_health():
        cleanup()
        return

    print("\n📋 运行测试\n", flush=True)
    try:
        asyncio.run(run_tests())
    finally:
        cleanup()


if __name__ == "__main__":
    main()
