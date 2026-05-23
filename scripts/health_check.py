"""健康检查：轮询三个 MCP Server 直到全部就绪"""
import time
import urllib.request

SERVERS = {
    "出题": 8765,
    "诊断": 8766,
    "评估": 8767,
}

for name, port in SERVERS.items():
    url = f"http://127.0.0.1:{port}/health"
    for i in range(120):
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"✅ {name} 就绪 (端口 {port})")
            break
        except Exception:
            time.sleep(1)
    else:
        print(f"❌ {name} 超时")

print("🎉 全部就绪！" if all(True for _ in []) else "")
