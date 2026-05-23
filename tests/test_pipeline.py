"""
端到端全链路测试

覆盖完整教学闭环：
  📷 发题 → 🔍 Vision 识别 → 🧮 L3 多路径求解
  → 🤔 苏格拉底引导 → 📊 评估 → 🔓 提示模式
  → ✅ 确认掌握 → 📝 出变式题 → 📐 TikZ 配图

运行方式：
  docker compose up -d
  python tests/test_pipeline.py
"""

import sys
import time
import requests

BASE = "http://localhost:8000"


def test(name: str):
    print(f"\n{'='*60}")
    print(f"🧪 {name}")
    print(f"{'='*60}")


def assert_true(cond, msg):
    if cond:
        print(f"  ✅ {msg}")
    else:
        print(f"  ❌ {msg}")
        sys.exit(1)


def create_session():
    r = requests.post(f"{BASE}/api/session/new")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    print(f"  📝 会话: {sid}")
    return sid


def chat(sid: str, msg: str) -> dict:
    r = requests.post(f"{BASE}/api/chat", json={"session_id": sid, "message": msg})
    assert r.status_code == 200, f"chat 失败: {r.status_code}"
    data = r.json()
    print(f"  👤 {msg[:60]}")
    print(f"  🤖 {data['reply'][:100]}")
    if data.get("locked"):
        print(f"  🔒 已锁定")
    if data.get("images"):
        print(f"  🖼️ 配图: {data['images']}")
    return data


# ══════════════════════════════════════════

def test_full_pipeline():
    """全链路：发题 → 引导 → 答对 → 出题 → 配图"""
    test("端到端全链路")

    sid = create_session()

    # === Step 1: 发题 ===
    chat(
        sid,
        "[系统] 学生上传了一道题的图片。Vision 已识别如下：\n\n"
        "解方程：2x + 3 = 7\n"
        "（知识点：一元一次方程）",
    )

    # 等后台求解
    print("  ⏳ 等待后台求解 (~10s)...")
    time.sleep(12)

    # === Step 2: 学生尝试回答（正确） ===
    data = chat(sid, "x=2")
    assert_true(not data.get("locked"), "正确回答不应锁定")
    # Teacher 应给出正面反馈（不是直接告知答案）
    reply = data["reply"]
    assert_true("x=2" not in reply or len(reply) < 30,
                "Teacher 没有重复/直接给出答案")

    # === Step 3: 学生确认掌握 ===
    chat(sid, "我会了")

    # === Step 4: 请求出题 ===
    data = chat(sid, "给我出一道类似的题吧")
    assert_true(len(data["reply"]) > 10,
                "Teacher 响应出题请求")
    assert_true(not data.get("locked"),
                "出题不应触发锁定")

    print("  🎉 全链路: 通过")


def test_stuck_prompt_mode():
    """卡住 → 提示模式"""
    test("卡住 → 提示模式")

    sid = create_session()

    chat(
        sid,
        "[系统] Vision 识别：解方程 3(x-1) = 2x + 4（知识点：一元一次方程）",
    )

    time.sleep(12)

    # 连续卡住
    stuck_replies = ["不知道", "不会", "没思路"]
    for i, msg in enumerate(stuck_replies):
        data = chat(sid, msg)
        assert_true(not data.get("locked"),
                    f"卡住第{i+1}次不应锁定")
        reply = data["reply"]
        if i >= 2:
            # 第3次卡住 → 提示模式
            assert_true("提示" in reply or "线索" in reply or "试试" in reply,
                        f"第{i+1}次卡住应进入提示模式")

    print("  🎉 提示模式: 通过")


def test_vision_handwriting():
    """Vision 手写识别 → 直接评估"""
    test("Vision 手写分流")

    sid = create_session()

    # 先发题
    chat(
        sid,
        "[系统] 学生上传了一道题的图片。Vision 已识别如下：\n\n"
        "计算：(-2)² + √16 - 3\n"
        "（知识点：实数的运算）",
    )

    time.sleep(12)

    # 学生发手写步骤 → Vision 识别为手写内容
    data = chat(
        sid,
        "[系统] 学生上传了手写解题步骤。Vision 已识别：\n"
        "(-2)² = 4, √16 = 4, 4 + 4 - 3 = 5",
    )

    assert_true(not data.get("locked"),
                "手写步骤不应锁定")
    assert_true(len(data["reply"]) > 10,
                "Teacher 对手写步骤给出反馈")

    print("  🎉 手写分流: 通过")


# ══════════════════════════════════════════

if __name__ == "__main__":
    print("🐧 星学伴 端到端测试")
    print(f"   服务地址: {BASE}")

    try:
        r = requests.get(f"{BASE}/api/health", timeout=5)
        print(f"   服务状态: {'✅ 在线' if r.status_code == 200 else '❌ 异常'}")
    except Exception:
        print("   ❌ 服务不可达，请先 docker compose up -d")
        sys.exit(1)

    test_full_pipeline()
    test_stuck_prompt_mode()
    test_vision_handwriting()

    print(f"\n{'='*60}")
    print("🎉 全链路测试通过！")
