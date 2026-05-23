"""
L4 学生行为管控测试

测试场景：
  1. 非学习内容 1 次提醒：学生发闲聊 → 温和提醒
  2. 非学习内容 2 次锁定：再发闲聊 → 会话锁定
  3. 锁后话语统一：锁后再发消息 → 回复与锁定时完全一致
  4. 学习内容放行：正常数学题不受影响

运行方式：
  docker compose up -d
  python tests/test_harness_l4.py
"""

import sys
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


def assert_equal(a, b, msg):
    if a == b:
        print(f"  ✅ {msg}: {a!r}")
    else:
        print(f"  ❌ {msg}: 期望 {b!r}，实际 {a!r}")
        sys.exit(1)


def create_session():
    r = requests.post(f"{BASE}/api/session/new")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    print(f"  📝 会话: {sid}")
    return sid


def chat(sid: str, msg: str) -> dict:
    r = requests.post(f"{BASE}/api/chat", json={"session_id": sid, "message": msg})
    assert r.status_code == 200
    data = r.json()
    print(f"  👤 {msg[:50]}")
    print(f"  🤖 {data['reply'][:80]}")
    if data.get("locked"):
        print(f"  🔒 已锁定")
    return data


# ══════════════════════════════════════════

def test_case_1_reminder():
    """案例1: 非学习内容 → 1次提醒"""
    test("L4-1 非学习提醒")
    sid = create_session()

    data = chat(sid, "你是谁")

    assert_true(not data.get("locked"),
                "第1次非学习不应锁定")
    assert_true("数学" in data["reply"] or "学习" in data["reply"],
                "提醒内容包含学习引导")

    print("  🎉 非学习提醒: 通过")


def test_case_2_lock():
    """案例2: 非学习内容 ×2 → 锁定"""
    test("L4-2 非学习锁定")
    sid = create_session()

    # 第1次 → 提醒
    chat(sid, "校长一个月多少工资")
    # 第2次 → 锁定
    data = chat(sid, "你是班主任吗")

    assert_true(data.get("locked"),
                "第2次非学习应锁定会话")

    print("  🎉 非学习锁定: 通过")


def test_case_3_consistent_lock_reply():
    """案例3: 锁后话语统一"""
    test("L4-3 锁后话语统一")
    sid = create_session()

    # 触发锁定
    chat(sid, "我想打篮球")
    lock_data = chat(sid, "今天天气真好")
    lock_reply = lock_data["reply"]

    assert_true(lock_data.get("locked"),
                "第2次非学习锁定")

    # 锁后再发 → 应返回完全相同的回复
    post_lock = chat(sid, "我明天不想学数学")
    assert_equal(post_lock["reply"], lock_reply,
                 "锁后回复与锁定时完全一致")

    print("  🎉 锁后话语统一: 通过")


def test_case_4_learning_passthrough():
    """案例4: 学习内容正常放行"""
    test("L4-4 学习内容放行")
    sid = create_session()

    # 直接发数学题，不应被拦截
    import time
    data = chat(
        sid,
        "[系统] Vision 已识别：解方程 2x + 3 = 7（知识点：一元一次方程）",
    )

    assert_true(not data.get("locked"),
                "学习内容不应被拦截")
    assert_true(len(data["reply"]) > 10,
                "Teacher 正常回复")

    print("  🎉 学习内容放行: 通过")


# ══════════════════════════════════════════

if __name__ == "__main__":
    print("🐧 星学伴 Harness L4 测试")
    print(f"   服务地址: {BASE}")

    try:
        r = requests.get(f"{BASE}/api/health", timeout=5)
        print(f"   服务状态: {'✅ 在线' if r.status_code == 200 else '❌ 异常'}")
    except Exception:
        print("   ❌ 服务不可达，请先 docker compose up -d")
        sys.exit(1)

    test_case_1_reminder()
    test_case_2_lock()
    test_case_3_consistent_lock_reply()
    test_case_4_learning_passthrough()

    print(f"\n{'='*60}")
    print("🎉 L4 全量测试通过！")
