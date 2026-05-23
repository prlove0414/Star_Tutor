"""
L3 解答正确性门禁测试

测试场景：
  1. 多路径求解：发题后验证后台异步求解完成
  2. 程序化评估：学生给答案后系统直接调 evaluate_answer，不等 LLM
  3. 自疑纠错：同一错误 ×2 触发 L3-1 异步重算
  4. L3→L4 上报：同一错误 ×3 触发 L4 锁定

运行方式：
  docker compose up -d   # 先启动服务
  python tests/test_harness_l3.py
"""

import time
import requests
import sys

BASE = "http://localhost:8000"


def test(name: str) -> str:
    """打印测试标题"""
    print(f"\n{'='*60}")
    print(f"🧪 {name}")
    print(f"{'='*60}")
    return name


def assert_true(cond, msg):
    """断言"""
    if cond:
        print(f"  ✅ {msg}")
    else:
        print(f"  ❌ {msg}")
        sys.exit(1)


def create_session() -> str:
    """创建新会话"""
    r = requests.post(f"{BASE}/api/session/new")
    assert r.status_code == 200, f"创建会话失败: {r.status_code}"
    sid = r.json()["session_id"]
    print(f"  📝 会话: {sid}")
    return sid


def chat(sid: str, msg: str) -> dict:
    """发送消息"""
    r = requests.post(f"{BASE}/api/chat", json={"session_id": sid, "message": msg})
    assert r.status_code == 200, f"发送消息失败: {r.status_code}"
    data = r.json()
    reply = data.get("reply", "")
    locked = data.get("locked", False)
    print(f"  👤 {msg[:50]}")
    print(f"  🤖 {reply[:80]}")
    if locked:
        print(f"  🔒 会话已锁定!")
    return data


def upload_question(sid: str, question: str, knowledge_point: str = "二元一次方程组的解法") -> dict:
    """上传题目（模拟 Vision 识别后的文本）"""
    msg = f"[系统] 学生上传了一道题的图片。Vision 已识别如下：\n\n{question}\n（知识点：{knowledge_point}）"
    return chat(sid, msg)


# ══════════════════════════════════════════
# 测试用例
# ══════════════════════════════════════════

def test_case_1_multi_path_solve():
    """案例1: L3 多路径求解正常完成"""
    test("L3-1 多路径求解")
    sid = create_session()

    # 上传题目 → 触发后台异步求解
    upload_question(
        sid,
        "解方程组：\\begin{cases} 3x - 2y = 5, \\\\ x - y = -1. \\end{cases}",
    )

    # 等后台求解完成
    print("  ⏳ 等待后台求解完成 (~15s)...")
    time.sleep(15)

    # 发一条消息确认答案已注入
    data = chat(sid, "我试一下")
    reply = data["reply"]

    # 验证 Teacher 没有直接给答案
    assert_true("x=7" not in reply and "y=8" not in reply,
                "Teacher 没有泄露答案")
    assert_true(len(reply) > 10,
                "Teacher 给出了有意义的引导回复")

    print("  🎉 L3 多路径求解: 通过")


def test_case_2_programmatic_eval():
    """案例2: L3 程序化评估 — 代码层直接调 evaluate_answer"""
    test("L3-2 程序化评估")
    sid = create_session()

    upload_question(
        sid,
        "解方程组：\\begin{cases} 3x - 2y = 5, \\\\ x - y = -1. \\end{cases}",
    )

    # 等后台求解
    time.sleep(12)

    # 学生给答案 → 系统应程序化评估，不等 LLM
    data = chat(sid, "x=y=1")
    reply = data["reply"]

    # 评估已代码层完成，Teacher 基于结果引导
    # 关键：回复中不应出现 Teacher 调用 evaluate_answer 的痕迹
    # （因为评估是代码层做的，直接注入了结果）
    assert_true(len(reply) > 10,
                "程序化评估后 Teacher 正常回复")
    assert_true("x=7" not in reply and "y=8" not in reply,
                "Teacher 没有泄露答案")

    print("  🎉 L3 程序化评估: 通过")


def test_case_3_self_doubt_and_lock():
    """案例3: L3 自疑 → L4 锁定（完整链条）"""
    test("L3-3 自疑 → L4 锁定")
    sid = create_session()

    upload_question(
        sid,
        "解方程组：\\begin{cases} 3x - 2y = 5, \\\\ x - y = -1. \\end{cases}",
    )

    time.sleep(12)

    # 第 1 次错误
    data1 = chat(sid, "x=y=1")
    assert_true(not data1.get("locked"),
                "第1次错误不应锁定")

    # 第 2 次错误 — 应触发 L3-1 自疑
    data2 = chat(sid, "x=y=1")
    assert_true(not data2.get("locked"),
                "第2次错误不锁定（L3-1 自疑但不锁）")
    # 等异步重算
    time.sleep(8)

    # 第 3 次错误 — 应触发 L3→L4 锁定
    data3 = chat(sid, "x=y=1")
    assert_true(data3.get("locked"),
                "第3次同错应触发 L4 锁定")

    print("  🎉 L3 自疑 → L4 锁定: 通过")


# ══════════════════════════════════════════

if __name__ == "__main__":
    print("🐧 星学伴 Harness L3 测试")
    print(f"   服务地址: {BASE}")

    # 健康检查
    try:
        r = requests.get(f"{BASE}/api/health", timeout=5)
        print(f"   服务状态: {'✅ 在线' if r.status_code == 200 else '❌ 异常'}")
    except Exception:
        print("   ❌ 服务不可达，请先 docker compose up -d")
        sys.exit(1)

    test_case_1_multi_path_solve()
    test_case_2_programmatic_eval()
    test_case_3_self_doubt_and_lock()

    print(f"\n{'='*60}")
    print("🎉 L3 全量测试通过！")
