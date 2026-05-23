"""
IRT 三参数模型 · 知识追踪模块
根据答题序列估计学生对各知识点的掌握概率 θ ∈ [0, 1]
"""
import math
from typing import Dict


class IRT:
    """三参数 Logistic 模型

    P(答对|θ) = c + (1-c) / (1 + e^(-D*a*(θ-b)))

    θ = 学生能力（要估计的）
    a = 区分度 (default 1.0)
    b = 难度   (default 0.0)
    c = 猜测参数 (default 0.25)
    D = 1.702（常数，使 logistic 逼近正态 ogive）
    """

    D = 1.702

    def __init__(self, a: float = 1.0, b: float = 0.0, c: float = 0.25):
        self.a = a   # 区分度
        self.b = b   # 难度
        self.c = c   # 猜测参数
        self._theta: Dict[str, float] = {}  # kp_id → θ
        self._history: Dict[str, list] = {}  # kp_id → [(correct, timestamp), ...]

    def get_theta(self, kp_id: str) -> float:
        """获取知识点当前 θ，未记录默认 0.5"""
        return self._theta.get(kp_id, 0.5)

    def set_theta(self, kp_id: str, theta: float):
        """直接设置 θ（用于初始化）"""
        self._theta[kp_id] = max(0.0, min(1.0, theta))

    def prob_correct(self, theta: float) -> float:
        """给定 θ，计算答对概率"""
        exponent = -self.D * self.a * (theta - self.b)
        return self.c + (1 - self.c) / (1 + math.exp(exponent))

    def update(self, kp_id: str, correct: bool):
        """
        贝叶斯更新：根据答题对错更新 θ
        使用 EAP（期望后验估计）的简化版本
        """
        old_theta = self.get_theta(kp_id)
        step = 0.12  # 学习率（提高以加快难度爬升）

        if correct:
            # 答对 → θ 上升（但接近 1 时减速）
            gain = step * (1 - old_theta) * (1 - self.c)
            new_theta = old_theta + gain
        else:
            # 答错 → θ 下降（但接近 0 时减速）
            loss = step * old_theta * (1 - self.c)
            new_theta = old_theta - loss

        self._theta[kp_id] = max(0.0, min(1.0, new_theta))

        # 记录历史
        if kp_id not in self._history:
            self._history[kp_id] = []
        self._history[kp_id].append(correct)

    def get_difficulty(self, kp_id: str) -> str:
        """
        ZPD 最近发展区 → 出题难度决策（整体偏难）
        θ < 0.3  → 简单（基础薄弱，建立信心）
        0.3-0.6  → 中等（巩固区）
        θ > 0.6  → 困难（挑战区，综合应用题）
        """
        theta = self.get_theta(kp_id)
        if theta < 0.3:
            return "简单"
        elif theta < 0.6:
            return "中等"
        else:
            return "困难"

    def get_zone(self, kp_id: str) -> str:
        """获取学生当前所在的学习区间名称"""
        theta = self.get_theta(kp_id)
        if theta < 0.3:
            return "基础区"
        elif theta < 0.6:
            return "巩固区"
        else:
            return "挑战区"

    def get_summary(self, kp_id: str) -> dict:
        """获取知识点掌握摘要"""
        theta = self.get_theta(kp_id)
        history = self._history.get(kp_id, [])
        total = len(history)
        correct_count = sum(history) if history else 0
        return {
            "kp_id": kp_id,
            "theta": round(theta, 3),
            "zone": self.get_zone(kp_id),
            "difficulty": self.get_difficulty(kp_id),
            "total_attempts": total,
            "correct_count": correct_count,
            "accuracy": round(correct_count / total, 2) if total else 0,
        }

    def summary_all(self) -> list[dict]:
        """所有知识点摘要"""
        return [self.get_summary(kp_id) for kp_id in self._theta]


# === 快速测试 ===
if __name__ == "__main__":
    irt = IRT()

    kp = "勾股定理"

    # 初始 θ=0.5
    print(f"初始 θ={irt.get_theta(kp):.3f} → {irt.get_difficulty(kp)}")

    # 模拟 5 次答对
    for i in range(5):
        irt.update(kp, True)
    print(f"5次答对后 θ={irt.get_theta(kp):.3f} → {irt.get_difficulty(kp)}")

    # 模拟 3 次答错
    for i in range(3):
        irt.update(kp, False)
    print(f"再3次答错 θ={irt.get_theta(kp):.3f} → {irt.get_difficulty(kp)}")

    # 摘要
    summary = irt.get_summary(kp)
    print(f"\n摘要: {summary}")
