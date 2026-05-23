"""
Harness 测试用例：30 题覆盖 6 核心知识点 × 3 难度 × 2 题型
"""
TEST_CASES = [
    # === 勾股定理 ===
    {"knowledge_point": "勾股定理", "difficulty": "简单", "question_type": "解答题"},
    {"knowledge_point": "勾股定理", "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "勾股定理", "difficulty": "困难", "question_type": "解答题"},
    {"knowledge_point": "勾股定理", "difficulty": "中等", "question_type": "选择题"},
    {"knowledge_point": "勾股定理", "difficulty": "简单", "question_type": "填空题"},

    # === 一元二次方程 ===
    {"knowledge_point": "一元二次方程", "difficulty": "简单", "question_type": "解答题"},
    {"knowledge_point": "一元二次方程", "difficulty": "中等", "question_type": "选择题"},
    {"knowledge_point": "一元二次方程", "difficulty": "困难", "question_type": "解答题"},
    {"knowledge_point": "一元二次方程", "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "一元二次方程", "difficulty": "简单", "question_type": "填空题"},

    # === 相似三角形 ===
    {"knowledge_point": "相似三角形", "difficulty": "简单", "question_type": "选择题"},
    {"knowledge_point": "相似三角形", "difficulty": "中等", "question_type": "解答题"},
    {"knowledge_point": "相似三角形", "difficulty": "困难", "question_type": "解答题"},
    {"knowledge_point": "相似三角形", "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "相似三角形", "difficulty": "简单", "question_type": "解答题"},

    # === 一次函数 ===
    {"knowledge_point": "一次函数", "difficulty": "简单", "question_type": "填空题"},
    {"knowledge_point": "一次函数", "difficulty": "中等", "question_type": "解答题"},
    {"knowledge_point": "一次函数", "difficulty": "困难", "question_type": "选择题"},
    {"knowledge_point": "一次函数", "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "一次函数", "difficulty": "简单", "question_type": "选择题"},

    # === 概率 ===
    {"knowledge_point": "概率", "difficulty": "简单", "question_type": "选择题"},
    {"knowledge_point": "概率", "difficulty": "中等", "question_type": "解答题"},
    {"knowledge_point": "概率", "difficulty": "困难", "question_type": "填空题"},
    {"knowledge_point": "概率", "difficulty": "中等", "question_type": "选择题"},
    {"knowledge_point": "概率", "difficulty": "简单", "question_type": "解答题"},

    # === 分式方程 ===
    {"knowledge_point": "分式方程", "difficulty": "简单", "question_type": "填空题"},
    {"knowledge_point": "分式方程", "difficulty": "中等", "question_type": "解答题"},
    {"knowledge_point": "分式方程", "difficulty": "困难", "question_type": "解答题"},
    {"knowledge_point": "分式方程", "difficulty": "中等", "question_type": "填空题"},
    {"knowledge_point": "分式方程", "difficulty": "简单", "question_type": "选择题"},
]

SMOKE_TEST = TEST_CASES[:6]  # 冒烟测试：前 6 题
