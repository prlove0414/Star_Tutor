#!/usr/bin/env python3
"""
蒸馏数据清洗：过滤失败条目 → 转 Alpaca 格式 → 80/20 切分
用途：让 4B student 学习模仿 9B teacher 的出题能力
"""
import json, random, os

# ============================================
# 配置
# ============================================
RAW     = "/mnt/d/AI_0114/programs_engineer/测试结果/raw_data.jsonl"
OUT_DIR = "/mnt/d/AI_0114/programs_engineer/素材/data/distill"

SYSTEM_PROMPT = (
    "你是一个初中数学出题助手。根据用户指定的知识点和难度，生成一道完整的数学题目，"
    "包含题目描述、详细解题步骤和最终答案。\n"
    "直接输出JSON。格式：{\"question\":\"题目\",\"solution\":\"解题步骤\",\"answer\":\"答案\"}"
)

TRAIN_RATIO = 0.8
SEED = 42

# ============================================
# 主流程
# ============================================
random.seed(SEED)

# 1. 读取
records = []
with open(RAW, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line.strip())
        if d["status"] == "ok":
            records.append(d)

print(f"读取: {records[0]['id']}→{records[-1]['id']} 共 {len(records)} 条有效")

# 2. 转 Alpaca 格式
alpaca = []
for d in records:
    alpaca.append({
        "instruction": SYSTEM_PROMPT,
        "input": f"知识点：{d['knowledge_point']}，难度：{d['difficulty']}，题型：{d['question_type']}",
        "output": json.dumps({
            "question": d["question"],
            "solution": d["solution"],
            "answer": d["answer"],
        }, ensure_ascii=False),
    })

# 3. 按 KP/diff 分层抽样切分（保证每个组合在 train/val 都有代表）
from collections import defaultdict
groups = defaultdict(list)
for item in alpaca:
    key = item["input"]  # "知识点：勾股定理，难度：简单，题型：解答题"
    groups[key].append(item)

train, val = [], []
for items in groups.values():
    random.shuffle(items)
    split = max(1, int(len(items) * TRAIN_RATIO))
    train.extend(items[:split])
    val.extend(items[split:])

random.shuffle(train)
random.shuffle(val)

# 4. 写入
os.makedirs(OUT_DIR, exist_ok=True)

for name, data in [("train.json", train), ("val.json", val)]:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ {name}: {len(data)} 条 → {path}")

# 5. 统计
print(f"\n总计: {len(train)}+{len(val)}={len(train)+len(val)} ({len(train)/len(alpaca)*100:.0f}/{len(val)/len(alpaca)*100:.0f})")
