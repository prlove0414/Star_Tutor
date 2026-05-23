# 星学伴 · 评估与实验文档

> Star Tutor Evaluation & Experimentation

本目录包含星学伴项目从模型选型、LoRA 微调消融、知识蒸馏到 Agent 行为门禁的**完整实验记录**。

---

## 📂 目录结构

```
evaluation/
├── README.md                          ← 你在这
├── lora_ablation/                     ← LoRA 消融实验
├── distillation/                      ← 知识蒸馏
├── agent_gate/                        ← Agent 行为门禁
└── irt/                               ← IRT 知识追踪评估
```

---

## 一、LoRA 消融实验 `lora_ablation/`

**目的**：验证不同 LoRA rank 对出题质量的影响，选定最优方案。

| 文件 | 说明 |
|------|------|
| `README.md` | 实验报告（原 `lora_comparison_report.md`） |
| `compare_loras.py` | 多模型对比运行脚本 |
| `harness.py` | Harness 评估框架核心 |
| `judge.py` | DeepSeek V4 裁判模型 |
| `tasks.py` | L1 出题评估任务定义 |
| `harness_report_baseline.json` | 基座 9B 逐题评分（30题） |
| `harness_report_r32.json` | LoRA r=32 逐题评分 |
| `harness_report_r64.json` | LoRA r=64 逐题评分 |
| `harness_report_r128.json` | LoRA r=128 逐题评分 |

### 关键结论

| 模型 | 综合评分 | 可解率 |
|------|:------:|:-----:|
| **基座 9B** | **90.3** | 100% |
| LoRA r=32 | 87.3 | 93% |
| LoRA r=64 | 83.7 | 83% |
| LoRA r=128 | 87.3 | 93% |

> **结论**：基座 9B 碾压所有 LoRA 变体。数据量（2986题）不足以支撑有效微调，生产级需万级数据回流。

### 运行

```bash
python lora_ablation/compare_loras.py    # 跑 4 组对比
python lora_ablation/harness.py          # 单组评估
```

---

## 二、知识蒸馏 `distillation/`

**目的**：9B 教师 → 4B 学生，探索小模型替代方案。

| 文件 | 说明 |
|------|------|
| `README.md` | 蒸馏效果报告（原 `distill_comparison.md`） |
| `distill_data_gen.py` | 蒸馏数据生成（9B 教师出题 → Alpaca 格式） |
| `distill_gen.py` | 蒸馏题目批量生成 |
| `eval_distill.py` | 蒸馏效果评估 |
| `clean_distill.py` | 蒸馏数据清洗 |
| `star_tutor_distill.yaml` | LLaMAFactory 训练配置 |
| `harness_report_student.json` | Student 4B 逐题评分 |

### 关键结论

| 指标 | Teacher 9B | Student 4B | 差距 |
|------|:------:|:------:|:--:|
| 综合评分 | 90.3 | 86.83 | -3.5 |
| 可解率 | 90% | **93%** | ✅ +3 |
| 知识点匹配度 | 4.80 | 4.73 | -0.07 |
| 难度校准 | 4.50 | 4.47 | -0.03 |

> **训练配置**：100 题蒸馏数据（6 知识点 × 3 难度 × 2 题型 × 3 变式），Alpaca 格式，QLoRA（rank=32, α=64, 4-bit），AutoDL vGPU-48GB（RTX 4090等效）。  
> **结论**：100 题蒸馏数据下，Student 可解率反超 Teacher（93% > 90%），基本实现蒸馏效果。同家族 tokenizer（Qwen3.5 系列）无分布偏差。

### 运行

```bash
python distillation/distill_data_gen.py   # 生成蒸馏数据（需 9B GPU）
python distillation/eval_distill.py       # 评估蒸馏效果
```

---

## 三、Agent 行为门禁 `agent_gate/`

**目的**：验证 Teacher Agent 的 5 个行为约束（铁律测试）。

| 文件 | 说明 |
|------|------|
| `l2_agent_harness.py` | Agent 行为门禁测试 |
| `l2_agent_harness.json` | 5 场景测试用例 |
| `l2_agent_harness_report.md` | 测试报告 |

### 5 个测试场景

1. 学生答对 → 确认，不质疑
2. 学生答错 → 引导，不直接给答案
3. 学生请求直接答案 → 拒绝，坚持引导
4. 学生完全不会 → 从基础概念引入
5. 学生中途放弃 → 鼓励 + 降低难度

### 运行

```bash
python agent_gate/l2_agent_harness.py
# → 输出 data/l2_agent_harness_report.md
```

---

## 四、IRT 知识追踪 `irt/`

**目的**：验证三参数 IRT 模型（θ 能力值）的数学逻辑正确性。

| 文件 | 说明 |
|------|------|
| `irt_harness.py` | IRT 评估脚本 |
| `irt_harness.json` | 6 个测试场景（优等生/中等生/差生等） |
| `irt_harness_report.md` | 测试报告 |

### 6 个测试场景

| 场景 | 验证点 |
|------|--------|
| 优等生全部答对 | θ 单调上升，难度递增 |
| 差等生全部答错 | θ 单调下降，难度递减 |
| 中等生波动 | θ 正常波动 |
| 冷启动 | θ=0.5 初始化 |
| 极端值 | θ 不爆边界 |
| 变式题 | 难度在 θ 附近 |

### 运行

```bash
python irt/irt_harness.py
```

---

## 📊 实验总览

| 实验 | 结论 | 决策 |
|------|------|------|
| L1 出题 Harness | 基座 9B (90.3) > 所有 LoRA | 出题用基座，不用 LoRA |
| L2 Agent 门禁 | Prompt 修复后 5/5 全通 | Teacher 铁律生效 |
| 蒸馏 9B→4B | Student 86.83 vs Teacher 90.3 | 可解率反超，方向可行 |
| IRT Harness | 6/6 数学逻辑正确 | IRT 管线就绪 |

---

## ⚠️ 已知限制

- 蒸馏数据仅 100 题，产品化需扩大规模
- LoRA 消融训练数据 2986 题，远低于模型参数量（9B），产品化需扩大规模至少10万级
- IRT 仅在 Harness 中验证，未在真实学生数据上测试

---

> 所有实验在 AutoDL vGPU-48GB (RTX 4090) 上完成。Harness 裁判模型为 DeepSeek V4。
