# 🐧 星学伴 Star Tutor

> 5-Agent 个性化初中数学学习系统 + Harness 四级防线 — 拍照即学，AI 引导式教学，全自动安全管控

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![ModelScope](https://img.shields.io/badge/ModelScope-模型权重-blue)](https://modelscope.cn/models/prlove/star_tutor_lora)

学生拍照上传数学题 → Vision 识别 → Teacher **先自己解一遍**（L3 多路径求解+自疑纠错）→ 苏格拉底式提问引导 → 诊断知识薄弱点 → 自动出变式题 + 配图。**不直接给答案，教学生自己推导。** 全程 L4 行为管控护航，防滥用防学习异常。

> 💡 **想了解每个技术决策的来龙去脉？** 查看 [`CHANGELOG.md`](CHANGELOG.md)——按「问题 → 根因 → 方案」组织，覆盖 MCP 选型、Agent 迭代、Harness 四级防线、微调消融、Docker 部署等 12 大技术板块，展示真实工程迭代中的踩坑与权衡。

---

## 🏗 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI :8000                              │
│                 (会话管理 + 图片上传 + 前端服务)                     │
└──────┬──────┬──────┬──────┬──────┬──────────────────────────────┘
       │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼
┌──────────┐┌──────┐┌──────┐┌──────┐┌──────┐
│ Question ││Diag- ││Evalu-││Vision││Figure│
│   MCP    ││nosis ││ation ││ MCP  ││ MCP  │
│  :8765   ││:8766 ││:8767 ││:8768 ││:8769 │
│  出题    ││ 诊断 ││ 评估 ││ 识别 ││ 配图 │
│ Qwen4B   ││Neo4j ││Deep- ││Qwen- ││TikZ/ │
│ 本地推理  ││  KG  ││Seek  ││ VL   ││Matpl.│
└──────────┘└──────┘└──────┘└──────┘└──────┘
       │                                          
       └─────── Teacher Agent ──────────────────┘
         ┌─────────────────────────────────┐
         │  🧮 L3 多路径求解 + 自疑纠错     │
         │  🛡️ L4 内容分类 + 行为管控      │
         │  🚧 stuck 检测 + 🔓 提示模式     │
         │  📏 上下文剪枝 + 锚定            │
         │  DeepSeek V4 · 苏格拉底引导      │
         └─────────────────────────────────┘
```

**5 个 MCP Server** + **1 个 Teacher Agent（含 Harness L3/L4 防线）**：

| 组件 | 类型 | 端口 | 技术栈 | 职责 |
|------|:--:|:---:|--------|------|
| Question | MCP Server | 8765 | Qwen3.5-4B 4-bit 本地 / DeepSeek API | 根据知识点+难度出变式题 |
| Diagnosis | MCP Server | 8766 | Neo4j AuraDB 知识图谱 | 追溯知识薄弱点的前置依赖 |
| Evaluation | MCP Server | 8767 | DeepSeek V4 | 三维度评估（思路/结果/表述） |
| Vision | MCP Server | 8768 | Qwen-VL-Plus (百炼) | 识别拍照题目 / 手写公式 |
| Figure | MCP Server | 8769 | TikZ (LaTeX) + Matplotlib | 几何图/函数图/统计图配图 |
| **solve_problem** | **Teacher 内部** | — | DeepSeek V4 × 2 + Qwen 仲裁 | L3 多路径求解：双路径 → 分叉则重试 → 仍分歧则 Qwen 仲裁 |
| **L4 行为管控** | **Teacher 内部** | — | 内容分类 + 会话锁定 + 日锁禁言 | 非学习内容拦截、重复错误检测、自动锁定推送人工 |

---

## 🔄 教学流水线

```
学生发题（文字/图片）
  │
  ├─ 📷 图片？→ Vision MCP 识别题目文本
  │
  ├─ 🧮 solve_problem（Teacher 后台异步求解，不等结果）
  │     ├─ L3 多路径：DeepSeek 双 Prompt 并行 → 不一致则重试 → Qwen 仲裁
  │     └─ 答案就绪后注入上下文，Teacher 基于正确结果引导
  │
  ├─ 🤔 苏格拉底提问引导（反问 / 拆解 / 类比）
  │     ├─ 学生回答 → ⚙️ L3 程序化评估（代码层直接调 evaluate_answer，不等 LLM）
  │     ├─ 同一错误 ×2 → 🔥 L3-1 自疑：后台异步重算验证
  │     ├─ 同一错误 ×3 → 🚨 L3→L4 上报：会话锁定 + 标记异常
  │     ├─ 连续 3 次卡住 → 🔓 提示模式（给关键线索）
  │     └─ 连续 2 次未掌握 → 🔍 trace_prerequisites 诊断
  │
  ├─ ✅ 确认掌握 → 📝 generate_question 出变式题
  │     └─ 几何/函数题 → 📐 generate_figure TikZ 配图
  │
  ├─ 🛡️ L4 前置：非学习内容 1 次提醒 → 2 次锁定 → 日锁 ≥3 全天禁言
  │
  └─ 📈 IRT 知识追踪 → 动态调整出题难度
```

---

## ⚡ 核心特性

| 特性 | 说明 |
|------|------|
| **先解题再引导** | Teacher 收到题目后后台异步求解（L3 多路径），不等结果先跟学生互动；答案就绪后基于正确结果引导 |
| **苏格拉底提问** | 用反问 / 拆解 / 类比引导学生自己思考，**不直接给答案** |
| **🔓 提示模式** | 学生连续 3 次无法回答时自动切换，给出关键线索帮助突破 |
| **三维度评估** | 思路正确性 + 结果正确性 + 表述完整性 |
| **知识图谱诊断** | Neo4j 存储 544 节点 / 265 知识点，追溯薄弱根因 |
| **自适应难度** | IRT 知识追踪根据历史表现动态调整：3 次答对即进入困难区 |
| **TikZ 配图** | 几何/函数题自动生成高精度 LaTeX TikZ 矢量配图 |
| **上下文管理** | 长对话自动剪枝 + 锚定，防止模型遗忘教学规则 |
| **数学等价识别** | 评估时识别 `2*根号13` = `2倍根号13` = `2√13` 等表达式变体 |
| **🛡️ L3 解答正确性** | 多路径求解 + SymPy/LLM 代入验证 + 自疑纠错，杜绝"教师自己算错" |
| **🛡️ L4 行为管控** | 内容分类 + 1 次提醒 → 2 次会话锁定 + 日锁 ≥3 次全天禁言，防滥用防学习异常 |

---

## 📂 目录结构

```
Star_Tutor/
├── agents/                    ← 5 个 MCP Server + Teacher Agent + KG 客户端 + IRT
│   ├── teacher_agent.py       ← 核心：苏格拉底引导 + L3 多路径求解 + L4 行为管控 (~1200行)
│   ├── mcp_question_server.py ← 出题 (默认本地 4B，可切换 API)
│   ├── mcp_diagnosis_server.py← 知识图谱诊断
│   ├── mcp_evaluation_server.py← 学生回答评估（含数学表达式等价判定）
│   ├── mcp_vision_server.py   ← 拍照识别 (题目+手写，双场景分流)
│   ├── mcp_figure_server.py   ← TikZ/Matplotlib 配图
│   ├── kg_client.py           ← Neo4j 知识图谱客户端
│   └── irt.py                 ← IRT 三参数知识追踪
│
├── docs/                      ← 项目文档 📖
│   ├── examples.md            ← 5 个完整对话案例（含 L3/L4 防线实战）
│   ├── DEPLOY.md              ← 详细部署教程
│   ├── 案例1~5_源文件.json     ← 案例原始 session 数据
│   └── 截图 (ans01~04.jpg, ex01.png, qs01~03.png)
│
├── evaluation/                ← 实验文档与数据 📊
│   ├── README.md              ← 实验总览
│   ├── lora_ablation/         ← LoRA 消融实验 (r32/64/128)
│   ├── distillation/          ← 知识蒸馏 (9B→4B + YAML配置)
│   ├── agent_gate/            ← Agent 行为门禁测试
│   └── irt/                   ← IRT 评估
│
├── scripts/                   ← 开发辅助脚本 🔧
│   ├── run.py                 ← 本地一键启动 + 测试
│   └── health_check.py        ← MCP Server 就绪检测
│
├── models/                    ← 模型权重（从 ModelScope 下载，不入库）
│   │                            🔗 modelscope.cn/models/prlove/star_tutor_lora
│   ├── Qwen3.5-4B/            ← 基座模型 (4-bit 量化，默认本地推理)
│   ├── star_tutor_lora/       ← LoRA 权重 (r32/r64/r128 + 蒸馏 4B)
│   │   ├── r32/
│   │   ├── r64/
│   │   ├── r128/
│   │   └── distill_4b/
│
├── data/                      ← 运行时数据
│   ├── kg_math.json           ← 知识图谱 (544节点, 265知识点, 177边)
│   ├── figures/               ← 生成的配图 PNG（不提交 Git）
│   └── sessions/              ← 对话历史 (JSON, 按 session 存储)
│
├── static/                    ← 前端 (原生 HTML/JS + KaTeX 本地渲染)
│   └── lib/                   ← KaTeX 本地依赖
│
├── tests/                     ← 测试脚本 🧪
│   ├── test_harness_l3.py     ← L3 解答门禁测试
│   ├── test_harness_l4.py     ← L4 行为管控测试
│   └── test_pipeline.py       ← 端到端全链路测试
│
├── CHANGELOG.md               ← 问题 → 根因 → 方案完整记录
├── app.py                     ← FastAPI 主入口
├── servers.py                 ← MCP Server 一键启动
├── config.py                  ← 统一配置加载 (from .env)
├── .env.template              ← API Key 模板
├── requirements.txt           ← Python 依赖
│
├── Dockerfile                 ← Docker 镜像 (含 TeX Live)
├── docker-compose.yml         ← GPU 版一键部署
├── docker-compose.cpu.yml     ← 无 GPU 版一键部署
├── .dockerignore              ← 构建排除
└── DEPLOY.md                  ← 详细部署教程
```

---

## 🚀 快速开始

### 0. 下载模型权重

```bash
# 从 ModelScope 下载（基座模型 + 4 组 LoRA 权重）
pip install modelscope
python scripts/download_models.py
```

模型说明见 [ModelScope 仓库](https://modelscope.cn/models/prlove/star_tutor_lora)。

### Docker 部署（推荐）

```bash
git clone https://github.com/prlove0414/Star_Tutor.git
cd Star_Tutor
cp .env.template .env        # 填 API Key
docker compose up -d
# → http://localhost:8000

# 无 GPU 版
docker compose -f docker-compose.cpu.yml up -d
```

详细教程见 [`docs/DEPLOY.md`](docs/DEPLOY.md)。

### 本地开发

```bash
# 1. 环境
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. 配置
cp .env.template .env         # 填 API Key
# 默认使用本地 Qwen3.5-4B 出题（需 GPU ≥4GB 显存）
# 无 GPU：在 .env 中设 QUESTION_PROVIDER=api 走 DeepSeek API

# 3. 启动
python app.py
# → http://localhost:8000
```

> ⚠️ **默认使用本地 4B 模型出题**（`QUESTION_PROVIDER=local`），需 NVIDIA GPU (≥4GB 显存)。
> 若 GPU 不可用，在 `.env` 中设 `QUESTION_PROVIDER=api` 自动切换为 DeepSeek API 出题。
> Teacher、Evaluation、Vision、Figure 始终走 API。

---

## 🎓 使用示例

完整对话案例见 [`docs/examples.md`](docs/examples.md)，包含 5 个端到端案例：

| 案例 | 内容 | 消息数 | 亮点 |
|------|------|:-----:|------|
| 案例 1 | 代数应用题（百钱百鸡变式） | 34 | 全程苏格拉底引导，零直接给答案 |
| 案例 2 | 几何题全流程 | 32 | 📷 拍照 → 🔍 识别 → 🧮 求解 → 📐 TikZ 配图 |
| 案例 3 | 手写解题过程识别 ✍️ | 29 | Vision 双场景分流，4 步手写依次评估 |
| 案例 4 | L3 自疑 → L4 锁定 🔒 | 16 | 同错 3 次自动触发 Harness 四级防线 |
| 案例 5 | 非学习内容拦截 🛡️ | 4 | 1 次提醒 → 2 次锁定，话语统一不泄露 |

---

## 📊 实验数据

### 保留的原始数据

| 位置 | 内容 | 格式 |
|------|------|------|
| `evaluation/lora_ablation/` | LoRA 消融 30 题逐题评分 | JSON |
| `evaluation/distillation/` | 蒸馏训练配置 + Student 30 题评分 | YAML / JSON |
| `evaluation/agent_gate/` | Agent 门禁 5 场景用例 | JSON |
| `evaluation/irt/` | IRT 6 场景测试数据 | JSON |

> 模型权重及训练日志（trainer_state.json 等）不在此仓库，见 [ModelScope](https://modelscope.cn/models/prlove/star_tutor_lora)。

### 知识图谱数据

| 文件 | 内容 |
|------|------|
| `data/kg_math.json` | 初中数学知识图谱 (544 节点, 265 知识点, 177 前置依赖边) |

图谱结构：3 大领域（数与代数 / 图形与几何 / 统计与概率）→ 29 章 → 265 个知识点。
运行时通过 `agents/kg_client.py` 连接 Neo4j AuraDB 查询。

### 训练数据说明

- **LoRA 微调**：2,986 题，AutoDL + LLaMAFactory 训练，Alpaca 格式
- **知识蒸馏**：100 题（9B Teacher 生成），QLoRA 训练 4B Student

---

## 🔑 关键实验结论

| 实验 | 结论 |
|------|------|
| LoRA 消融 | 基座 9B (90.3 分) > 所有 LoRA 变体；数据量不足（~3k）是主因 |
| 蒸馏 9B→4B | Student 可解率 93% 反超 Teacher 90%，蒸馏方向可行 |
| L2 Agent 门禁 | Prompt 强化后 5/5 全通，Teacher 铁律生效 |
| IRT Harness | 6/6 数学逻辑正确，θ 驱动难度策略就绪 |
| **L3 解答正确性** | **多路径求解 + 自疑纠错通过端到端验证，程序化评估 100% 不遗漏** |
| **L4 行为管控** | **非学习拦截 + 重复错误锁定 + 日锁禁言三级防线全部跑通** |

详见 [`evaluation/README.md`](evaluation/README.md) 和 [`docs/examples.md`](docs/examples.md)。

---

## 🛠 技术栈

| 层 | 技术 |
|----|------|
| Agent 编排 | Teacher Agent（中心调度） + MCP 协议 |
| 推理后端 | DeepSeek V4 (API，Teacher/评估/求解/配图) + Qwen3.5-4B (本地 4-bit，出题) |
| 视觉识别 | Qwen-VL-Plus (百炼 DashScope) |
| 知识图谱 | Neo4j AuraDB (在线) + 本地 JSON 副本 |
| 配图渲染 | TikZ/pgf (LaTeX) + pdfcrop + ImageMagick |
| 前端公式 | KaTeX 本地渲染 |
| 知识追踪 | IRT 三参数 Logistic 模型 |
| L3 解答门禁 | 多路径求解 (DeepSeek 双 Prompt + Qwen 仲裁) + SymPy/LLM 代入验证 |
| L4 行为管控 | 内容分类 + 程序化评估 + 会话锁定 + 日锁禁言 |
| 前端 | 原生 HTML/JS，支持图片+文字同时发送 |
| 部署 | Docker + docker-compose (GPU/CPU 双方案) |
| 微调框架 | LLaMAFactory + QLoRA + BitsAndBytes |
| 评估 | lm-evaluation-harness + 自建裁判模型 |

---

## 📄 License

MIT
