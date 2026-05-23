# 🐧 星学伴 · Docker 部署教程

> 本文档教你如何在任意一台电脑上一键部署星学伴。

---

## 一、硬件要求

| 硬件 | 最低要求 | 推荐 |
|------|---------|------|
| 显卡 | NVIDIA GTX 1060 6GB 或更高 | RTX 3060 / RTX 4060 |
| 显存 | ≥ 4 GB（已量化，只跑 4B 模型） | ≥ 6 GB |
| 内存 | ≥ 8 GB | ≥ 16 GB |
| 磁盘 | ≥ 20 GB 空闲（镜像 ~12GB + 模型 ~3GB） | SSD |

如果你的显卡不在上表中，只要满足 **NVIDIA 显卡 + 显存 ≥ 4GB + 驱动 ≥ 525** 就能跑。

> ⚠️ AMD / Intel 显卡暂不支持（本地推理依赖 CUDA）。未来可能出 CPU 版。

---

## 二、环境准备

### 🪟 Windows 用户

#### ① 安装 WSL2

打开 **PowerShell（管理员）**，执行：

```powershell
wsl --install
```

重启电脑。重启后会弹出 Ubuntu 终端窗口，设置用户名和密码，完成。

> 如果之前装过 WSL1，执行 `wsl --set-default-version 2` 切换。

#### ② 安装 Docker Desktop

1. 下载 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. 安装，一路下一步
3. 启动 Docker Desktop
4. 打开 **Settings → Resources → WSL Integration**
5. ✅ 勾选你的 Ubuntu 发行版
6. 点击 **Apply & Restart**

#### ③ 验证

打开 WSL（Ubuntu）终端，输入：

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

如果看到显卡信息表格 → ✅ 成功！

---

### 🐧 Linux 用户

#### ① 安装 Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
# 退出终端重新登录，或执行 newgrp docker
```

#### ② 安装 NVIDIA Container Toolkit

```bash
# Ubuntu / Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### ③ 验证

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

看到显卡信息 → ✅。

---

## 三、部署星学伴

### 1. 下载项目

```bash
git clone https://github.com/你的用户名/Star_Tutor.git
cd Star_Tutor
```

### 2. 配置 API Key

```bash
cp .env.template .env
```

用文本编辑器打开 `.env`，填入你的 API Key：

| 变量 | 去哪申请 | 用途 |
|------|---------|------|
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) | 教师 Agent 推理 |
| `VISION_API_KEY` | [阿里云百炼](https://bailian.console.aliyun.com) | 拍照识别题目 |
| `NEO4J_URI/PASSWORD` | [Neo4j AuraDB](https://neo4j.com/cloud/aura/) | 知识图谱（可选） |

`MODEL_PATH` 不用改，Docker 环境会自动设为 `/app/models/Qwen3.5-4B`。

### 3. 下载模型权重

创建 `models/` 目录并下载基座模型（Qwen3.5-4B）：

```bash
mkdir -p models
cd models

# 方法一：ModelScope（国内推荐，速度快）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', local_dir='Qwen3.5-4B')"

# 方法二：HuggingFace
# pip install huggingface_hub
# huggingface-cli download Qwen/Qwen3.5-4B --local-dir Qwen3.5-4B

cd ..
```

> 模型约 2.5GB（4-bit 量化），下载需要几分钟。

最终的 `models/` 目录结构应包含：

```
models/
└── Qwen3.5-4B/          # 基座模型（必须）
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

### 4. 一键启动 🚀

```bash
docker compose up -d
```

第一次会构建镜像（下载 CUDA 基础镜像 + 安装 LaTeX + Python 依赖），约需 **5-15 分钟**（网速决定）。之后再次启动只需几秒。

启动完成后：

```bash
# 查看日志，确认没有报错
docker compose logs -f
```

看到 `✅ 星学伴 API 就绪` 就说明成功了。

### 5. 访问

打开浏览器 → **http://localhost:8000**

---

## 四、验证功能

1. 📷 点击左下角相机图标，上传一道数学题照片
2. 等待 Vision 识别（约 5-10 秒）
3. 观察 Teacher 是否用引导式提问（不是说"解题思路是…"）
4. 对话 2-3 轮，看看是否会自动出变式题 + 配图

---

## 五、常用命令

```bash
# 启动
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down

# 重新构建（更新代码后）
docker compose up -d --build

# 完全清除（删除容器+镜像+数据）
docker compose down -v
docker rmi star-tutor
```

---

## 六、常见问题

### Q1: `docker: command not found`

→ Docker Desktop 没装或没启动。Windows 用户检查托盘图标。

### Q2: `could not select device driver "nvidia"`

→ 显卡不支持或 NVIDIA Container Toolkit 没装。回到「环境准备」步骤验证。

### Q3: 镜像拉取太慢 / 超时

→ Docker Hub 在国内慢。配置国内镜像加速器：

Docker Desktop → Settings → Docker Engine，添加：

```json
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
```

点击 Apply & Restart。

### Q4: 模型加载后显存不足

→ 检查是否有其他程序占用显存（Chrome、游戏等）。关闭后重试。

### Q5: LaTeX 公式渲染失败

→ 首次启动会安装完整 TeX Live（~2GB），耐心等。日志里出现 `✅ MCP :8769 就绪` 即完成。

### Q6: 前端页面打不开 localhost:8000

→ 检查 `docker compose logs` 看是否有报错。常见原因：`.env` 没配好、模型没下载、端口被占用。

---

## 七、无 GPU 怎么办？（CPU / 核显用户）

星学伴提供 **API 模式**：出题功能改用 DeepSeek API 远程调用，其他 4 个 Agent 本身就走的 API，完全不需要显卡。

### 部署步骤

跟 GPU 版几乎一样，只需改一步：

```bash
git clone https://github.com/你的用户名/Star_Tutor.git
cd Star_Tutor
cp .env.template .env
# 编辑 .env，填 API Key
# ⚠️ 关键是改这一行：
#    QUESTION_PROVIDER=api
```

然后**不需要下载模型**，直接启动：

```bash
# ⚠️ 注意：用 docker-compose.cpu.yml 而不是默认的！
docker compose -f docker-compose.cpu.yml up -d
```

### 区别对比

| | GPU 版 | API 版（无 GPU） |
|------|:--:|:--:|
| 配置文件 | `docker-compose.yml` | `docker-compose.cpu.yml` |
| 出题速度 | ~2 秒（本地） | ~5 秒（API） |
| 需要显卡 | ✅ NVIDIA | ❌ 任意 |
| 需要下载模型 | ✅ ~2.5GB | ❌ |
| 出题费用 | 免费 | 走 DeepSeek API（极低） |
| 功能完整度 | 100% | 100% |

> 💡 API 模式下出题走 DeepSeek API，按 token 计费。一道题约 500 token，DeepSeek 百万 token 仅 ¥1，**出一万道题才 ¥5**，几乎等于免费。

---

> 🐧 有问题提 Issue，看到就回。
