# 星学伴 Docker 镜像
# 构建: docker build -t star-tutor .
# 需要 CUDA + LaTeX，镜像较大 (~12GB)

FROM nvidia/cuda:12.6.3-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# === 系统依赖 ===
# 换国内 apt 源，加速下载
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null; \
    sed -i 's|http://ports.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    texlive-base texlive-latex-base texlive-pictures \
    texlive-latex-recommended texlive-latex-extra \
    texlive-lang-chinese \
    texlive-fonts-recommended texlive-extra-utils \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# === 修复 ImageMagick PDF 安全策略 ===
RUN sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml

# === Python 依赖 ===
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# === 项目文件 ===
COPY . .

# === 入口 ===
EXPOSE 8000
CMD ["python3", "app.py"]
