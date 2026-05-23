"""从 ModelScope 下载星学伴模型权重"""
import os
from modelscope import snapshot_download

MODEL_REPO = "prlove/star_tutor_lora"
TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def main():
    print(f"📥 从 ModelScope 下载模型: {MODEL_REPO}")
    print(f"📁 目标目录: {TARGET_DIR}")

    # 下载到 models/star_tutor_lora/
    local_dir = snapshot_download(MODEL_REPO, cache_dir=TARGET_DIR)

    print(f"✅ 下载完成: {local_dir}")
    print("\n模型包含:")
    for name in sorted(os.listdir(local_dir)):
        full = os.path.join(local_dir, name)
        if os.path.isdir(full):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, filenames in os.walk(full)
                for f in filenames
            )
            print(f"  📦 {name} ({size / 1e9:.2f} GB)")

    print("\n💡 基座模型 Qwen3.5-4B 需单独下载（HuggingFace / ModelScope）")

if __name__ == "__main__":
    main()
