"""
星学伴 · 统一配置
从 .env 文件加载，无外部依赖
"""
import os

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _load_env() -> dict:
    """加载 .env 文件，返回 key→value 字典"""
    env = {}
    if not os.path.exists(_ENV_PATH):
        return env
    with open(_ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            env[key] = value
    return env


_env = _load_env()


def get(key: str, default: str = "") -> str:
    """读取配置项，优先环境变量其次 .env 文件"""
    return os.environ.get(key) or _env.get(key) or default


# ============================================
#  导出
# ============================================
DEEPSEEK_API_KEY = get("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

VISION_API_KEY = get("VISION_API_KEY", DEEPSEEK_API_KEY)
VISION_API_BASE = get("VISION_API_BASE", DEEPSEEK_API_BASE)
VISION_MODEL_NAME = get("VISION_MODEL_NAME", "qwen-vl-max")

NEO4J_URI = get("NEO4J_URI")
NEO4J_USER = get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get("NEO4J_PASSWORD")

MODEL_PATH = get("MODEL_PATH", "models/Qwen3.5-4B")

# 出题模式：local（本地GPU推理）或 api（走DeepSeek API，无GPU可用）
QUESTION_PROVIDER = get("QUESTION_PROVIDER", "local")
