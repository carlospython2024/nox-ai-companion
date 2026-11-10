from pathlib import Path


ROOT = Path(__file__).resolve().parent
PERSONALITY_PATH = ROOT / "personality.txt"
DATABASE_PATH = ROOT / "data" / "nox.db"
DEFAULT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
CONTEXT_TOKENS = 4096
RESPONSE_TOKENS = 512
RECENT_TURNS = 8
OLDER_USER_TURNS = 8
