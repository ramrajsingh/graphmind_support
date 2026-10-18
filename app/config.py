import os

from dotenv import load_dotenv

load_dotenv()

FALKORDB_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.environ.get("FALKORDB_PORT") or 6379)
FALKORDB_USERNAME = os.environ.get("FALKORDB_USERNAME") or None
FALKORDB_PASSWORD = os.environ.get("FALKORDB_PASSWORD") or None
FALKORDB_GRAPH = os.environ.get("FALKORDB_GRAPH", "support_memory")

# Embeddings for semantic search over known issues (FalkorDB vector index). Uses OPENAI_API_KEY;
# without it, search falls back to FalkorDB's full-text index.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_STT_MODEL = os.environ.get("SARVAM_STT_MODEL", "saaras:v4")
SARVAM_TTS_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v4-flash")
SARVAM_TTS_SPEAKER = os.environ.get("SARVAM_TTS_SPEAKER", "priya")

# Which LLM runs the agent: "anthropic" or "openai".
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "low")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")

# Languages Bulbul can speak. Saaras understands more; anything outside this
# set gets an on-screen reply in its own language and speech in English.
TTS_LANGUAGES = {
    "en-IN", "hi-IN", "bn-IN", "ta-IN", "te-IN", "kn-IN",
    "ml-IN", "mr-IN", "gu-IN", "pa-IN", "od-IN",
}
