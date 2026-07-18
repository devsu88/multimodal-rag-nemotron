import os
import torch
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_MODEL_PATH = "nvidia/llama-nemotron-embed-vl-1b-v2"
EMBED_COMMIT_HASH = "5b5ca69c35bf6ec1484d2d5ff238626e67a745e2"

RERANK_MODEL_PATH = "nvidia/llama-nemotron-rerank-vl-1b-v2"
RERANK_COMMIT_HASH = "47e5a355d1a050c3e5f69d53f14964b1d34bcd9d"

GENERATION_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
REDIS_URL = os.getenv("REDIS_URL")
REDIS_DBNAME = os.getenv("REDIS_DBNAME")
