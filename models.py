import torch
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from pinecone import Pinecone
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema

from config import (
    DEVICE,
    EMBED_MODEL_PATH,
    EMBED_COMMIT_HASH,
    RERANK_MODEL_PATH,
    RERANK_COMMIT_HASH,
    GENERATION_MODEL_ID,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    REDIS_URL,
    REDIS_DBNAME
)

# ============================================================================
# Initialize External Services
# ============================================================================

print("[INFO] Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
pinecone_index = pc.Index(PINECONE_INDEX_NAME) if pc else None

print("[INFO] Initializing RedisVL SearchIndex...")
try:
    schema_dict = {
        "index": {
            "name": REDIS_DBNAME,
            "prefix": "cache"
        },
        "fields": [
            {"name": "prompt", "type": "text"},
            {"name": "response", "type": "text"},
            {"name": "vector", "type": "vector", "attrs": {"dims": 2048, "distance_metric": "cosine", "algorithm": "flat", "datatype": "float32"}}
        ]
    }
    schema = IndexSchema.from_dict(schema_dict)
    redis_cache = SearchIndex(schema, redis_url=REDIS_URL)
    redis_cache.create(overwrite=True)
except Exception as e:
    print(f"[WARNING] Could not initialize RedisVL SearchIndex: {e}")
    redis_cache = None

# ============================================================================
# Load Dataset
# ============================================================================

print("[INFO] Loading dataset...")
dataset = load_dataset(path="mrdbourke/recipe-synthetic-images-10k")
print(f"[INFO] Dataset loaded with {len(dataset['train'])} samples")

# ============================================================================
# Load Models
# ============================================================================

modality_to_tokens = {
    "image": 2048,
    "image_text": 10240,
    "text": 8192
}

print(f"[INFO] Loading embedding model from: {EMBED_MODEL_PATH} with commit: {EMBED_COMMIT_HASH}")
embed_model = AutoModel.from_pretrained(
    EMBED_MODEL_PATH,
    revision=EMBED_COMMIT_HASH,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    device_map="auto",
).eval()

embed_modality = "image_text"
embed_processor_kwargs = {
    "max_input_tiles": 6,
    "use_thumbnail": True,
    "p_max_length": modality_to_tokens[embed_modality]
}

embed_processor = AutoProcessor.from_pretrained(
    EMBED_MODEL_PATH,
    revision=EMBED_COMMIT_HASH,
    trust_remote_code=True,
    **embed_processor_kwargs
)
print(f"[INFO] Embedding model loaded!")

print(f"[INFO] Loading rerank model from: {RERANK_MODEL_PATH} with commit: {RERANK_COMMIT_HASH}")
rerank_model = AutoModelForSequenceClassification.from_pretrained(
    RERANK_MODEL_PATH,
    revision=RERANK_COMMIT_HASH,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    device_map="auto",
).eval()

rerank_modality = "image_text"
rereank_processor_kwargs = {
    "max_input_tiles": 6,
    "use_thumbnail": True,
    "rerank_max_length": modality_to_tokens[rerank_modality]
}

rerank_processor = AutoProcessor.from_pretrained(
    RERANK_MODEL_PATH,
    revision=RERANK_COMMIT_HASH,
    trust_remote_code=True,
    **rereank_processor_kwargs
)
print(f"[INFO] Rerank model loaded!")

print("[INFO] Loading generation model...")
qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    GENERATION_MODEL_ID,
    dtype="auto",
    device_map="auto"
)
qwen_processor = AutoProcessor.from_pretrained(GENERATION_MODEL_ID)
print(f"[INFO] Generation model loaded")
