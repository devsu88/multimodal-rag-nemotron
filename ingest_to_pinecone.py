import os
from dotenv import load_dotenv
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from pinecone import Pinecone
import math

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in .env")

print("[INFO] Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

print("[INFO] Loading dataset...")
dataset = load_dataset(path="mrdbourke/recipe-synthetic-images-10k")
train_dataset = dataset["train"]
print(f"[INFO] Dataset loaded with {len(train_dataset)} samples")

print("[INFO] Loading embeddings...")
image_text_embeddings_file = load_file("image_text_embeddings_10k.safetensors")
image_text_embeddings = image_text_embeddings_file["image_text_embeddings"]
print(f"[INFO] Embeddings loaded: {image_text_embeddings.shape}")

BATCH_SIZE = 100
total_samples = len(train_dataset)
num_batches = math.ceil(total_samples / BATCH_SIZE)

print("[INFO] Starting upsert to Pinecone...")
for i in range(num_batches):
    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, total_samples)
    
    batch_vectors = []
    for j in range(start_idx, end_idx):
        embedding = image_text_embeddings[j].tolist()
        sample = train_dataset[j]
        
        # In Pinecone, ID must be a string
        vector_id = str(j)
        
        # Metadata can be extracted from sample, skipping the 'image' as it's a PIL Image object
        # We store the ID to be able to fetch the image from HF dataset later if needed
        # We can also store text/markdown to avoid loading HF dataset for text only queries
        metadata = {
            "dataset_index": j,
        }
        
        batch_vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })
        
    index.upsert(vectors=batch_vectors)
    if (i + 1) % 10 == 0:
        print(f"[INFO] Upserted batch {i + 1}/{num_batches}")

print("[INFO] Upsert to Pinecone completed successfully!")
