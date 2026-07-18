---
title: Multimodal Rag Nemotron
emoji: ⚡
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
python_version: '3.11'
app_file: app.py
pinned: false
---

# 👁️📑 Multimodal RAG with Nemotron & LangGraph

This repository contains a full-fledged **Multimodal Retrieval-Augmented Generation (RAG)** application built with state-of-the-art vision-language models, vector databases, and semantic caching. 

The application allows users to search for recipes (both through text and image queries), retrieve relevant recipe documents, rerank them for precision, and optionally generate an AI-powered summary—all orchestrated through **LangGraph** and served via a **Gradio** interface.

---

## ✨ Features

- **Multimodal Retrieval**: Search recipes using either text queries or by uploading an image.
- **Advanced Embeddings & Reranking**: Utilizes NVIDIA's Llama-Nemotron Vision-Language models for accurate, multimodal document matching and reranking.
- **Semantic Caching**: Integrates **RedisVL** to instantly return cached results for semantically similar queries, saving time and compute resources. The vector distance threshold is dynamically adjustable via the UI.
- **Agentic Workflow**: The entire pipeline (Cache Check -> Pinecone Retrieval -> Reranking -> Generation -> Cache Update) is orchestrated as a state machine using **LangGraph**.
- **Parametric UI**: Fully interactive Gradio interface with adjustable sliders for Retrieval Top-K and Cache Thresholds, alongside optional Rerank and Summary generation toggles.

---

## 🛠️ Architecture & Tech Stack

### Models
- **Embedding**: [`nvidia/llama-nemotron-embed-vl-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2)
- **Reranking**: [`nvidia/llama-nemotron-rerank-vl-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2)
- **Generation**: [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

### Infrastructure
- **Vector Store**: [Pinecone](https://www.pinecone.io/) (for storing and querying recipe vectors)
- **Semantic Cache**: [Redis](https://redis.io/) / RedisVL (for caching responses to avoid redundant compute)
- **Orchestration**: [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/)
- **UI**: [Gradio](https://www.gradio.app/)
- **Dataset**: `mrdbourke/recipe-synthetic-images-10k` (HuggingFace)

---

## 🚀 Getting Started

### Prerequisites

You need a machine with a CUDA-compatible GPU (VRAM requirement ~16GB+ depending on the models used). 
Ensure you have Python 3.10+ installed.

### 1. Clone the repository

```bash
git clone https://github.com/devsu88/multimodal-rag-nemotron.git
cd multimodal-rag-nemotron
```

### 2. Install Dependencies

You can install the required packages using pip (a `requirements.txt` should be generated, but here are the core dependencies):

```bash
pip install torch transformers datasets pinecone-client redisvl python-dotenv gradio langgraph pillow numpy
```

### 3. Environment Variables

Create a `.env` file in the root directory of the project and populate it with your API keys and connection strings:

```env
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_INDEX_NAME="your_pinecone_index_name"
REDIS_URL="redis://your_redis_host:port"
REDIS_DBNAME="your_redis_index_name"
```

### 4. Populate Pinecone (One-time setup)

Before running the application for the first time, you need to embed and upload the dataset to your Pinecone index running:

```bash
python ingest_to_pinecone.py
```
*(Note: Make sure your Pinecone index is created with 2048 dimensions and the `cosine` metric before running the population script).*

### 5. Run the Application

Start the Gradio server by running:

```bash
python app.py
```

The server will automatically download the dataset and the HuggingFace models upon the first execution. Once loaded, a local URL (usually `http://127.0.0.1:7860`) will be provided in the terminal.

---

## 📂 Project Structure

- `config.py`: Environment loading and global configuration constants.
- `models.py`: Initialization of Pinecone, Redis, and loading of HuggingFace models & processors into GPU memory.
- `graph.py`: The LangGraph state definitions, nodes, and conditional edges that orchestrate the RAG flow.
- `utils.py`: Helper functions for LLM generation and HTML formatting.
- `app.py`: The presentation layer; contains the Gradio UI definitions and the main entry point.
- `ingest_to_pinecone.py`: The script to embed and upload the dataset to Pinecone.
