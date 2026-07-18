import json
import time
import uuid
import torch
import numpy as np
from typing import TypedDict, Optional, Union
from PIL import Image
from langgraph.graph import StateGraph, START, END
from redisvl.query import VectorQuery

from config import DEVICE
from models import (
    embed_model, 
    rerank_model, 
    rerank_processor, 
    qwen_model, 
    qwen_processor,
    pinecone_index, 
    redis_cache, 
    dataset
)
from utils import generate_recipe_summary

# ============================================================================
# LangGraph State & Nodes
# ============================================================================

class GraphState(TypedDict):
    input_query: Union[str, Image.Image]
    rerank_option: str
    generate_summary_option: str
    cache_threshold: float
    top_k: int
    query_embedding: Optional[list[float]]
    retrieved_docs: Optional[list[dict]]
    reranked_docs: Optional[list[dict]]
    summary: Optional[str]
    cache_hit: bool
    timing_dict: dict

def check_cache_node(state: GraphState):
    print("[NODE] Entering check_cache_node...")
    start_time = time.time()
    query = state["input_query"]
    
    with torch.inference_mode():
        if isinstance(query, Image.Image):
            query_embeddings = embed_model.encode_documents(images=[query])
        else:
            query_embeddings = embed_model.encode_queries([query])
    
    query_embedding_list = query_embeddings[0].tolist()
    state["query_embedding"] = query_embedding_list
    state["cache_hit"] = False
    
    if redis_cache:
        try:
            query_str = query if isinstance(query, str) else "image_query"
            
            v_query = VectorQuery(
                vector=query_embedding_list,
                vector_field_name="vector",
                return_fields=["response", "vector_distance"],
                num_results=1,
                dialect=2
            )
            
            results = redis_cache.query(v_query)
            
            if results and float(results[0]["vector_distance"]) < state.get("cache_threshold", 0.15):
                state["cache_hit"] = True
                cached_data = json.loads(results[0]["response"])
                
                retrieved_docs = []
                for idx, score, rerank_string in zip(cached_data["retrieved_docs_indices"], cached_data.get("scores", []), cached_data.get("rerank_strings", [])):
                    sample = dataset["train"][idx]
                    retrieved_docs.append({
                        "dataset_index": idx,
                        "score": score,
                        "sample": sample,
                        "rerank_string": rerank_string
                    })
                
                state["retrieved_docs"] = retrieved_docs
                state["summary"] = cached_data.get("summary", "")
        except Exception as e:
            print(f"[WARNING] Redis cache check failed: {e}")
            
    state["timing_dict"] = state.get("timing_dict", {})
    state["timing_dict"]["cache_check_time"] = round(time.time() - start_time, 4)
    return state

def retrieve_pinecone_node(state: GraphState):
    print("[NODE] Entering retrieve_pinecone_node...")
    start_time = time.time()
    query_embedding = state["query_embedding"]
    
    if pinecone_index:
        res = pinecone_index.query(vector=query_embedding, top_k=state.get("top_k", 20), include_metadata=True)
        retrieved_docs = []
        for match in res["matches"]:
            idx = int(match["id"])
            score = match["score"]
            sample = dataset["train"][idx]
            retrieved_docs.append({
                "dataset_index": idx,
                "score": score,
                "sample": sample,
                "rerank_string": f"Score: {round(score, 4)}"
            })
        state["retrieved_docs"] = retrieved_docs
    else:
        state["retrieved_docs"] = []
        
    state["timing_dict"]["query_embed_and_match_time"] = round(time.time() - start_time, 4)
    return state

def rerank_node(state: GraphState):
    print("[NODE] Entering rerank_node...")
    start_time = time.time()
    retrieved_docs = state["retrieved_docs"]
    texts_to_rerank = [doc["sample"]["recipe_markdown"] for doc in retrieved_docs]
    images_to_rerank = [doc["sample"]["image"] for doc in retrieved_docs]
    query_text = state["input_query"] if isinstance(state["input_query"], str) else "image query"
    
    samples_to_rerank = [
        {"question": query_text, "doc_text": text, "doc_image": image}
        for text, image in zip(texts_to_rerank, images_to_rerank)
    ]
    
    rerank_logits_list = []
    chunk_size = 4
    for i in range(0, len(samples_to_rerank), chunk_size):
        chunk = samples_to_rerank[i:i+chunk_size]
        batch_dict_rerank = rerank_processor.process_queries_documents_crossencoder(chunk)
        batch_dict_rerank = {
            k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
            for k, v in batch_dict_rerank.items()
        }
        
        with torch.inference_mode():
            outputs = rerank_model(**batch_dict_rerank, return_dict=True)
            rerank_logits_list.append(outputs.logits.squeeze(-1))
            
        del batch_dict_rerank
        del outputs
        torch.cuda.empty_cache()
    
    rerank_logits = torch.cat(rerank_logits_list, dim=0)
    rerank_sorted_indices = torch.argsort(rerank_logits, descending=True).tolist()
    
    reranked_docs = []
    for new_rank, original_rank in enumerate(rerank_sorted_indices):
        doc = retrieved_docs[original_rank]
        movement = new_rank - original_rank
        movement_string = f"{movement}" if movement == 0 else (f"+{abs(movement)}" if movement < 0 else f"-{movement}")
        doc["rerank_string"] = f"Orig rank: {original_rank} | New rank: {new_rank} | Move: {movement_string}"
        reranked_docs.append(doc)
        
    state["reranked_docs"] = reranked_docs
    state["timing_dict"]["rerank_time"] = round(time.time() - start_time, 4)
    return state

def generate_node(state: GraphState):
    print("[NODE] Entering generate_node...")
    start_time = time.time()
    docs = state.get("reranked_docs") or state["retrieved_docs"]
    recipe_texts = [doc["sample"]["recipe_markdown"] for doc in docs[:3]]
    
    summary = generate_recipe_summary(recipe_texts, model=qwen_model, processor=qwen_processor)
    state["summary"] = summary.replace("```markdown", "").replace("```", "")
    state["timing_dict"]["generation_time"] = round(time.time() - start_time, 4)
    return state

def update_cache_node(state: GraphState):
    print("[NODE] Entering update_cache_node...")
    if state["cache_hit"] or not redis_cache:
        return state
        
    try:
        docs = state.get("reranked_docs") or state["retrieved_docs"]
        dataset_indices = [doc["dataset_index"] for doc in docs]
        scores = [float(doc.get("score", 0)) for doc in docs]
        rerank_strings = [doc.get("rerank_string", "") for doc in docs]
        
        response_data = {
            "summary": state.get("summary", ""),
            "retrieved_docs_indices": dataset_indices,
            "scores": scores,
            "rerank_strings": rerank_strings
        }
        
        query = state["input_query"]
        query_str = query if isinstance(query, str) else "image_query"
        
        doc_id = str(uuid.uuid4())
        vector_bytes = np.array(state["query_embedding"], dtype=np.float32).tobytes()
        
        redis_cache.load([{
            "id": doc_id,
            "prompt": query_str,
            "response": json.dumps(response_data),
            "vector": vector_bytes
        }], id_field="id")
    except Exception as e:
        print(f"[WARNING] Redis cache update failed: {e}")
        
    return state

# ============================================================================
# Build LangGraph
# ============================================================================

workflow = StateGraph(GraphState)

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("retrieve_pinecone", retrieve_pinecone_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("generate", generate_node)
workflow.add_node("update_cache", update_cache_node)

def post_cache_route(state: GraphState):
    print("[NODE] Entering post_cache_route...")
    if not state["cache_hit"]:
        return "retrieve_pinecone"
        
    wants_summary = state.get("generate_summary_option") == "True"
    has_valid_summary = bool(state.get("summary"))
    
    if wants_summary and not has_valid_summary:
        return "generate"
        
    return "end"

def route_after_retrieve(state: GraphState):
    print("[NODE] Entering route_after_retrieve...")
    if state["rerank_option"] == "True":
        return "rerank"
    elif state.get("generate_summary_option") == "True":
        return "generate"
    else:
        return "update_cache"

def route_after_rerank(state: GraphState):
    print("[NODE] Entering route_after_rerank...")
    if state.get("generate_summary_option") == "True":
        return "generate"
    else:
        return "update_cache"

workflow.add_conditional_edges("check_cache", post_cache_route, {"end": END, "retrieve_pinecone": "retrieve_pinecone", "generate": "generate"})
workflow.add_conditional_edges("retrieve_pinecone", route_after_retrieve, {"rerank": "rerank", "generate": "generate", "update_cache": "update_cache"})
workflow.add_conditional_edges("rerank", route_after_rerank, {"generate": "generate", "update_cache": "update_cache"})
workflow.add_edge("generate", "update_cache")
workflow.add_edge("update_cache", END)

workflow.add_edge(START, "check_cache")

graph = workflow.compile()
