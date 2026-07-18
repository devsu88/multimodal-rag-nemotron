"""
Multimodal RAG Demo with Nemotron Embed VL and Rerank VL, Pinecone, Redis, and LangGraph

A Gradio demo for multimodal retrieval augmented generation using:
- Dataset: mrdbourke/recipe-synthetic-images-10k
- Embedding model: nvidia/llama-nemotron-embed-vl-1b-v2
- Rerank model: nvidia/llama-nemotron-rerank-vl-1b-v2
- Generation model: Qwen/Qwen3-VL-2B-Instruct
- Vector Store: Pinecone
- Cache: Redis Semantic Cache
- Orchestration: LangGraph
"""

import gradio as gr
import spaces
from PIL import Image

from models import redis_cache
from graph import graph
from utils import create_recipe_cards_html

# ============================================================================
# Main Retrieve Function
# ============================================================================

@spaces.GPU
def retrieve(
    query_text: str | None,
    query_image: Image.Image | None,
    rerank_option: str,
    generate_summary_option: str,
    cache_threshold: float,
    top_k: int
):
    if query_text and query_text.strip():
        input_query = query_text
    elif query_image is not None:
        input_query = query_image
    else:
        raise gr.Error("Please provide either a text query or an image query.")
        
    initial_state = {
        "input_query": input_query,
        "rerank_option": rerank_option,
        "generate_summary_option": generate_summary_option,
        "cache_threshold": cache_threshold,
        "top_k": top_k,
        "timing_dict": {}
    }
    
    final_state = graph.invoke(initial_state)
    
    docs = final_state.get("reranked_docs") or final_state.get("retrieved_docs", [])
    
    output_image_gallery = [
        (doc["sample"]["image"], doc.get("rerank_string", f"Score: {doc.get('score', 0)}"))
        for doc in docs[:3]
    ]
    
    output_recipe_cards_html = create_recipe_cards_html(
        scores_and_samples=docs,
        num_results=3
    )
    
    summary = final_state.get("summary", "No summary generated.")
    timing_dict = final_state.get("timing_dict", {})
    
    return output_image_gallery, output_recipe_cards_html, summary, timing_dict

# ============================================================================
# Gradio Interface
# ============================================================================

def clear_cache_ui():
    print("[UI] Clear Cache button clicked.")
    if redis_cache:
        try:
            redis_cache.client.flushdb()
            redis_cache.create(overwrite=True)
            return "✅ Cache cleared successfully!"
        except Exception as e:
            return f"❌ Error clearing cache: {e}"
    return "❌ Cache not enabled."

with gr.Blocks(title="Multimodal RAG Demo") as demo:
    gr.Markdown("""# 👁️📑 Multimodal RAG Demo with Nemotron, Pinecone, Redis & LangGraph""")
    with gr.Row():
        with gr.Column(scale=1):
            query_text = gr.Textbox(label="Text Query", lines=2)
            query_image = gr.Image(label="Image Query", type="pil", height=200)
            generate_summary_option = gr.Radio(choices=["True", "False"], value="False", label="Generate summary")
            rerank_option = gr.Radio(choices=["True", "False"], value="False", label="Rerank results")
            cache_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.15, step=0.01, label="Cache Vector Distance Threshold")
            top_k_slider = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Retrieval Top K")
            search_btn = gr.Button("Search", variant="primary", size="lg")
            clear_cache_btn = gr.Button("Clear Cache", variant="secondary")
            cache_status_msg = gr.Markdown("")
        with gr.Column(scale=2):
            gallery_output = gr.Gallery(label="Retrieved Images", columns=3, height="auto", object_fit="cover")
            recipes_html = gr.HTML(label="Retrieved Texts")
            summary_generation = gr.Markdown(label="Summary")
            timing_output = gr.JSON(label="Timings")

    gr.Examples(
        examples=[
            ["best omelette recipes", None, "False", "False", 0.15, 20],
            ["best omelette recipes", None, "False", "True", 0.15, 20],
            ["best omelette recipes", None, "True", "True", 0.15, 20],
            ["eggplant dip", None, "True", "True", 0.15, 20]
        ],
        inputs=[query_text, query_image, rerank_option, generate_summary_option, cache_threshold, top_k_slider],
        label="Example Queries"
    )

    search_btn.click(
        fn=retrieve,
        inputs=[query_text, query_image, rerank_option, generate_summary_option, cache_threshold, top_k_slider],
        outputs=[gallery_output, recipes_html, summary_generation, timing_output]
    )

    clear_cache_btn.click(
        fn=clear_cache_ui,
        inputs=[],
        outputs=[cache_status_msg]
    )

if __name__ == "__main__":
    demo.launch()
