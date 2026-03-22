"""
app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gradio web app for text-to-image retrieval supporting both CLIP and SigLIP.

How it works:
  1. At startup: load both models + both ChromaDB collections.
  2. On query : encode the user's prompt with the selected model → 
     search the respective collection → return top-K images.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from pathlib import Path

import chromadb
import gradio as gr
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_CONFIG = {
    "CLIP": {
        "path": "openai/clip-vit-base-patch16",
        "collection_name": "flickr8k_clip"
    },
    "SigLIP": {
        "path": "google/siglip-base-patch16-224",
        "collection_name": "flickr8k_siglip"
    }
}

IMAGES_DIR   = Path("data/images")
CHROMA_DIR   = Path("chroma_db")
DEFAULT_TOPK = 10
MAX_TOPK     = 60
# ──────────────────────────────────────────────────────────────────────────────
USE_LOCAL_IMAGES = IMAGES_DIR.exists()

if USE_LOCAL_IMAGES:
    print(f"Image source: local disk ({IMAGES_DIR})\n")
    dataset = None
else:
    print("Image source: HuggingFace dataset (data/images/ not found locally)")
    print("Loading Flickr8k …")
    from datasets import load_dataset
    dataset = load_dataset("jxie/flickr8k", split="train+validation+test")
    print(f"  Dataset ready: {len(dataset)} images.\n")


def load_image(meta: dict) -> Image.Image:
    """
    Load an image from local disk or HuggingFace dataset depending on
    what is available at runtime.
    """
    if USE_LOCAL_IMAGES:
        return Image.open(IMAGES_DIR / meta["filename"]).convert("RGB")
    else:
        return dataset[meta["dataset_index"]]["image"].convert("RGB")

# ── Load once at startup ──────────────────────────────────────────────────────
print(f"\nStarting up on device: {DEVICE}")

loaded_models = {}
loaded_processors = {}
loaded_collections = {}

# 1. Carica i Modelli
for model_key, config in MODELS_CONFIG.items():
    print(f"Loading {model_key} model from {config['path']} …")
    model = AutoModel.from_pretrained(config["path"]).to(DEVICE)
    processor = AutoProcessor.from_pretrained(config["path"])
    model.eval()
    
    loaded_models[model_key] = model
    loaded_processors[model_key] = processor

print("\nConnecting to ChromaDB …")
if not (CHROMA_DIR / "chroma.sqlite3").exists():
    raise FileNotFoundError(
        f"ChromaDB not found at '{CHROMA_DIR}'. "
        "Run build_index.py first, then re-launch."
    )

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# 2. Carica le Collezioni
for model_key, config in MODELS_CONFIG.items():
    try:
        col = chroma_client.get_collection(config["collection_name"])
        loaded_collections[model_key] = col
        print(f"  Collection '{config['collection_name']}' ready: {col.count()} images indexed.")
    except Exception as e:
        print(f"  Warning: Could not load collection '{config['collection_name']}'. Did you run build_index.py for {model_key}?")


# ── Core retrieval function ───────────────────────────────────────────────────
def retrieve(query: str, model_choice: str, top_k: int = DEFAULT_TOPK):
    """
    Encode `query` with the chosen model and return the top-k matching (image, score) pairs.
    """
    query = query.strip()
    if not query:
        return []

    if model_choice not in loaded_models or model_choice not in loaded_collections:
        return []

    model = loaded_models[model_choice]
    processor = loaded_processors[model_choice]
    collection = loaded_collections[model_choice]

    # Encode text
    inputs = processor(text=[query], return_tensors="pt", padding="max_length", truncation=True).to(DEVICE)
    
    with torch.inference_mode():
        output = model.get_text_features(**inputs)
        # Gestisce output che potrebbero differire leggermente tra architetture
        text_features = output.pooler_output if hasattr(output, "pooler_output") else output
        
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    query_vec = text_features.cpu().numpy().tolist()[0]

    # Vector search
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=int(top_k),
        include=["metadatas", "distances"],
    ) 

    output_images = []
    if results["distances"] and len(results["distances"]) > 0:
        for i , (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
            img = load_image(meta)
            caption = f"#{i + 1}"
            output_images.append((img,caption))
    
    return output_images


# ── Gradio UI ─────────────────────────────────────────────────────────────────
_EXAMPLES = [
    ["a dog playing in the snow", "CLIP"],
    ["children playing at a park", "SigLIP"],
    ["a man surfing ocean waves", "CLIP"],
    ["a cat sitting on a windowsill", "SigLIP"],
]

with gr.Blocks(
    title="Dual-Model Text-to-Image Retrieval",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 🔍 Text-to-Image Retrieval
        Compare **CLIP** and **SigLIP** models on the Flickr8k dataset.
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            query_box = gr.Textbox(
                placeholder="e.g.  a dog playing in the snow",
                label="Search prompt",
            )
        with gr.Column(scale=2):
            model_selector = gr.Radio(
                choices=["CLIP", "SigLIP"], 
                value="CLIP", 
                label="Model Engine"
            )
        with gr.Column(scale=2):
            topk_slider = gr.Slider(
                minimum=1, maximum=MAX_TOPK, value=DEFAULT_TOPK, step=1,
                label="Results to fetch",
            )
        with gr.Column(scale=1):
            search_btn = gr.Button("Search 🔎", variant="primary")

    gallery = gr.Gallery(
        label="Top results",
        columns=5,
        rows=2,
        height="auto",
        object_fit="cover",
        show_label=False,
    )

    gr.Examples(
        examples=_EXAMPLES,
        inputs=[query_box, model_selector],
        label="Try one of these …",
    )

    # Wire up interactions (pass model_selector come input aggiuntivo)
    search_btn.click(fn=retrieve, inputs=[query_box, model_selector, topk_slider], outputs=gallery)
    query_box.submit(fn=retrieve, inputs=[query_box, model_selector, topk_slider], outputs=gallery)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   
        share=False,             
    )