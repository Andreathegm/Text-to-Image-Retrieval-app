"""
app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gradio web app for text-to-image retrieval.

How it works:
  1. At startup: load CLIP (text encoder) + ChromaDB collection (pre-built)
  2. On query : encode the user's text prompt → cosine search → return top-K images

Run locally:
  python app.py

Deploy to HuggingFace Spaces:
  Push this file + requirements.txt + chroma_db/ + data/images/ to your Space.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from pathlib import Path

import chromadb
import gradio as gr
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "openai/clip-vit-base-patch16"
IMAGES_DIR   = Path("data/images")
CHROMA_DIR   = Path("chroma_db")
COLLECTION   = "flickr8k"
DEFAULT_TOPK = 10
MAX_TOPK     = 60
# ──────────────────────────────────────────────────────────────────────────────


# ── Load once at startup (not inside the handler) ─────────────────────────────
print(f"\nStarting up on device: {DEVICE}")

print("Loading CLIP model …")
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
print("  CLIP ready.\n")

print("Connecting to ChromaDB …")
if not (CHROMA_DIR / "chroma.sqlite3").exists():
    raise FileNotFoundError(
        f"ChromaDB not found at '{CHROMA_DIR}'. "
        "Run build_index.py first, then re-launch."
    )
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection(COLLECTION)
print(f"  Collection ready: {collection.count()} images indexed.\n")


# ── Core retrieval function ───────────────────────────────────────────────────
def retrieve(query: str, top_k: int = DEFAULT_TOPK) -> list[tuple[Image.Image, str]]:
    """
    Encode `query` with CLIP and return the top-k matching (image, score) pairs.
    Returns an empty list when the query is blank.
    """
    query = query.strip()
    if not query:
        return []

    # Encode text
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.inference_mode():
        output = model.get_text_features(**inputs)
        text_features = output.pooler_output if hasattr(output, "pooler_output") else output
        
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    query_vec = text_features.cpu().numpy().tolist()[0]

    # Vector search
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=int(top_k),
        include=["metadatas", "distances"],
    )

    # Load and label images
    # output = []
    # for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
    #     img_path = IMAGES_DIR / meta["filename"]
    #     if not img_path.exists():
    #         continue
    #     img = Image.open(img_path).convert("RGB")
    #     # Chroma cosine distance: 0 = identical, 2 = opposite
    #     # Convert to a 0–100 similarity score for display
    #     similarity = round((1 - dist / 2) * 100, 1)
    #     caption = f"Score: {similarity}%"
    #     output.append((img, caption))

    # return output
    # Il range effettivo di similarità coseno di CLIP va tipicamente da 0.15 (scarso) a 0.35 (ottimo)
    # Distanza Chroma = 1 - CosSim -> quindi le distanze andranno da circa 0.85 (scarso) a 0.65 (ottimo)
    
    MIN_EXPECTED_SIM = 0.15 
    MAX_EXPECTED_SIM = 0.32 

    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        img_path = IMAGES_DIR / meta["filename"]
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        
        # 1. Recupera la similarità coseno originale dalla distanza di Chroma
        cos_sim = 1 - dist
        
        # 2. Clampa il valore per evitare percentuali fuori scala (<0% o >100%)
        cos_sim_clamped = max(MIN_EXPECTED_SIM, min(MAX_EXPECTED_SIM, cos_sim))
        
        # 3. Mappa il range ristretto sulla scala 0-100
        normalized_score = (cos_sim_clamped - MIN_EXPECTED_SIM) / (MAX_EXPECTED_SIM - MIN_EXPECTED_SIM)
        similarity_pct = round(normalized_score * 100, 1)
        
        caption = f"Score: {similarity_pct}%"
        output.append((img, caption))
    
    return output


# ── Gradio UI ─────────────────────────────────────────────────────────────────
_EXAMPLES = [
    ["a dog playing in the snow"],
    ["children playing at a park"],
    ["a man surfing ocean waves"],
    ["a woman reading a book"],
    ["a group of people watching a performance"],
    ["a cat sitting on a windowsill"],
    ["a bike race on a mountain trail"],
    ["fireworks over a city at night"],
]

with gr.Blocks(
    title="CLIP Text-to-Image Retrieval",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 🔍 Text-to-Image Retrieval
        Enter a natural language description and find matching images from the **Flickr8k** dataset.
        Built with [CLIP](https://openai.com/research/clip) (ViT-B/16) + [ChromaDB](https://www.trychroma.com/).
        """
    )

    with gr.Row():
        query_box = gr.Textbox(
            placeholder="e.g.  a dog playing in the snow",
            label="Search prompt",
            scale=5,
        )
        topk_slider = gr.Slider(
            minimum=1, maximum=MAX_TOPK, value=DEFAULT_TOPK, step=1,
            label="Results",
            scale=1,
        )
        search_btn = gr.Button("Search 🔎", variant="primary", scale=1)

    gallery = gr.Gallery(
        label="Top results",
        columns=5,
        rows=2,
        height="auto",
        object_fit="cover",
        show_label=True,
    )

    gr.Examples(
        examples=_EXAMPLES,
        inputs=query_box,
        label="Try one of these …",
    )

    # Wire up interactions
    search_btn.click(fn=retrieve, inputs=[query_box, topk_slider], outputs=gallery)
    query_box.submit(fn=retrieve, inputs=[query_box, topk_slider], outputs=gallery)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # listen on all interfaces (needed for LAN access)
        share=False,              # set True for a temporary public gradio.live URL
    )