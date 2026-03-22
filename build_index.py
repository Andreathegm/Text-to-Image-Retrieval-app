"""
build_index.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run ONCE on a machine with a GPU to:
  1. Download the Flickr8k dataset from HuggingFace
  2. Encode every image with CLIP or SigLIP
  3. Persist embeddings in a local ChromaDB vector database (separate collections)
  4. Save 256×256 JPEG thumbnails to data/images/

Usage:
  python build_index.py --model-type clip
  python build_index.py --model-type siglip
  python build_index.py --model-type siglip --batch-size 32
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
from pathlib import Path

import chromadb
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# ── Configurazione Modelli ────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "clip": {
        "hf_path": "openai/clip-vit-base-patch16",
        "collection": "flickr8k_clip"
    },
    "siglip": {
        "hf_path": "google/siglip-base-patch16-224",
        "collection": "flickr8k_siglip"
    }
}

# ── Defaults ──────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = "jxie/flickr8k"
ALL_SPLITS   = ["train", "validation", "test"]
BATCH_SIZE   = 64
THUMBNAIL_SIZE = (256, 256)
IMAGES_DIR   = Path("data/images")
CHROMA_DIR   = Path("chroma_db")
CHROMA_UPSERT_CHUNK = 500
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Build Vector Index for Flickr8k")
    parser.add_argument(
        "--model-type", type=str, required=True, choices=["clip", "siglip"],
        help="Scegli quale modello usare per generare gli embedding"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Images per forward pass (lower if CUDA OOM)"
    )
    parser.add_argument(
        "--splits", nargs="+", default=ALL_SPLITS,
        choices=ALL_SPLITS,
        help="Which dataset splits to index"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete and rebuild the collection from scratch"
    )
    return parser.parse_args()


def encode_batch(images: list, model, processor, device) -> np.ndarray:
    """Return L2-normalised image embeddings for a list of PIL images."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.get_image_features(pixel_values=inputs["pixel_values"])
        # Gestisce eventuali differenze nell'output tra architetture
        features = output.pooler_output if hasattr(output, "pooler_output") else output
    
    features = torch.nn.functional.normalize(features, dim=-1)
    return features.cpu().numpy()


def save_thumbnail(img: Image.Image, filepath: Path):
    """Resize image to THUMBNAIL_SIZE and save as JPEG (skip if exists)."""
    if filepath.exists():
        return
    rgb = img.convert("RGB")
    rgb.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(filepath, "JPEG", quality=85)


def flush_to_chroma(collection, embeddings, ids, metadatas):
    """Upsert a batch of vectors into ChromaDB."""
    if not ids:
        return
    collection.upsert(
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )


def main():
    args = parse_args()
    
    model_config = MODEL_REGISTRY[args.model_type]
    hf_model_path = model_config["hf_path"]
    collection_name = model_config["collection"]

    print(f"\n{'─'*60}")
    print(f"  Device      : {DEVICE}")
    print(f"  Model Type  : {args.model_type.upper()}")
    print(f"  HF Path     : {hf_model_path}")
    print(f"  Collection  : {collection_name}")
    print(f"  Splits      : {args.splits}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"{'─'*60}\n")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Model ────────────────────────────────────────────────────────────
    print(f"Loading {args.model_type.upper()} model …")
    model = AutoModel.from_pretrained(hf_model_path).to(DEVICE)
    processor = AutoProcessor.from_pretrained(hf_model_path)
    model.eval()
    print("  Done.\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Downloading {DATASET_NAME} …")
    splits = [load_dataset(DATASET_NAME, split=s) for s in args.splits]
    dataset = concatenate_datasets(splits)
    print(f"  Total images to index: {len(dataset)}\n")

    # ── ChromaDB setup ────────────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if args.force:
        print("  --force flag set: deleting existing collection …")
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    already_indexed = collection.count()
    if already_indexed > 0 and not args.force:
        print(
            f"  Collection already has {already_indexed} vectors. "
            "Run with --force to rebuild.\n"
        )

    # ── Encode and index ──────────────────────────────────────────────────────
    print("Encoding images …\n")

    pending_embs, pending_ids, pending_meta = [], [], []

    for batch_start in tqdm(range(0, len(dataset), args.batch_size), unit="batch"):
        batch = dataset[batch_start : batch_start + args.batch_size]
        pil_images = batch["image"]

        new_images, new_indices = [], []
        for i, global_idx in enumerate(range(batch_start, batch_start + len(pil_images))):
            if str(global_idx) not in pending_ids:
                new_images.append(pil_images[i].convert("RGB"))
                new_indices.append(global_idx)

        if not new_images:
            continue

        for img, idx in zip(new_images, new_indices):
            save_thumbnail(img, IMAGES_DIR / f"{idx:05d}.jpg")

        embeddings = encode_batch(new_images, model, processor, DEVICE)

        for emb, idx in zip(embeddings, new_indices):
            pending_embs.append(emb.tolist())
            pending_ids.append(str(idx))
            pending_meta.append({"filename": f"{idx:05d}.jpg", "dataset_index": idx})

        if len(pending_ids) >= CHROMA_UPSERT_CHUNK:
            flush_to_chroma(collection, pending_embs, pending_ids, pending_meta)
            pending_embs, pending_ids, pending_meta = [], [], []

    flush_to_chroma(collection, pending_embs, pending_ids, pending_meta)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = collection.count()
    print(f"\n{'─'*60}")
    print(f"  ✓ Indexed {total} images into '{collection_name}'")
    print(f"  ✓ ChromaDB saved to  : {CHROMA_DIR}/")
    print(f"  ✓ Thumbnails saved to: {IMAGES_DIR}/")
    print(f"{'─'*60}\n")

if __name__ == "__main__":
    main()