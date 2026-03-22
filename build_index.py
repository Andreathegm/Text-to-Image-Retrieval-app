"""
build_index.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run ONCE on a machine with a GPU to:
  1. Download the Flickr8k dataset from HuggingFace
  2. Encode every image with CLIP (ViT-B/16)
  3. Persist embeddings in a local ChromaDB vector database
  4. Save 256×256 JPEG thumbnails to data/images/

Outputs (commit both to your HuggingFace Space):
  chroma_db/      ← vector index (~20 MB)
  data/images/    ← thumbnails  (~80–150 MB, use git-lfs)

Usage:
  python build_index.py
  python build_index.py --batch-size 32   # if you hit OOM
  python build_index.py --splits train    # index only the train split
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
from pathlib import Path

import chromadb
import torch
from datasets import load_dataset, concatenate_datasets
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ── Defaults ──────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "openai/clip-vit-base-patch16"
DATASET_NAME = "jxie/flickr8k"
ALL_SPLITS   = ["train", "validation", "test"]   # Flickr8k has all three
BATCH_SIZE   = 64          # safe default for 6 GB VRAM; lower if OOM
THUMBNAIL_SIZE = (256, 256)
IMAGES_DIR   = Path("data/images")
CHROMA_DIR   = Path("chroma_db")
COLLECTION   = "flickr8k"
CHROMA_UPSERT_CHUNK = 500  # how many vectors to upsert at once
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Build CLIP index for Flickr8k")
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


def encode_batch(images: list, model, processor, device) -> "np.ndarray":
    """Return L2-normalised CLIP image embeddings for a list of PIL images."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.get_image_features(pixel_values=inputs["pixel_values"])
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
    print(f"\n{'─'*60}")
    print(f"  Device      : {DEVICE}")
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Splits      : {args.splits}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"{'─'*60}\n")

    # ── Create output dirs ────────────────────────────────────────────────────
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load CLIP ─────────────────────────────────────────────────────────────
    print("Loading CLIP model …")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
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
            client.delete_collection(COLLECTION)
        except Exception:
            pass

    # get_or_create so re-runs are safe without --force
    collection = client.get_or_create_collection(
        name=COLLECTION,
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

        # Skip items already in the collection (safe for incremental runs)
        new_images, new_indices = [], []
        for i, global_idx in enumerate(range(batch_start, batch_start + len(pil_images))):
            if str(global_idx) not in pending_ids:
                new_images.append(pil_images[i].convert("RGB"))
                new_indices.append(global_idx)

        if not new_images:
            continue

        # Save thumbnails
        for img, idx in zip(new_images, new_indices):
            save_thumbnail(img, IMAGES_DIR / f"{idx:05d}.jpg")

        # Encode
        embeddings = encode_batch(new_images, model, processor, DEVICE)

        for emb, idx in zip(embeddings, new_indices):
            pending_embs.append(emb.tolist())
            pending_ids.append(str(idx))
            pending_meta.append({"filename": f"{idx:05d}.jpg", "dataset_index": idx})

        # Flush to Chroma periodically
        if len(pending_ids) >= CHROMA_UPSERT_CHUNK:
            flush_to_chroma(collection, pending_embs, pending_ids, pending_meta)
            pending_embs, pending_ids, pending_meta = [], [], []

    # Final flush
    flush_to_chroma(collection, pending_embs, pending_ids, pending_meta)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = collection.count()
    print(f"\n{'─'*60}")
    print(f"  ✓ Indexed {total} images")
    print(f"  ✓ ChromaDB saved to  : {CHROMA_DIR}/")
    print(f"  ✓ Thumbnails saved to: {IMAGES_DIR}/")
    print(f"{'─'*60}\n")
    print("Next step: commit chroma_db/ and data/images/ to your HF Space.\n")


if __name__ == "__main__":
    main()