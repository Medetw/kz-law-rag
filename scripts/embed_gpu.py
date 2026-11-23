"""Embed exported chunks on a GPU machine (one-off bulk indexing).

    # on the laptop
    kzlaw export-chunks data/chunks.jsonl
    # on the GPU box (needs only this file + chunks.jsonl)
    uv run --with sentence-transformers --with torch python embed_gpu.py chunks.jsonl vectors.npz
    # back on the laptop
    kzlaw load-vectors data/chunks.jsonl data/vectors.npz

Uses the same "passage: " prefix and normalisation as kzlaw.embeddings, so the API can embed queries on CPU.
"""

import argparse
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("chunks")
parser.add_argument("out")
parser.add_argument("--model", default="intfloat/multilingual-e5-small", help="HF id or local folder")
parser.add_argument("--name", default="intfloat/multilingual-e5-small", help="model id stored with the vectors")
parser.add_argument("--batch", type=int, default=128)
args = parser.parse_args()

with open(args.chunks, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]
model = SentenceTransformer(args.model, device="cuda")
t0 = time.time()
vecs = model.encode(
    [f"passage: {r['text']}" for r in rows],
    batch_size=args.batch,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype(np.float32)
np.savez_compressed(args.out, ids=np.array([r["id"] for r in rows]), vectors=vecs, model=np.array(args.name))
print(f"{len(rows)} chunks, dim {vecs.shape[1]}, {time.time() - t0:.1f}s -> {args.out}")  # noqa: T201
