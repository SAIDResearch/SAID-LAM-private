"""
Minimal smoke test for LAM encode() API.

Matches the usage example in hf_model_card/README.md:

    from said_lam import LAM
    model = LAM("SAIDResearch/SAID-LAM-v1")
    embeddings = model.encode(["Hello world", "Semantic search"])
    similarity = embeddings[0] @ embeddings[1]
"""

from __future__ import annotations

import numpy as np

from said_lam import LAM


def main() -> None:
    # Let LAM auto-select device (CUDA if available, else CPU)
    model = LAM("SAIDResearch/SAID-LAM-v1")
    texts = ["Hello world", "Semantic search"]
    embeddings = model.encode(texts)

    assert embeddings.shape == (2, 384), f"Unexpected shape: {embeddings.shape}"

    sim = float(embeddings[0] @ embeddings[1])
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Cosine similarity between texts: {sim:.4f}")


if __name__ == "__main__":
    main()

