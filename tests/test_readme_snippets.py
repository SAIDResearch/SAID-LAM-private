#!/usr/bin/env python3
"""
Test all README and COMPLETE_SUBMISSION_GUIDE code snippets.
Run from repo root:  PYTHONPATH=. python tests/test_readme_snippets.py
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

def run(name, fn):
    try:
        fn()
        print(f"  PASS: {name}")
        return True
    except Exception as e:
        print(f"  FAIL: {name} -> {e}")
        return False

def main():
    from said_lam import LAM
    import numpy as np

    results = []

    # --- README: After (LAM) 286-294 ---
    def snippet_after_lam():
        model = LAM("SAIDResearch/SAID-LAM-v1")
        embeddings = model.encode(["Hello world", "Semantic search"])
        assert embeddings.shape == (2, 384), embeddings.shape
        similarity = embeddings[0] @ embeddings[1]
        assert -1 <= similarity <= 1
    results.append(run("README After (LAM) — encode + similarity", snippet_after_lam))

    # --- README: FREE Tier 310-322 ---
    def snippet_free_tier():
        model = LAM("SAIDResearch/SAID-LAM-v1")
        embeddings = model.encode(["Hello world", "Semantic search is powerful"])
        assert embeddings.shape == (2, 384)
        similarity = embeddings[0] @ embeddings[1]
        assert -1 <= similarity <= 1
    results.append(run("README FREE Tier — embeddings + cosine", snippet_free_tier))

    # --- README: Similarity between texts 332-338 ---
    def snippet_similarity():
        model = LAM("SAIDResearch/SAID-LAM-v1")
        emb = model.encode(["The cat sat on the mat", "A kitten rested on the rug"])
        similarity = float(emb[0] @ emb[1])
        assert -1 <= similarity <= 1
    results.append(run("README Similarity between texts", snippet_similarity))

    # --- README: Batch similarity matrix 340-351 ---
    def snippet_batch_matrix():
        model = LAM("SAIDResearch/SAID-LAM-v1")
        queries = ["How is the weather?", "What time is it?"]
        candidates = ["Is it raining today?", "Do you have the time?", "Nice shoes"]
        emb_q = model.encode(queries)
        emb_c = model.encode(candidates)
        assert emb_q.shape == (2, 384)
        assert emb_c.shape == (3, 384)
        sim_matrix = emb_q @ emb_c.T
        assert sim_matrix.shape == (2, 3)
    results.append(run("README Batch similarity matrix", snippet_batch_matrix))

    # --- README: Semantic search over corpus 353-366 ---
    def snippet_semantic_search():
        model = LAM("SAIDResearch/SAID-LAM-v1")
        corpus = ["Python is a language", "The Eiffel Tower is in Paris",
                  "ML uses neural networks", "Speed of light is 299792458 m/s"]
        corpus_emb = model.encode(corpus)
        query_emb = model.encode(["fastest thing in physics"])
        scores = (query_emb @ corpus_emb.T)[0]
        ranked = np.argsort(scores)[::-1]
        assert len(ranked) == 4
        assert all(0 <= r < 4 for r in ranked)
        assert np.all(scores >= -1) and np.all(scores <= 1)
    results.append(run("README Semantic search over corpus", snippet_semantic_search))

    # --- README: Matryoshka 370-375 ---
    def snippet_matryoshka():
        model = LAM("SAIDResearch/SAID-LAM-v1")
        emb_128 = model.encode(["Hello world"], output_dim=128)
        emb_64 = model.encode(["Hello world"], output_dim=64)
        assert emb_128.shape == (1, 128)
        assert emb_64.shape == (1, 64)
    results.append(run("README Matryoshka dimensionality reduction", snippet_matryoshka))

    # --- README: Token limits / encode 393-399 ---
    def snippet_token_limits():
        model = LAM("SAIDResearch/SAID-LAM-v1")
        embeddings = model.encode(["short text", "very long text..."])
        assert embeddings.shape == (2, 384)
        embeddings_128 = model.encode(["short text", "very long text..."], output_dim=128)
        assert embeddings_128.shape == (2, 128)
    results.append(run("README encode() 12K / output_dim", snippet_token_limits))

    # --- COMPLETE_SUBMISSION_GUIDE: Verify Auto-Download 110-117 ---
    def snippet_hf_auto_download():
        # Uses HF id; will use cache if present or download from HF
        model = LAM("SAIDResearch/SAID-LAM-v1")
        emb = model.encode(["Hello world"])
        assert emb.shape == (1, 384), emb.shape
    results.append(run("COMPLETE_SUBMISSION_GUIDE Verify Auto-Download (HF)", snippet_hf_auto_download))

    # Summary
    print()
    total = len(results)
    passed = sum(results)
    print(f"--- Summary: {passed}/{total} passed ---")
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
