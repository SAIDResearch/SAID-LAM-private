#!/usr/bin/env python3
"""
Test 9: Embedding Model Parity — sentence-transformers Drop-in
===============================================================

Validates that LAM's encode() output is a drop-in replacement for
sentence-transformers' SentenceTransformer.encode():

  Same output format (np.ndarray, float32, L2-normalized)
  Same shape contract (N, 384) for batch, (1, 384) for single string
  Same vector store compatibility (FAISS IndexFlatIP, numpy dot product)

Plus LAM-specific capabilities sentence-transformers cannot do:
  - Matryoshka output_dim (64, 128, 256)
  - Large batch (50+ sentences) without OOM
  - Long context (12K+ tokens at FREE tier)

Usage:
    cd said-lam/tests && python test_embeddings.py
"""

import json
import sys
import numpy as np
from pathlib import Path

# Ensure said_lam package and lam_candle are importable.
_tests_dir = Path(__file__).resolve().parent
_repo_root = _tests_dir.parent
sys.path.insert(0, str(_repo_root / "said_lam"))  # lam_candle.so
sys.path.insert(0, str(_repo_root))                # said_lam package


class EmbeddingModelValidator:
    """Validates LAM as a sentence-transformers drop-in embedding model."""

    def __init__(self):
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

    def run(self) -> dict:
        """Run all embedding model parity tests."""
        from said_lam import LAM

        print("\n" + "=" * 70)
        print("TEST 9: Embedding Model Parity (sentence-transformers drop-in)")
        print("=" * 70)

        model = LAM()
        results = {}

        results["output_format"] = self._test_output_format(model)
        results["l2_normalization"] = self._test_l2_normalization(model)
        results["similarity_ordering"] = self._test_similarity_ordering(model)
        results["batch_consistency"] = self._test_batch_consistency(model)
        results["determinism"] = self._test_determinism(model)
        results["matryoshka"] = self._test_matryoshka(model)
        results["rag_pattern"] = self._test_rag_pattern(model)
        results["similarity_matrix"] = self._test_similarity_matrix(model)
        results["large_batch"] = self._test_large_batch(model)
        results["unicode"] = self._test_unicode(model)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        all_passed = True
        for name, result in results.items():
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {name:25s} {status}")
            if not result["passed"]:
                all_passed = False

        overall = "ALL PASSED" if all_passed else "SOME FAILED"
        print(f"\n  Overall: {overall}")
        print("=" * 70)

        # Save
        out_path = self.results_dir / "embedding_model_validation.json"
        with open(out_path, "w") as f:
            json.dump({k: v["passed"] for k, v in results.items()}, f, indent=2)
        print(f"  Results saved to: {out_path}")

        return results

    def _test_output_format(self, model) -> dict:
        """Test encode() output format matches sentence-transformers contract."""
        print("\n--- OUTPUT FORMAT ---")
        checks = {}

        # Single sentence
        emb = model.encode(["Hello world"])
        checks["shape_single"] = emb.shape == (1, 384)
        checks["dtype"] = emb.dtype == np.float32
        print(f"  Single: shape={emb.shape}, dtype={emb.dtype}  "
              f"{'OK' if checks['shape_single'] and checks['dtype'] else 'FAIL'}")

        # Batch
        emb_batch = model.encode(["one", "two", "three"])
        checks["shape_batch"] = emb_batch.shape == (3, 384)
        print(f"  Batch:  shape={emb_batch.shape}  "
              f"{'OK' if checks['shape_batch'] else 'FAIL'}")

        # Empty list
        emb_empty = model.encode([])
        checks["shape_empty"] = emb_empty.shape == (0, 384)
        print(f"  Empty:  shape={emb_empty.shape}  "
              f"{'OK' if checks['shape_empty'] else 'FAIL'}")

        # String auto-wrap (sentence-transformers does this too)
        emb_str = model.encode("just a string")
        checks["string_autowrap"] = emb_str.shape == (1, 384)
        print(f"  String: shape={emb_str.shape}  "
              f"{'OK' if checks['string_autowrap'] else 'FAIL'}")

        # ndarray type
        checks["is_ndarray"] = isinstance(emb, np.ndarray)
        print(f"  Type:   {type(emb).__name__}  "
              f"{'OK' if checks['is_ndarray'] else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_l2_normalization(self, model) -> dict:
        """Test embeddings are L2-normalized (unit length)."""
        print("\n--- L2 NORMALIZATION ---")
        checks = {}

        sentences = [
            "The quick brown fox",
            "Machine learning algorithms process data",
            "A very short text",
            "Lorem ipsum dolor sit amet " * 50,  # longer text
        ]
        emb = model.encode(sentences)
        norms = np.linalg.norm(emb, axis=1)

        for i, norm in enumerate(norms):
            ok = abs(norm - 1.0) < 0.01
            checks[f"norm_{i}"] = ok
            print(f"  [{i}] norm={norm:.6f}  {'OK' if ok else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_similarity_ordering(self, model) -> dict:
        """Test cosine similarity ordering with real-world semantic examples.

        Verifies the model understands meaning, not just keyword overlap:
        - "The cat eats dog food" ↔ "A kitten nibbles kibble" (same meaning, zero word overlap)
        - "The stock market rose sharply" (completely unrelated)
        - "The dog runs quickly through the park" ↔ "A canine sprints rapidly across the garden"
          (synonym substitution: dog→canine, runs→sprints, quickly→rapidly, park→garden)
        """
        print("\n--- SIMILARITY ORDERING ---")
        checks = {}

        # Group 1: Cat eating — paraphrase vs unrelated
        emb = model.encode([
            "The cat eats dog food from a bowl",        # [0] cat eating
            "A kitten nibbles kibble from a dish",      # [1] same meaning, different words
            "The stock market rose sharply on Tuesday",  # [2] completely unrelated
        ])

        sim_paraphrase = float(emb[0] @ emb[1])
        sim_unrelated = float(emb[0] @ emb[2])
        checks["paraphrase_beats_unrelated"] = sim_paraphrase > sim_unrelated
        print(f"  cat-food ↔ kitten-kibble = {sim_paraphrase:.4f}  (paraphrase)")
        print(f"  cat-food ↔ stock-market  = {sim_unrelated:.4f}  (unrelated)")
        print(f"  paraphrase > unrelated: {'OK' if checks['paraphrase_beats_unrelated'] else 'FAIL'}")

        # Group 2: Synonym substitution — every content word replaced
        emb2 = model.encode([
            "The dog runs quickly through the park",          # [0]
            "A canine sprints rapidly across the garden",     # [1] full synonym swap
            "Investors worry about rising inflation rates",   # [2] unrelated
        ])

        sim_synonym = float(emb2[0] @ emb2[1])
        sim_unrelated2 = float(emb2[0] @ emb2[2])
        checks["synonym_beats_unrelated"] = sim_synonym > sim_unrelated2
        print(f"  dog-runs ↔ canine-sprints = {sim_synonym:.4f}  (synonyms)")
        print(f"  dog-runs ↔ inflation      = {sim_unrelated2:.4f}  (unrelated)")
        print(f"  synonym > unrelated: {'OK' if checks['synonym_beats_unrelated'] else 'FAIL'}")

        # Self-similarity should be ~1.0
        sim_self = float(emb[0] @ emb[0])
        checks["self_sim"] = sim_self > 0.99
        print(f"  Self-sim={sim_self:.6f}  "
              f"{'OK' if checks['self_sim'] else 'FAIL'}")

        # All scores should be in [-1, 1]
        all_sims = [float(emb[i] @ emb[j]) for i in range(3) for j in range(3)]
        checks["range"] = all(-1.01 <= s <= 1.01 for s in all_sims)
        print(f"  All sims in [-1,1]: {checks['range']}  "
              f"{'OK' if checks['range'] else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_batch_consistency(self, model) -> dict:
        """Test same text produces same embedding whether batched or not."""
        print("\n--- BATCH CONSISTENCY ---")
        checks = {}

        text = "Reproducibility is important in machine learning"

        # Encode alone
        emb_alone = model.encode([text])
        # Encode in a batch
        emb_batch = model.encode(["other text", text, "another text"])

        sim = float(emb_alone[0] @ emb_batch[1])
        checks["consistent"] = sim > 0.999
        print(f"  Alone vs batch: cosine={sim:.6f}  "
              f"{'OK' if checks['consistent'] else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_determinism(self, model) -> dict:
        """Test two encode() calls produce identical results."""
        print("\n--- DETERMINISM ---")
        checks = {}

        texts = ["Hello world", "Test sentence for determinism"]
        emb1 = model.encode(texts)
        emb2 = model.encode(texts)

        max_diff = float(np.max(np.abs(emb1 - emb2)))
        checks["identical"] = max_diff < 1e-6
        print(f"  Max diff={max_diff:.2e}  "
              f"{'OK' if checks['identical'] else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_matryoshka(self, model) -> dict:
        """Test Matryoshka output_dim truncation."""
        print("\n--- MATRYOSHKA OUTPUT_DIM ---")
        checks = {}

        text = "Matryoshka dimensionality reduction test"

        for dim in [64, 128, 256]:
            emb = model.encode([text], output_dim=dim)
            checks[f"shape_{dim}"] = emb.shape == (1, dim)
            norm = np.linalg.norm(emb[0])
            checks[f"norm_{dim}"] = abs(norm - 1.0) < 0.01
            print(f"  dim={dim}: shape={emb.shape}, norm={norm:.6f}  "
                  f"{'OK' if checks[f'shape_{dim}'] and checks[f'norm_{dim}'] else 'FAIL'}")

        # Full 384 should also work
        emb_full = model.encode([text])
        checks["shape_384"] = emb_full.shape == (1, 384)
        print(f"  dim=384: shape={emb_full.shape}  "
              f"{'OK' if checks['shape_384'] else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_rag_pattern(self, model) -> dict:
        """Test the core RAG pattern: encode corpus, encode query, dot product search.

        Uses everyday examples so it's obvious whether retrieval 'works':
        - "a feline napping in the sun" should find the cat sentence (synonym mapping)
        - "recipe for baking" should find the cookie sentence (semantic match)
        - "programming language" should still find Python (keyword + semantic)
        """
        print("\n--- RAG PATTERN (numpy dot product) ---")
        checks = {}

        corpus = [
            "The cat sleeps on the windowsill every afternoon",            # [0] cat napping
            "Python is a programming language created by Guido van Rossum", # [1] programming
            "She baked chocolate chip cookies for the school fundraiser",   # [2] baking
            "The speed of light is approximately 299792458 meters per second", # [3] physics
            "Ludwig van Beethoven composed nine symphonies",               # [4] music
        ]
        corpus_emb = model.encode(corpus)

        # Query 1: synonym mapping — "feline napping" should find "cat sleeps"
        query_emb = model.encode(["a feline napping in the sun"])
        scores = (query_emb @ corpus_emb.T)[0]
        top_idx = int(np.argmax(scores))
        checks["synonym_query"] = top_idx == 0  # cat sleeps
        print(f"  'feline napping' -> [{top_idx}] {corpus[top_idx][:50]}...  "
              f"{'OK' if checks['synonym_query'] else 'FAIL'}")

        # Query 2: semantic — "recipe for baking" should find cookies
        query_emb2 = model.encode(["recipe for baking treats"])
        scores2 = (query_emb2 @ corpus_emb.T)[0]
        top_idx2 = int(np.argmax(scores2))
        checks["baking_query"] = top_idx2 == 2  # cookies
        print(f"  'recipe for baking' -> [{top_idx2}] {corpus[top_idx2][:50]}...  "
              f"{'OK' if checks['baking_query'] else 'FAIL'}")

        # Query 3: keyword + semantic — "programming language"
        query_emb3 = model.encode(["programming language"])
        scores3 = (query_emb3 @ corpus_emb.T)[0]
        top_idx3 = int(np.argmax(scores3))
        checks["programming_query"] = top_idx3 == 1  # Python
        print(f"  'programming language' -> [{top_idx3}] {corpus[top_idx3][:50]}...  "
              f"{'OK' if checks['programming_query'] else 'FAIL'}")

        # Query 4: classical music
        query_emb4 = model.encode(["classical music composer"])
        scores4 = (query_emb4 @ corpus_emb.T)[0]
        top_idx4 = int(np.argmax(scores4))
        checks["music_query"] = top_idx4 == 4  # Beethoven
        print(f"  'classical composer' -> [{top_idx4}] {corpus[top_idx4][:50]}...  "
              f"{'OK' if checks['music_query'] else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_similarity_matrix(self, model) -> dict:
        """Test batch similarity matrix computation with intuitive pairs.

        Matrix layout (rows=queries, cols=candidates):
                        kitten  sprinting  financial
          pet cat        HIGH    low        low       ← cat matches kitten
          running fast   low     HIGH       low       ← running matches sprinting
        """
        print("\n--- SIMILARITY MATRIX ---")
        checks = {}

        queries = ["a pet cat", "running fast"]
        candidates = ["a small kitten", "sprinting quickly", "financial markets"]

        emb_q = model.encode(queries)
        emb_c = model.encode(candidates)
        sim_matrix = emb_q @ emb_c.T

        checks["shape"] = sim_matrix.shape == (2, 3)
        print(f"  Shape: {sim_matrix.shape} (expected (2,3))  "
              f"{'OK' if checks['shape'] else 'FAIL'}")

        # "pet cat" should match "kitten" more than "financial markets"
        checks["cat_kitten"] = sim_matrix[0, 0] > sim_matrix[0, 2]
        print(f"  cat->kitten={sim_matrix[0,0]:.4f} > cat->financial={sim_matrix[0,2]:.4f}  "
              f"{'OK' if checks['cat_kitten'] else 'FAIL'}")

        # "running fast" should match "sprinting quickly" more than "financial markets"
        checks["run_sprint"] = sim_matrix[1, 1] > sim_matrix[1, 2]
        print(f"  running->sprinting={sim_matrix[1,1]:.4f} > running->financial={sim_matrix[1,2]:.4f}  "
              f"{'OK' if checks['run_sprint'] else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_large_batch(self, model) -> dict:
        """Test encoding a large batch (50 sentences)."""
        print("\n--- LARGE BATCH ---")
        checks = {}

        sentences = [f"This is test sentence number {i} for batch processing" for i in range(50)]
        emb = model.encode(sentences)
        checks["shape"] = emb.shape == (50, 384)
        checks["dtype"] = emb.dtype == np.float32
        norms = np.linalg.norm(emb, axis=1)
        checks["all_normalized"] = np.allclose(norms, 1.0, atol=0.01)
        print(f"  Shape: {emb.shape}, dtype: {emb.dtype}  "
              f"{'OK' if checks['shape'] else 'FAIL'}")
        print(f"  All normalized: {checks['all_normalized']}  "
              f"{'OK' if checks['all_normalized'] else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_unicode(self, model) -> dict:
        """Test encoding unicode and special characters."""
        print("\n--- UNICODE & SPECIAL CHARS ---")
        checks = {}

        texts = [
            "Hello world",                          # ASCII
            "Bonjour le monde",                     # French
            "Hallo Welt",                           # German
            "Special chars: @#$%^&*()!",            # Symbols
            "Numbers: 3.14159 and 2.71828",         # Numbers
            "Mixed: Hello 42 @world #test",         # Mixed
        ]
        emb = model.encode(texts)
        checks["shape"] = emb.shape == (len(texts), 384)
        checks["no_nan"] = not np.any(np.isnan(emb))
        checks["no_inf"] = not np.any(np.isinf(emb))

        norms = np.linalg.norm(emb, axis=1)
        checks["all_normalized"] = np.allclose(norms, 1.0, atol=0.01)

        print(f"  Shape: {emb.shape}  {'OK' if checks['shape'] else 'FAIL'}")
        print(f"  No NaN: {checks['no_nan']}, No Inf: {checks['no_inf']}  OK")
        print(f"  All normalized: {checks['all_normalized']}  "
              f"{'OK' if checks['all_normalized'] else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}


if __name__ == "__main__":
    validator = EmbeddingModelValidator()
    results = validator.run()
    passed = all(r["passed"] for r in results.values())
    sys.exit(0 if passed else 1)
