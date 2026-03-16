"""
Matryoshka dimensionality reduction smoke test for SAID-LAM.

Verifies that:
- output_dim={384, 256, 128, 64} all work
- embeddings remain L2-normalized
- truncated views stay highly aligned with the full 384-dim embedding

This mirrors the Matryoshka example in hf_model_card/README.md.
"""

from __future__ import annotations

import numpy as np

from said_lam import LAM


def main() -> None:
    # Let LAM auto-select device (CUDA if available, else CPU)
    model = LAM("SAIDResearch/SAID-LAM-v1")
    text = ["Hello world"]

    dims = [384, 256, 128, 64]
    rows = []

    # Full 384-dim reference embedding
    emb_full = model.encode(text)  # (1, 384)
    norm_full = float(np.linalg.norm(emb_full))

    for d in dims:
        emb = model.encode(text, output_dim=d)
        norm = float(np.linalg.norm(emb))

        # Cosine with the corresponding prefix of the 384-dim embedding
        prefix = emb_full[:, :d]
        cos = float(prefix[0] @ emb[0])

        rows.append((d, norm, cos))

    print("Matryoshka dimensionality reduction (\"Hello world\")")
    print(f"Full embedding norm (384 dim): {norm_full:.6f}")
    print()
    print(f"{'dim':>6} | {'norm':>10} | {'cos_with_full_prefix':>20}")
    print("-" * 40)
    for d, norm, cos in rows:
        print(f"{d:6d} | {norm:10.6f} | {cos:20.6f}")


if __name__ == "__main__":
    main()

"""
Test: Matryoshka Truncated Embeddings
======================================

Validates truncate_embeddings() — Matryoshka Representation Learning.

Checks:
1. Output shapes for each valid dimension (64, 128, 256, 384)
2. L2 normalization after truncation
3. Prefix consistency (dim-64 == first 64 of dim-128, etc.)
4. Invalid dimension rejection
5. Cosine similarity preserved across dimensions
"""

import sys
import json
import numpy as np
from pathlib import Path

# Ensure said_lam package and lam_candle are importable.
_tests_dir = Path(__file__).resolve().parent
_repo_root = _tests_dir.parent
sys.path.insert(0, str(_repo_root / "said_lam"))  # lam_candle.so
sys.path.insert(0, str(_repo_root))                # said_lam package


class MatryoshkaValidator:
    """Validates Matryoshka truncate_embeddings functionality."""

    def __init__(self):
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

    def run(self) -> dict:
        """Run all Matryoshka validation tests."""
        from said_lam import LAM

        print("=" * 70)
        print("TEST: Matryoshka Truncated Embeddings")
        print("=" * 70)

        model = LAM()
        results = {}

        results["output_shapes"] = self._test_output_shapes(model)
        results["l2_normalization"] = self._test_l2_normalization(model)
        results["prefix_consistency"] = self._test_prefix_consistency(model)
        results["invalid_dims"] = self._test_invalid_dims(model)
        results["cosine_similarity"] = self._test_cosine_similarity(model)
        results["batch"] = self._test_batch(model)

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
        print(f"  Overall: {overall}")
        print("=" * 70)

        # Save results
        out_path = self.results_dir / "matryoshka_validation.json"
        with open(out_path, "w") as f:
            json.dump({k: v["passed"] for k, v in results.items()}, f, indent=2)
        print(f"  Results saved to: {out_path}")

        return results

    def _test_output_shapes(self, model) -> dict:
        """Test output shapes for each valid dimension."""
        print("\n--- OUTPUT SHAPES ---")
        checks = {}
        emb = model.encode(["hello world", "test sentence"])

        for dim in [64, 128, 256, 384]:
            truncated = model.truncate_embeddings(emb, dim)
            expected = (2, dim)
            ok = truncated.shape == expected
            checks[f"dim_{dim}"] = ok
            print(f"  dim={dim}: shape={truncated.shape} (expected {expected})  {'OK' if ok else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_l2_normalization(self, model) -> dict:
        """Test that truncated embeddings are L2-normalized."""
        print("\n--- L2 NORMALIZATION ---")
        checks = {}
        emb = model.encode(["The quick brown fox jumps over the lazy dog"])

        for dim in [64, 128, 256]:
            truncated = model.truncate_embeddings(emb, dim)
            norm = np.linalg.norm(truncated[0])
            ok = abs(norm - 1.0) < 0.01
            checks[f"norm_{dim}"] = ok
            print(f"  dim={dim}: L2 norm={norm:.6f}  {'OK' if ok else 'FAIL'}")

        # 384 should also be normalized (unchanged from encode which normalizes)
        emb_384 = model.truncate_embeddings(emb, 384)
        norm_384 = np.linalg.norm(emb_384[0])
        ok = abs(norm_384 - 1.0) < 0.01
        checks["norm_384"] = ok
        print(f"  dim=384: L2 norm={norm_384:.6f} (unchanged)  {'OK' if ok else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_prefix_consistency(self, model) -> dict:
        """Test Matryoshka prefix property: first N dims match across truncation levels."""
        print("\n--- PREFIX CONSISTENCY ---")
        checks = {}
        emb = model.encode(["Matryoshka nesting test"])

        # Get unnormalized prefix comparison:
        # After truncation + renorm, the directions should be consistent
        t64 = model.truncate_embeddings(emb, 64)
        t128 = model.truncate_embeddings(emb, 128)
        t256 = model.truncate_embeddings(emb, 256)

        # The first 64 dims of t128 (before renorm) should point same direction as t64
        # Compare via cosine similarity of the prefix
        prefix_128 = t128[0, :64]
        prefix_128_norm = prefix_128 / (np.linalg.norm(prefix_128) + 1e-8)
        cos_64_vs_128 = np.dot(t64[0], prefix_128_norm)
        ok1 = cos_64_vs_128 > 0.99
        checks["64_in_128"] = ok1
        print(f"  cos(t64, t128[:64]): {cos_64_vs_128:.6f}  {'OK' if ok1 else 'FAIL'}")

        prefix_256 = t256[0, :128]
        prefix_256_norm = prefix_256 / (np.linalg.norm(prefix_256) + 1e-8)
        cos_128_vs_256 = np.dot(t128[0], prefix_256_norm)
        ok2 = cos_128_vs_256 > 0.99
        checks["128_in_256"] = ok2
        print(f"  cos(t128, t256[:128]): {cos_128_vs_256:.6f}  {'OK' if ok2 else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_invalid_dims(self, model) -> dict:
        """Test that invalid dimensions are rejected."""
        print("\n--- INVALID DIMENSIONS ---")
        checks = {}
        emb = model.encode(["test"])

        for dim in [32, 100, 200, 512]:
            try:
                model.truncate_embeddings(emb, dim)
                checks[f"reject_{dim}"] = False
                print(f"  dim={dim}: NOT rejected  FAIL")
            except (ValueError, Exception):
                checks[f"reject_{dim}"] = True
                print(f"  dim={dim}: rejected  OK")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_cosine_similarity(self, model) -> dict:
        """Test that semantic similarity is preserved across dimensions.

        Uses real-world examples to verify the model understands meaning:
        - "the cat sat on the mat" vs "a kitten rested on the rug" (paraphrase)
        - "the dog chased the ball in the park" (different animal, different action)
        - "stock markets crashed on Monday" (completely unrelated topic)
        """
        print("\n--- COSINE SIMILARITY PRESERVATION ---")
        checks = {}

        sentences = [
            "The cat sat on the mat",               # [0] cat on mat
            "A kitten rested on the rug",            # [1] same meaning, different words
            "The dog chased the ball in the park",   # [2] different animal + action
            "A feline curled up on the carpet",      # [3] synonym paraphrase of [0]
            "Stock markets crashed on Monday",       # [4] completely unrelated
        ]
        emb = model.encode(sentences)

        for dim in [64, 128, 256, 384]:
            truncated = model.truncate_embeddings(emb, dim)
            sim_cat_kitten = np.dot(truncated[0], truncated[1])   # cat ↔ kitten (paraphrase)
            sim_cat_feline = np.dot(truncated[0], truncated[3])   # cat ↔ feline (synonym)
            sim_cat_dog = np.dot(truncated[0], truncated[2])      # cat ↔ dog (different topic)
            sim_cat_stocks = np.dot(truncated[0], truncated[4])   # cat ↔ stocks (unrelated)

            # Paraphrase should beat unrelated topic
            ok1 = sim_cat_kitten > sim_cat_stocks
            checks[f"paraphrase_vs_unrelated_{dim}"] = ok1
            print(f"  dim={dim}: cat-kitten={sim_cat_kitten:.4f} > cat-stocks={sim_cat_stocks:.4f}  "
                  f"{'OK' if ok1 else 'FAIL'}")

            # Synonym should beat different animal
            ok2 = sim_cat_feline > sim_cat_dog
            checks[f"synonym_vs_different_{dim}"] = ok2
            print(f"  dim={dim}: cat-feline={sim_cat_feline:.4f} > cat-dog={sim_cat_dog:.4f}  "
                  f"{'OK' if ok2 else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_batch(self, model) -> dict:
        """Test batch truncation and single-vector input."""
        print("\n--- BATCH & SINGLE VECTOR ---")
        checks = {}

        # Batch
        emb = model.encode(["one", "two", "three"])
        t = model.truncate_embeddings(emb, 128)
        ok1 = t.shape == (3, 128)
        checks["batch_3"] = ok1
        print(f"  batch=3, dim=128: shape={t.shape}  {'OK' if ok1 else 'FAIL'}")

        # Single (1D input should be reshaped)
        single = emb[0]  # 1D array
        t_single = model.truncate_embeddings(single, 64)
        ok2 = t_single.shape == (1, 64)
        checks["single_1d"] = ok2
        print(f"  single 1D, dim=64: shape={t_single.shape}  {'OK' if ok2 else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}


if __name__ == "__main__":
    validator = MatryoshkaValidator()
    results = validator.run()
    passed = all(r["passed"] for r in results.values())
    sys.exit(0 if passed else 1)
