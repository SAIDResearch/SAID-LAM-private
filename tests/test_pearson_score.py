"""
Test 1: STS-B Pearson Score Validation
======================================

Validates LAM's semantic quality on the STS-B benchmark (test set).
Uses the Rust engine (lam_candle) directly — no PyTorch dependency.

Expected: Pearson >= 0.80 on STS-B test set (1379 pairs)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, bootstrap
from typing import List, Tuple

# Ensure said_lam package and lam_candle are importable.
_tests_dir = Path(__file__).resolve().parent
_repo_root = _tests_dir.parent
sys.path.insert(0, str(_repo_root / "said_lam"))  # lam_candle.so
sys.path.insert(0, str(_repo_root))                # said_lam package

# Optional visualization
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False


class PearsonScoreValidator:
    """STS-B Pearson score validation using the Rust LAM engine."""

    def __init__(self, split: str = "test"):
        self.split = split
        self.results_dir = Path(__file__).parent / "results"
        self.viz_dir = Path(__file__).parent / "visualizations"
        self.results_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)

    def load_model(self):
        """Load LAM model via Rust engine."""
        from said_lam import LAM
        model = LAM()
        return model

    def load_stsb_dataset(self) -> Tuple[List[str], List[str], List[float]]:
        """Load STS-B dataset from HuggingFace."""
        from datasets import load_dataset

        print(f"Loading STS-B {self.split} dataset...")
        dataset = load_dataset("sentence-transformers/stsb", split=self.split)

        sentences1 = list(dataset["sentence1"])
        sentences2 = list(dataset["sentence2"])
        if "label" in dataset.column_names:
            scores = list(dataset["label"])
        else:
            scores = list(dataset["score"])

        print(f"  Loaded {len(sentences1)} sentence pairs")
        return sentences1, sentences2, scores

    def run(self) -> dict:
        """Run full STS-B validation."""
        print("\n" + "=" * 60)
        print("TEST 1: STS-B Pearson Score Validation")
        print("=" * 60)

        model = self.load_model()
        sentences1, sentences2, human_scores = self.load_stsb_dataset()

        # Encode
        print("Encoding sentence1...")
        t0 = time.perf_counter()
        emb1 = model.encode(sentences1)
        t1 = time.perf_counter()
        print(f"  {emb1.shape[0]} sentences in {t1 - t0:.2f}s")

        print("Encoding sentence2...")
        emb2 = model.encode(sentences2)
        t2 = time.perf_counter()
        print(f"  {emb2.shape[0]} sentences in {t2 - t1:.2f}s")

        # Cosine similarities (embeddings are already normalized)
        similarities = np.sum(emb1 * emb2, axis=1)
        human_arr = np.array(human_scores)

        # Pearson and Spearman
        pearson_r, pearson_p = pearsonr(similarities, human_arr)
        spearman_r, spearman_p = spearmanr(similarities, human_arr)

        # Bootstrap 95% CI for Pearson
        print("Computing bootstrap confidence intervals...")
        rng = np.random.default_rng(42)
        boot_result = bootstrap(
            (similarities, human_arr),
            statistic=lambda x, y: pearsonr(x, y)[0],
            n_resamples=1000,
            paired=True,
            random_state=rng,
        )
        ci_low = boot_result.confidence_interval.low
        ci_high = boot_result.confidence_interval.high

        results = {
            "test": "pearson_score_validation",
            "split": self.split,
            "n_pairs": len(sentences1),
            "embedding_dim": int(emb1.shape[1]),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "encode_time_s": float(t2 - t0),
        }

        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Pearson:    {pearson_r:.4f}  (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
        print(f"  Spearman:   {spearman_r:.4f}")
        print(f"  Pairs:      {len(sentences1)}")
        print(f"  Embed dim:  {emb1.shape[1]}")
        print(f"  Total time: {t2 - t0:.2f}s")
        print("=" * 60)

        # Save results
        out_path = self.results_dir / "pearson_score_validation.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_path}")

        # Generate visualization (use Spearman as the headline score)
        if HAS_PLOTS:
            self._plot(similarities, human_arr, spearman_r)

        return results

    def _plot(self, similarities, human_scores, spearman_r):
        """Generate scatter plot of predicted vs human scores."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(human_scores, similarities, alpha=0.3, s=8, color="#4ECDC4")
        ax.set_xlabel("Human Score")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"LAM STS-B: Spearman r = {spearman_r:.4f}")
        ax.plot([0, 5], [0, 1], "r--", alpha=0.5, label="Perfect")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = self.viz_dir / "stsb_scatter.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Visualization saved to: {out}")


if __name__ == "__main__":
    validator = PearsonScoreValidator(split="test")
    results = validator.run()
    print(f"\nPearson: {results['pearson_r']:.4f}")
