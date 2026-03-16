"""
Test 2: Linear Scaling Validation
==================================

Proves LAM has O(n) time and memory complexity.
Uses the Rust engine (lam_candle) directly — no PyTorch dependency.

Expected: Linear R² > 0.90 for time scaling
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Ensure said_lam package and lam_candle are importable.
_tests_dir = Path(__file__).resolve().parent
_repo_root = _tests_dir.parent
sys.path.insert(0, str(_repo_root / "said_lam"))  # lam_candle.so
sys.path.insert(0, str(_repo_root))                # said_lam package

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False


class LinearScalingValidator:
    """Validates O(n) scaling for LAM Rust engine."""

    def __init__(self):
        self.results_dir = Path(__file__).parent / "results"
        self.viz_dir = Path(__file__).parent / "visualizations"
        self.results_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)

    def load_model(self):
        """Load LAM model via Rust engine."""
        from said_lam import LAM
        model = LAM()
        return model

    def _generate_text(self, n_words: int) -> str:
        """Generate text of approximately n_words."""
        base = "The quick brown fox jumps over the lazy dog in the sunny park. "
        repeats = max(1, n_words // len(base.split()))
        return base * repeats

    def run(self) -> dict:
        """Run linear scaling validation."""
        print("\n" + "=" * 60)
        print("TEST 2: Linear Scaling Validation")
        print("=" * 60)

        model = self.load_model()

        # Test sequence lengths (in approximate word counts)
        # Extended to show scaling out to very long documents
        lengths = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
        timing_data = []
        memory_data = []

        # Warmup
        print("Warming up...")
        _ = model.encode(["warmup sentence for model initialization"])

        print("\nRunning scaling tests:")
        print(f"{'Words':>8} | {'Time (s)':>10} | {'Memory (MB)':>12}")
        print("-" * 40)

        for n_words in lengths:
            text = self._generate_text(n_words)
            actual_words = len(text.split())

            if HAS_PSUTIL:
                proc = psutil.Process()
                mem_before = proc.memory_info().rss / (1024 * 1024)

            # Time encoding (average of 3 runs for stability)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                emb = model.encode([text])
                t1 = time.perf_counter()
                times.append(t1 - t0)

            avg_time = np.mean(times)

            if HAS_PSUTIL:
                mem_after = proc.memory_info().rss / (1024 * 1024)
                mem_used = mem_after - mem_before
            else:
                mem_used = 0.0

            timing_data.append((actual_words, avg_time))
            memory_data.append((actual_words, mem_used))
            print(f"{actual_words:>8} | {avg_time:>10.4f} | {mem_used:>10.1f} MB")

        # Compute R² for linear fit
        words = np.array([t[0] for t in timing_data])
        times = np.array([t[1] for t in timing_data])

        # Linear fit: time = a * words + b
        if len(words) >= 3:
            coeffs = np.polyfit(words, times, 1)
            predicted = np.polyval(coeffs, words)
            ss_res = np.sum((times - predicted) ** 2)
            ss_tot = np.sum((times - np.mean(times)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            r_squared = 0.0
            coeffs = [0, 0]

        # Compute speedup ratio (time per word should be roughly constant)
        time_per_word = times / words
        consistency = 1.0 - (np.std(time_per_word) / np.mean(time_per_word)) if np.mean(time_per_word) > 0 else 0.0

        results = {
            "test": "linear_scaling_validation",
            "lengths_tested": lengths,
            "timing_data": [{"words": int(w), "time_s": float(t)} for w, t in timing_data],
            "memory_data": [{"words": int(w), "memory_mb": float(m)} for w, m in memory_data],
            "linear_r_squared": float(r_squared),
            "slope_ms_per_word": float(coeffs[0] * 1000),
            "intercept_s": float(coeffs[1]),
            "time_per_word_consistency": float(consistency),
        }

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Linear R²:         {r_squared:.4f}")
        print(f"  Slope:             {coeffs[0] * 1000:.4f} ms/word")
        print(f"  Consistency:       {consistency:.4f}")
        print(f"  Verdict:           {'O(n) CONFIRMED' if r_squared > 0.90 else 'NEEDS REVIEW'}")
        print("=" * 60)

        # Save results
        out_path = self.results_dir / "linear_scaling_validation.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_path}")

        # Generate visualization
        if HAS_PLOTS:
            self._plot(timing_data, r_squared)

        return results

    def _plot(self, timing_data, r_squared):
        """Generate scaling plot."""
        words = [t[0] for t in timing_data]
        times = [t[1] for t in timing_data]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(words, times, "o-", color="#4ECDC4", markersize=8, label="LAM (measured)")

        # Linear fit line
        coeffs = np.polyfit(words, times, 1)
        x_fit = np.linspace(min(words), max(words), 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="#FF6B6B", alpha=0.7,
                label=f"Linear fit (R²={r_squared:.3f})")

        ax.set_xlabel("Document Length (words)")
        ax.set_ylabel("Encoding Time (seconds)")
        ax.set_title(f"LAM Scaling: O(n) Linear Complexity (R²={r_squared:.3f})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = self.viz_dir / "scaling_plot.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Visualization saved to: {out}")


if __name__ == "__main__":
    validator = LinearScalingValidator()
    results = validator.run()
