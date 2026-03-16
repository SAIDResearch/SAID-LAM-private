"""
LAM Evaluation Suite - Master Runner
=====================================

Runs all evaluation tests for the SAID-LAM Rust engine:
1. Pearson Score Validation (STS-B benchmark)
2. Linear Scaling Validation (O(n) complexity)
3. Long Context Processing (12K/32K token limits)
4. Tier System & SCA Ablation (FREE/BETA tiers, recall accuracy)
5. Crystalline MTEB End-to-End (index_mteb/search_mteb pipeline)
6. User API (said_lam.py — encode, index, search, tiers)
7. MTEB (LAM)
8. Matryoshka Truncated Embeddings
9. Embedding Model Parity (sentence-transformers drop-in)

Usage:
    python run_all_tests.py
    python run_all_tests.py --skip-pearson   # Skip STS-B (needs dataset download)
    python run_all_tests.py --skip-sts       # Skip STS + run Crystalline MTEB
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Ensure said_lam package and lam_candle are importable.
_tests_dir = Path(__file__).resolve().parent
_repo_root = _tests_dir.parent
sys.path.insert(0, str(_repo_root / "said_lam"))  # lam_candle.so
sys.path.insert(0, str(_repo_root))                # said_lam package


def run_test(test_number: int, test_name: str, test_func):
    """Run a single test with error handling."""
    print(f"\n{'=' * 70}")
    print(f"  TEST {test_number}: {test_name}")
    print(f"{'=' * 70}")

    t0 = time.perf_counter()
    try:
        result = test_func()
        elapsed = time.perf_counter() - t0
        return {
            "test": test_name,
            "status": "passed",
            "duration_s": round(elapsed, 2),
            "result": result,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "test": test_name,
            "status": "failed",
            "duration_s": round(elapsed, 2),
            "error": str(e),
        }


def main():
    skip_pearson = "--skip-pearson" in sys.argv or "--skip-sts" in sys.argv
    skip_sts = "--skip-sts" in sys.argv

    test_count = 9
    if skip_pearson:
        test_count -= 1
    skips = []
    if skip_pearson:
        skips.append("Pearson/STS")

    print("\n" + "=" * 70)
    print("  SAID-LAM EVALUATION SUITE (Rust Engine)")
    print("=" * 70)
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Engine:   lam_candle (Rust/Candle)")
    print(f"  Tests:    {test_count}{' (skipping: ' + ', '.join(skips) + ')' if skips else ''}")
    print("=" * 70)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    all_results = []
    t_start = time.perf_counter()

    # Test 1: Pearson Score
    if not skip_pearson:
        from test_pearson_score import PearsonScoreValidator
        r = run_test(1, "STS-B Pearson Score", lambda: PearsonScoreValidator().run())
        all_results.append(r)
    else:
        print("\n  Skipping Test 1 (Pearson Score) — use without --skip-pearson to include")

    # Test 2: Linear Scaling
    from test_linear_scaling import LinearScalingValidator
    r = run_test(2, "Linear Scaling (O(n))", lambda: LinearScalingValidator().run())
    all_results.append(r)

    # Test 3: Long Context
    from test_long_context import LongContextValidator
    r = run_test(3, "Long Context Processing", lambda: LongContextValidator().run())
    all_results.append(r)

    # Test 4: Tier & SCA Ablation
    from test_ablation_study import AblationStudyValidator
    r = run_test(4, "Tier System & SCA Ablation", lambda: AblationStudyValidator().run())
    all_results.append(r)

    # Test 5: Crystalline MTEB End-to-End (index_mteb/search_mteb pipeline)
    from test_crystalline_mteb import CrystallineMTEBValidator
    r = run_test(5, "Crystalline MTEB End-to-End", lambda: CrystallineMTEBValidator().run())
    all_results.append(r)

    # Test 6: User API (said_lam.py — encode, index, search, tiers)
    from test_said_lam import test_user_api
    r = run_test(6, "User API (encode/index/search)", test_user_api)
    all_results.append(r)

    # Test 7: MTEB (LAM)
    from test_said_lam import test_mteb_wrapper
    r = run_test(7, "MTEB (LAM)", test_mteb_wrapper)
    all_results.append(r)

    # Test 8: Matryoshka Truncated Embeddings
    from test_matryoshka import MatryoshkaValidator
    r = run_test(8, "Matryoshka Truncated Embeddings", lambda: MatryoshkaValidator().run())
    all_results.append(r)

    # Test 9: Embedding Model Parity (sentence-transformers drop-in)
    try:
        from test_embeddings import EmbeddingModelValidator
        r = run_test(9, "Embedding Model Parity", lambda: EmbeddingModelValidator().run())
        all_results.append(r)
    except ImportError:
        print("\n  Skipping Test 9 (test_embeddings.py not found)")

    total_time = time.perf_counter() - t_start

    # Summary
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in all_results if r["status"] == "passed")
    failed = sum(1 for r in all_results if r["status"] == "failed")

    for r in all_results:
        icon = "PASS" if r["status"] == "passed" else "FAIL"
        print(f"  [{icon}] {r['test']} ({r['duration_s']:.1f}s)")

    print(f"\n  Total: {passed} passed, {failed} failed ({total_time:.1f}s)")
    print("=" * 70)

    # Save comprehensive report
    report = {
        "suite": "SAID-LAM Evaluation (Rust Engine)",
        "timestamp": datetime.now().isoformat(),
        "total_duration_s": round(total_time, 2),
        "passed": passed,
        "failed": failed,
        "tests": all_results,
    }
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'item'):
                return obj.item()
            try:
                return bool(obj)
            except (TypeError, ValueError):
                return super().default(obj)

    report_path = results_dir / "comprehensive_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, cls=_Encoder)
    print(f"\n  Report saved to: {report_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
