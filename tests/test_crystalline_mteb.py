#!/usr/bin/env python3
"""
Test 5: Crystalline Backend — MTEB End-to-End
==============================================

Validates the Crystalline index/search pipeline via real MTEB evaluation.

Three test modes:
1. Engine direct:   index_mteb + search_mteb on LamEngine (unit)
2. Crystalline:     index_mteb + search_mteb with MTEB-style corpus dicts (unit)
3. MTEB evaluate:   Full mteb.evaluate() (integration)

Usage (standalone):
    cd said-lam/tests && python test_crystalline_mteb.py

Usage (MTEB evaluation with score output):
    python test_crystalline_mteb.py --mteb

Usage (as part of suite):
    python run_all_tests.py --skip-pearson
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path

# Ensure said_lam package (LAM in __init__.py) and lam_candle are importable.
# Order: repo_root first so "said_lam" is the package; then said_lam dir for lam_candle.so
_tests_dir = Path(__file__).resolve().parent
_repo_root = _tests_dir.parent
sys.path.insert(0, str(_repo_root / "said_lam"))  # lam_candle.so
sys.path.insert(0, str(_repo_root))                # said_lam package


class CrystallineMTEBValidator:
    """Validates the Crystalline MTEB pipeline end-to-end."""

    TASK_NAMES = ["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"]

    def __init__(self, run_mteb: bool = False, task_names: list = None, task_types: list = None):
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.run_mteb = run_mteb
        self.task_types = task_types
        if task_names:
            self.TASK_NAMES = task_names

    def run(self) -> dict:
        """Run all Crystalline MTEB tests."""
        print("\n" + "=" * 70)
        print("TEST 5: Crystalline Backend — MTEB End-to-End")
        print("=" * 70)

        results = {
            "test": "crystalline_mteb",
            "engine_direct": self._test_engine_direct(),
            "lam_protocol": self._test_lam_protocol(),
            "evaluate_mteb": self._test_evaluate_mteb(),
        }

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        all_passed = True
        for key in ["engine_direct", "lam_protocol", "evaluate_mteb"]:
            status = results[key].get("passed", False)
            skipped = results[key].get("skipped", False)
            label = "SKIP" if skipped else ("PASS" if status else "FAIL")
            print(f"  {key:<25} {label}")
            if not status and not skipped:
                all_passed = False

        print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print("=" * 70)

        # Save
        out_path = self.results_dir / "crystalline_mteb_validation.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)
        print(f"\n  Results saved to: {out_path}")

        return results

    # ─────────────────────────────────────────────────────────────────────
    # Test 1: Engine direct — index_mteb + search_mteb on LamEngine
    # ─────────────────────────────────────────────────────────────────────
    def _test_engine_direct(self) -> dict:
        """Test index_mteb/search_mteb directly on LamEngine (Crystalline)."""
        print("\n--- ENGINE DIRECT: index_mteb + search_mteb ---")
        from said_lam import LAM
        checks = {}

        model = LAM(backend="crystalline")
        engine = model._engine
        engine.auto_activate_mteb()

        # Build a small corpus
        corpus_ids = ["doc_needle", "doc_haystack_1", "doc_haystack_2", "doc_haystack_3"]
        corpus_texts = [
            "The secret activation code is QUANTUM7DELTA hidden in this document.",
            "Machine learning algorithms process datasets to discover hidden patterns.",
            "The Eiffel Tower is one of the most visited landmarks in Paris France.",
            "Quantum computing utilizes superposition and entanglement for computation.",
        ]

        # Index
        t0 = time.perf_counter()
        engine.index_mteb(corpus_ids, corpus_texts, "lembneedleretrieval", None)
        index_time = time.perf_counter() - t0
        checks["index_ok"] = True  # No exception
        print(f"  index_mteb: {len(corpus_ids)} docs in {index_time:.3f}s  OK")

        # Search
        qids = ["q1", "q2"]
        qtexts = ["QUANTUM7DELTA", "Eiffel Tower Paris"]
        t0 = time.perf_counter()
        raw = engine.search_mteb(qids, qtexts, "lembneedleretrieval", 10, None)
        search_time = time.perf_counter() - t0

        checks["search_returns_dict"] = isinstance(raw, dict)
        checks["search_has_all_queries"] = set(raw.keys()) == {"q1", "q2"}
        print(f"  search_mteb: {len(qids)} queries in {search_time:.3f}s  OK")

        # Validate q1 top result is the needle doc
        q1_scores = raw.get("q1", {})
        if q1_scores:
            top_doc = max(q1_scores, key=q1_scores.get)
            checks["needle_found"] = top_doc == "doc_needle"
            print(f"  q1 top doc: {top_doc} (expected: doc_needle)  "
                  f"{'OK' if checks['needle_found'] else 'FAIL'}")
        else:
            checks["needle_found"] = False
            print(f"  q1: no results  FAIL")

        # Validate q2 top result is Eiffel doc
        q2_scores = raw.get("q2", {})
        if q2_scores:
            top_doc2 = max(q2_scores, key=q2_scores.get)
            checks["eiffel_found"] = top_doc2 == "doc_haystack_2"
            print(f"  q2 top doc: {top_doc2} (expected: doc_haystack_2)  "
                  f"{'OK' if checks['eiffel_found'] else 'FAIL'}")
        else:
            checks["eiffel_found"] = False
            print(f"  q2: no results  FAIL")

        # Verify scores are floats in [0, 1] or reasonable range
        all_scores = [s for qscores in raw.values() for s in qscores.values()]
        checks["scores_are_float"] = all(isinstance(s, float) for s in all_scores)
        print(f"  Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]  "
              f"{'OK' if checks['scores_are_float'] else 'FAIL'}")

        passed = all(checks.values())
        return {"passed": passed, "checks": checks, "index_time": index_time, "search_time": search_time}

    # ─────────────────────────────────────────────────────────────────────
    # Test 2: Engine index_mteb + search_mteb (Crystalline pipeline)
    # ─────────────────────────────────────────────────────────────────────
    def _test_lam_protocol(self) -> dict:
        """Test Crystalline index_mteb/search_mteb with MTEB-style corpus dicts."""
        print("\n--- CRYSTALLINE PIPELINE: index_mteb + search_mteb ---")
        checks = {}

        from said_lam import LAM
        model = LAM(backend="crystalline")
        engine = model._engine
        engine.auto_activate_mteb()

        # Build mock corpus (dict format, like MTEB provides)
        corpus = {
            "c0": {"title": "Needle Document", "text": "The secret code is ALPHA9BRAVO embedded deep in this text."},
            "c1": {"title": "Science", "text": "Neural networks learn hierarchical representations from data."},
            "c2": {"title": "History", "text": "The Roman Empire expanded across Europe and North Africa."},
            "c3": {"title": "Cooking", "text": "Mediterranean cuisine uses olive oil and fresh herbs extensively."},
        }

        # Parse corpus dict into parallel id/text lists (same as MTEB wrapper does)
        corpus_ids = []
        corpus_texts = []
        for doc_id, doc in corpus.items():
            corpus_ids.append(str(doc_id))
            corpus_texts.append(f"{doc.get('title', '')} {doc.get('text', '')}".strip())

        # Index via engine
        task_name = "lembneedleretrieval"
        t0 = time.perf_counter()
        engine.index_mteb(corpus_ids, corpus_texts, task_name, None)
        index_time = time.perf_counter() - t0
        checks["index_no_error"] = True
        print(f"  index_mteb(): {len(corpus_ids)} docs in {index_time:.3f}s  OK")

        # Search via engine
        qids = ["q0", "q1"]
        qtexts = ["ALPHA9BRAVO", "Roman Empire history"]

        t0 = time.perf_counter()
        results = engine.search_mteb(qids, qtexts, task_name, 10, None)
        search_time = time.perf_counter() - t0

        checks["search_returns_dict"] = isinstance(results, dict)
        checks["search_has_all_queries"] = set(results.keys()) == {"q0", "q1"}
        print(f"  search_mteb(): {len(qids)} queries in {search_time:.3f}s  OK")

        # Validate results structure: {qid: {doc_id: score}}
        for qid, doc_scores in results.items():
            checks[f"{qid}_is_dict"] = isinstance(doc_scores, dict)
            for did, score in doc_scores.items():
                checks[f"{qid}_{did}_score_float"] = isinstance(score, float)
                break  # just check first

        # Check needle retrieval
        q0_scores = results.get("q0", {})
        if q0_scores:
            top_doc = max(q0_scores, key=q0_scores.get)
            checks["needle_top1"] = top_doc == "c0"
            print(f"  q0 'ALPHA9BRAVO' top: {top_doc} (expected: c0)  "
                  f"{'OK' if checks['needle_top1'] else 'FAIL'}")
        else:
            checks["needle_top1"] = False
            print(f"  q0: no results  FAIL")

        # Check history retrieval
        q1_scores = results.get("q1", {})
        if q1_scores:
            top_doc = max(q1_scores, key=q1_scores.get)
            checks["history_top1"] = top_doc == "c2"
            print(f"  q1 'Roman Empire' top: {top_doc} (expected: c2)  "
                  f"{'OK' if checks['history_top1'] else 'FAIL'}")
        else:
            checks["history_top1"] = False
            print(f"  q1: no results  FAIL")

        passed = all(checks.values())
        return {"passed": passed, "checks": checks, "index_time": index_time, "search_time": search_time}

    # ─────────────────────────────────────────────────────────────────────
    # Test 3: Full MTEB evaluation via mteb.evaluate()
    # ─────────────────────────────────────────────────────────────────────
    def _test_evaluate_mteb(self) -> dict:
        """
        Run full MTEB evaluation via mteb.evaluate() (official MTEB API), not
        the custom compare_backends.evaluate_mteb().

        Same path as:
            import mteb
            from said_lam import LAM
            model = LAM("SAIDResearch/SAID-LAM-v1")
            tasks = mteb.get_tasks(tasks=["LEMBNeedleRetrieval"])
            results = mteb.evaluate(model, tasks)
        """
        print("\n--- MTEB EVALUATE: Full pipeline ---")
        checks = {}

        try:
            import mteb
        except ImportError:
            if self.run_mteb:
                print("  mteb not installed — FAIL (--mteb requires mteb)")
                return {"passed": False, "error": "mteb not installed"}
            print("  mteb not installed — SKIP")
            return {"passed": True, "skipped": True}

        if not self.run_mteb:
            print("  Skipping full MTEB (use --mteb to enable)")
            return {"passed": True, "skipped": True}

        from said_lam import LAM
        from compare_backends import export_mteb_results

        # Output directory for MTEB results
        output_dir = self.results_dir / "mteb_crystalline"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Tasks:  {self.TASK_NAMES}")
        print(f"  Output: {output_dir}")

        # Create model — LAM implements MTEB protocol (has mteb_model_meta)
        model = LAM("SAIDResearch/SAID-LAM-v1")

        # Load tasks — always filter to English only (LAM is English-only)
        if self.task_types:
            tasks = mteb.get_tasks(task_types=self.task_types, languages=["eng"])
        else:
            tasks = mteb.get_tasks(tasks=self.TASK_NAMES, languages=["eng"])
        task_list = list(tasks)
        checks["tasks_loaded"] = len(task_list) > 0
        print(f"  Loaded {len(task_list)} tasks  OK")

        # Run MTEB evaluation via mteb.evaluate() (official API)
        t0 = time.perf_counter()
        scores = {}
        try:
            if self.task_types:
                tasks = mteb.get_tasks(task_types=self.task_types, languages=["eng"])
            else:
                tasks = mteb.get_tasks(tasks=self.TASK_NAMES, languages=["eng"])
            bench_results = mteb.evaluate(model, tasks, show_progress_bar=True)
            elapsed = time.perf_counter() - t0
            checks["evaluate_no_error"] = True
            print(f"  mteb.evaluate(): completed in {elapsed:.1f}s  OK")

            # Extract and display scores
            for tr in bench_results.task_results:
                if hasattr(tr, "get_score"):
                    scores[tr.task_name] = tr.get_score()
                else:
                    scores[tr.task_name] = None

            # Print score table
            print()
            print(f"  {'Task':<35} {'Score':>12}")
            print(f"  {'─' * 47}")
            for task_name, score in scores.items():
                score_str = f"{score:.4f}" if score is not None else "error"
                ok = score is not None and score > 0
                checks[f"score_{task_name}"] = ok
                print(f"  {task_name:<35} {score_str:>12}  {'OK' if ok else 'FAIL'}")
            print()

            # Save summary CSV
            try:
                df = bench_results.to_dataframe(aggregation_level="task")
                csv_path = output_dir / "scores_crystalline.csv"
                df.to_csv(csv_path, index=False)
                print(f"  Scores saved to: {csv_path}")
            except Exception:
                pass  # dataframe export is optional

            # Save full results
            try:
                export_mteb_results(bench_results, output_dir)
            except Exception:
                pass

            print(f"  Results saved to: {output_dir}")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            checks["evaluate_no_error"] = False
            print(f"  mteb.evaluate() failed after {elapsed:.1f}s: {e}  FAIL")

        passed = all(checks.values())
        return {"passed": passed, "checks": checks, "scores": scores if 'scores' in dir() else {}}

def _json_default(obj):
    """JSON encoder for numpy types."""
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        return bool(obj)
    except (TypeError, ValueError):
        return str(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crystalline MTEB Validation")
    parser.add_argument("--mteb", action="store_true",
                        help="Run full MTEB evaluation (requires mteb package)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="MTEB task names (default: LEMBNeedleRetrieval LEMBPasskeyRetrieval)")
    parser.add_argument("--task", dest="task_single", metavar="NAME",
                        help="Single task name (alias for --tasks NAME)")
    parser.add_argument("--task-types", nargs="+", default=None,
                        help="MTEB task types e.g. STS Retrieval Classification (English only)")
    args = parser.parse_args()

    task_names = args.tasks
    if getattr(args, "task_single", None) is not None:
        task_names = [args.task_single] if task_names is None else [args.task_single] + task_names

    validator = CrystallineMTEBValidator(
        run_mteb=args.mteb,
        task_names=task_names,
        task_types=args.task_types,
    )
    results = validator.run()
