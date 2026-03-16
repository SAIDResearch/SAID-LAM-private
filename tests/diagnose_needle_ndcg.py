#!/usr/bin/env python3
"""
LEMBNeedleRetrieval NDCG@1 diagnostic — aligned with MTEB retrieval scoring.

MTEB scoring (confirmed):
  1. For each split (test_256 .. test_32768), RetrievalEvaluator runs:
     search_model.index(corpus); results = search_model.search(queries).
  2. Per-split scores: pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_1"}).evaluate(results).
     → Returns per-query NDCG@1; per-split main_score = mean of those NDCG@1 values.
  3. TaskResult.get_score() = mean of the 8 per-split main_scores (one number per split).
     So the official MTEB score is NOT "mean over all 400 queries" but "mean of 8 split means".

This script:
  - Without --pytrec: "Overall NDCG@1" = mean over all queries (simple top-1-in-qrels 0/1).
  - With --pytrec: Uses pytrec_eval (same as MTEB); prints per-split mean NDCG@1 and
    "MTEB get_score() style (mean of 8 split means, pytrec_eval)" which matches
    what mteb.evaluate() returns for this task.
  - Pipeline: same index_mteb + search_mteb, same qrels; results format qid -> {doc_id: score}.

Usage (from said-lam root):
    PYTHONPATH=. python tests/diagnose_passkey_ndcg.py
    PYTHONPATH=. python tests/diagnose_passkey_ndcg.py --pytrec   # use pytrec_eval like MTEB
    PYTHONPATH=. python tests/diagnose_passkey_ndcg.py --split test_32768
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo path for imports
_tests_dir = Path(__file__).resolve().parent
_repo_root = _tests_dir.parent
sys.path.insert(0, str(_repo_root / "said_lam"))
sys.path.insert(0, str(_repo_root))

# After path setup
import mteb
from said_lam import LAM

# Reuse compare_backends helpers for splits and parsing
from compare_backends import (
    _get_task_splits,
    _parse_corpus_dataset,
    _parse_queries_dataset,
)


CONTEXT_LENGTHS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def run_split(
    engine,
    task_name: str,
    corpus_ids: list,
    corpus_texts: list,
    qids: list,
    q_texts: list,
    qrels: dict,
    top_k: int = 10,
) -> tuple[list[bool], list[float], list[str | None], dict[str, dict[str, float]]]:
    """Index, search; return hits, NDCG@1 list, top-1 per query, and results dict for pytrec_eval."""
    engine.index_mteb(corpus_ids, corpus_texts, task_name, None)
    raw = engine.search_mteb(qids, q_texts, task_name, top_k, None)

    hits = []
    ndcg_at_1 = []
    top1_per_q = []
    # Build results in MTEB format: qid -> {doc_id: score}
    results = {}
    for i, qid in enumerate(qids):
        qid_str = str(qid)
        doc_scores = raw.get(qid, raw.get(qid_str, {}))
        results[qid_str] = {str(d): float(s) for d, s in doc_scores.items()}
        if not doc_scores:
            hits.append(False)
            ndcg_at_1.append(0.0)
            top1_per_q.append(None)
            continue
        ranked = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
        top1 = ranked[0] if ranked else None
        top1_per_q.append(str(top1) if top1 else None)
        rel = qrels.get(qid, qrels.get(qid_str, {}))
        expected_ids = set(str(d) for d in rel.keys()) if rel else set()
        hit = (str(top1) in expected_ids) if top1 else False
        hits.append(hit)
        ndcg_at_1.append(1.0 if hit else 0.0)
    return hits, ndcg_at_1, top1_per_q, results


def qrels_to_pytrec(qrels, qids: list) -> dict[str, dict[str, int]]:
    """Convert task qrels to pytrec_eval format: {query_id: {doc_id: relevance_int}}."""
    out = {}
    for qid in qids:
        qid_str = str(qid)
        rel = qrels.get(qid, qrels.get(qid_str, {}))
        out[qid_str] = {}
        for d in rel:
            v = rel[d]
            if v is None:
                out[qid_str][str(d)] = 1
            elif isinstance(v, (int, float)):
                out[qid_str][str(d)] = int(v) if int(v) > 0 else 1
            else:
                out[qid_str][str(d)] = 1
    return out


def ndcg_at_1_via_pytrec(results: dict, qrels: dict) -> tuple[list[float], float]:
    """Compute per-query NDCG@1 and mean using pytrec_eval (same as MTEB). Returns (per_query_ndcg, mean)."""
    try:
        import pytrec_eval
    except ImportError:
        return [], 0.0
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_1"})
    scores = evaluator.evaluate(results)
    per_query = [scores.get(q, {}).get("ndcg_cut_1", 0.0) for q in results]
    mean_ndcg = sum(per_query) / len(per_query) if per_query else 0.0
    return per_query, mean_ndcg


def main():
    parser = argparse.ArgumentParser(description="Diagnose LEMBNeedleRetrieval NDCG@1 per split")
    parser.add_argument("--split", type=str, default=None,
                        help="Run only this split (e.g. test_32768)")
    parser.add_argument("--max-split", type=str, default=None,
                        help="Run splits up to this (e.g. test_4096) to save time")
    parser.add_argument("--verbose", action="store_true", help="Print per-query top-1 and expected")
    parser.add_argument("--pytrec", action="store_true",
                        help="Score with pytrec_eval (same as mteb.evaluate()) and show per-query NDCG@1")
    parser.add_argument("--inspect", action="store_true",
                        help="For each miss, print query text and snippets of expected vs top-1 document")
    args = parser.parse_args()

    print("Loading LEMBNeedleRetrieval task...")
    tasks = list(mteb.get_tasks(tasks=["LEMBNeedleRetrieval"], languages=["eng"]))
    if not tasks:
        print("ERROR: No LEMBNeedleRetrieval task found")
        sys.exit(1)
    task = tasks[0]
    task.load_data()

    splits = _get_task_splits(task)
    context_splits = {
        k: v for k, v in splits.items()
        if k.startswith("test_") and k.split("_")[1].isdigit()
    }
    if not context_splits:
        print("ERROR: No test_NNN splits found")
        sys.exit(1)

    # Filter splits if requested
    if args.split:
        context_splits = {k: v for k, v in context_splits.items() if k == args.split}
        if not context_splits:
            print(f"ERROR: Split {args.split} not found. Available: {list(splits.keys())}")
            sys.exit(1)
    if args.max_split:
        max_ctx = int(args.max_split.replace("test_", "")) if args.max_split.startswith("test_") else 0
        context_splits = {k: v for k, v in context_splits.items()
                         if int(k.replace("test_", "")) <= max_ctx}

    print("Loading SAID-LAM model...")
    model = LAM("SAIDResearch/SAID-LAM-v1")
    engine = getattr(model, "_engine", None)
    if engine is None:
        print("ERROR: No engine on model")
        sys.exit(1)
    if hasattr(engine, "auto_activate_mteb"):
        engine.auto_activate_mteb()
    task_name = "lembneedleretrieval"

    all_ndcg = []
    all_misses = []  # (split_name, query_id, top1_doc, expected_docs)
    split_means_pytrec = []  # per-split mean NDCG@1 (MTEB-style)

    for split_name in sorted(context_splits.keys(), key=lambda x: int(x.replace("test_", ""))):
        split_data = context_splits[split_name]
        corpus = split_data.get("corpus")
        queries = split_data.get("queries")
        qrels = split_data.get("relevant_docs", {})

        if corpus is None or queries is None:
            print(f"  {split_name}: no corpus/queries — SKIP")
            continue

        corpus_ids, corpus_texts = _parse_corpus_dataset(corpus)
        qids, q_texts = _parse_queries_dataset(queries)
        id_to_text = dict(zip(corpus_ids, corpus_texts))  # for --inspect

        expected_per_q = []
        for qid in qids:
            e = qrels.get(qid, qrels.get(str(qid), {}))
            expected_per_q.append([str(d) for d in e.keys()])

        print(f"\n  {split_name}: {len(corpus_ids)} docs, {len(qids)} queries ... ", end="", flush=True)
        hits, ndcg_at_1, top1_per_q, results_dict = run_split(
            engine, task_name,
            corpus_ids, corpus_texts,
            qids, q_texts,
            qrels,
            top_k=10,
        )
        correct = sum(hits)
        total = len(hits)
        avg_ndcg = sum(ndcg_at_1) / total if total else 0.0
        all_ndcg.extend(ndcg_at_1)

        print(f"{correct}/{total} correct, NDCG@1 = {avg_ndcg:.4f}")

        if args.pytrec and results_dict and qids:
            qrels_pytrec = qrels_to_pytrec(qrels, qids)
            per_q, mean_ndcg = ndcg_at_1_via_pytrec(results_dict, qrels_pytrec)
            split_means_pytrec.append(mean_ndcg)
            if per_q:
                miss_qids = [q for q, v in zip(results_dict.keys(), per_q) if v < 1.0]
                if miss_qids:
                    print(f"    pytrec_eval: mean NDCG@1 = {mean_ndcg:.5f}  MISSING: {miss_qids}")
                else:
                    print(f"    pytrec_eval: mean NDCG@1 = {mean_ndcg:.5f}  (all 1.0)")

        for i, qid in enumerate(qids):
            if hits[i]:
                continue
            top1 = top1_per_q[i]
            expected = expected_per_q[i]
            if args.inspect:
                q_text = q_texts[i] if i < len(q_texts) else ""
                top1_text = (id_to_text.get(top1) or id_to_text.get(str(top1)) or "")[:600] if top1 else ""
                exp_id = expected[0] if expected else None
                exp_text = (id_to_text.get(exp_id) or id_to_text.get(str(exp_id)) or "")[:600] if exp_id else ""
                all_misses.append((split_name, qid, top1, expected, q_text, top1_text, exp_text))
            else:
                all_misses.append((split_name, qid, top1, expected))
            if args.verbose:
                print(f"    MISS: split={split_name} qid={qid} top1={top1} expected={expected}")

    # Overall (same as MTEB aggregate)
    overall_ndcg = sum(all_ndcg) / len(all_ndcg) if all_ndcg else 0.0
    total_q = len(all_ndcg)
    total_correct = sum(1 for x in all_ndcg if x == 1.0)

    print("\n" + "=" * 60)
    print("SUMMARY (MTEB-style aggregate)")
    print("=" * 60)
    print(f"  Total queries: {total_q}")
    print(f"  Total correct (top-1 in qrels): {total_correct}/{total_q}")
    print(f"  Overall NDCG@1 (mean over all queries): {overall_ndcg:.4f}")

    if args.pytrec and split_means_pytrec:
        # MTEB TaskResult.get_score() = mean of per-split main_scores (one per split)
        mteb_style = sum(split_means_pytrec) / len(split_means_pytrec)
        print(f"  MTEB get_score() style (mean of {len(split_means_pytrec)} split means, pytrec_eval): {mteb_style:.4f}")

    if all_misses:
        print(f"\n  MISSES ({len(all_misses)} queries causing score < 100%):")
        for miss in all_misses:
            split_name, qid, top1, expected = miss[:4]
            print(f"    {split_name}  qid={qid}  top1={top1}  expected={expected[:3]}{'...' if len(expected) > 3 else ''}")
        if args.inspect and all_misses and len(all_misses[0]) >= 7:
            print("\n  --- INSPECT: query and document snippets for each miss ---")
            for miss in all_misses:
                split_name, qid, top1, expected, q_text, top1_text, exp_text = miss[:7]
                print(f"\n  [{split_name}] qid={qid}")
                print(f"  QUERY ({len(q_text)} chars):")
                print("    " + (q_text[:500] + "..." if len(q_text) > 500 else q_text).replace("\n", " "))
                print(f"  EXPECTED doc ({expected[0] if expected else 'n/a'}) snippet:")
                print("    " + (exp_text[:500] + "..." if len(exp_text) > 500 else exp_text).replace("\n", " "))
                print(f"  TOP-1 (wrong) doc ({top1}) snippet:")
                print("    " + (top1_text[:500] + "..." if len(top1_text) > 500 else top1_text).replace("\n", " "))
    else:
        print("\n  No misses — all queries have top-1 in qrels. If mteb.evaluate() still reports < 1.0,")
        print("  the difference may be in how MTEB computes NDCG (e.g. relevance grades or tie-breaking).")

    print("=" * 60)
    return 0 if not all_misses else 1


if __name__ == "__main__":
    sys.exit(main())
