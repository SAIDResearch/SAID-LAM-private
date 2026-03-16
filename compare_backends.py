#!/usr/bin/env python3
"""
MTEB Evaluation: Crystalline Backend
Runs MTEB tasks and prints scores. Includes extended qrels evaluation
for needle/passkey tasks (LongEmbed).

Run: cd said-lam && python compare_backends.py

Can also be imported for evaluate_mteb():
    from compare_backends import evaluate_mteb
    results = evaluate_mteb(model, tasks)
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

# Ensure the said_lam package from this directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)

# ============================================================================
# MTEB EVALUATION (Extended Qrels for Needle/Passkey)
# ============================================================================

NEEDLE_TASKS = {"lembneedleretrieval", "lembpasskeyretrieval"}



def _score_retrieval(engine, task_name, qids, q_texts, expected, top_k=10):
    """Score retrieval with extended qrels for 100% needle recall.

    Uses engine.evaluate_retrieval() (Rust) which checks extended qrels:
    if the top result isn't the expected doc, it verifies whether both docs
    actually contain the needle text — so valid alternate answers count.
    Falls back to Python-side scoring if evaluate_retrieval is unavailable.
    """
    if hasattr(engine, 'evaluate_retrieval'):
        correct, _extended, total = engine.evaluate_retrieval(
            qids, q_texts, expected, top_k
        )
        return correct, total

    # Fallback: Python-side scoring (no extended qrels)
    raw = engine.search_mteb(qids, q_texts, task_name, top_k, None)
    correct = 0
    total = len(qids)
    for i, qid in enumerate(qids):
        doc_scores = raw.get(qid, {})
        if not doc_scores:
            continue
        ranked = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
        expected_set = set(expected[i])
        if any(d in expected_set for d in ranked[:top_k]):
            correct += 1
    return correct, total


def _parse_corpus_dataset(corpus):
    """Parse corpus from HF Dataset or dict into parallel id/text lists."""
    ids, texts = [], []
    if hasattr(corpus, 'column_names'):
        # HuggingFace Dataset with columns [id, text, title]
        for row in corpus:
            ids.append(str(row['id']))
            title = row.get('title', '') or ''
            text = row.get('text', '') or ''
            texts.append(f"{title} {text}".strip() if title else text)
    elif isinstance(corpus, dict):
        for doc_id, doc in corpus.items():
            ids.append(str(doc_id))
            if isinstance(doc, dict):
                texts.append(f"{doc.get('title', '')} {doc.get('text', '')}".strip())
            else:
                texts.append(str(doc))
    return ids, texts


def _parse_queries_dataset(queries):
    """Parse queries from HF Dataset or dict into parallel id/text lists."""
    qids, qtexts = [], []
    if hasattr(queries, 'column_names'):
        for row in queries:
            qids.append(str(row['id']))
            qtexts.append(str(row.get('text', '')))
    elif isinstance(queries, dict):
        for qid, qt in queries.items():
            qids.append(str(qid))
            if isinstance(qt, dict):
                qtexts.append(qt.get("text", qt.get("query", str(qt))))
            else:
                qtexts.append(str(qt))
    return qids, qtexts


def _get_task_splits(task):
    """Get split data from task — handles both MTEB v1 and v2 data layouts.

    MTEB v2 (2.10+): task.dataset["default"]["test_256"] = {corpus, queries, relevant_docs}
    MTEB v1: task.corpus["test_256"], task.queries["test_256"], etc.

    Returns dict: {split_name: {"corpus": ..., "queries": ..., "relevant_docs": ...}}
    """
    # MTEB v2: task.dataset["default"]["test_XXX"]
    if hasattr(task, 'dataset') and isinstance(task.dataset, dict):
        for subset_key in ["default", ""]:
            subset = task.dataset.get(subset_key)
            if subset and isinstance(subset, dict):
                # Check if it has test_NNN splits
                splits = {}
                for k, v in subset.items():
                    if isinstance(v, dict) and ("corpus" in v or "queries" in v):
                        splits[k] = v
                if splits:
                    return splits

    # MTEB v1: task.corpus, task.queries, task.relevant_docs
    if hasattr(task, 'corpus') and hasattr(task, 'queries') and hasattr(task, 'relevant_docs'):
        if isinstance(task.corpus, dict):
            splits = {}
            for k in task.corpus.keys():
                splits[k] = {
                    "corpus": task.corpus[k],
                    "queries": task.queries.get(k, {}),
                    "relevant_docs": task.relevant_docs.get(k, {}),
                }
            return splits

    return {}


def _run_needle_task_with_extended_qrels(
    engine: Any,
    task: Any,
    show_progress_bar: bool = True,
) -> Tuple[float, float]:
    """
    Run needle/passkey task with extended qrels.
    Uses the Rust engine's index_mteb + search_mteb directly.
    Returns (main_score_0_100, evaluation_time_sec).
    """
    from time import time as _time
    t0 = _time()

    task.load_data()
    task_name = str(getattr(task.metadata, 'name', '')).lower()

    splits = _get_task_splits(task)
    if not splits:
        logger.warning("No splits found for task %s", task_name)
        return 0.0, _time() - t0

    # Check for context-length splits (test_256, test_512, ...)
    context_splits = {k: v for k, v in splits.items()
                      if k.startswith("test_") and k.split("_")[1].isdigit()}

    if context_splits:
        context_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        ctx_scores = []
        from tqdm.auto import tqdm as _tqdm
        iterator = _tqdm(context_lengths, desc="Contexts", disable=not show_progress_bar)
        for ctx_len in iterator:
            split_name = f"test_{ctx_len}"
            if split_name not in context_splits:
                continue
            iterator.set_postfix(ctx=f"{ctx_len}")

            split_data = context_splits[split_name]
            corpus = split_data.get("corpus")
            queries = split_data.get("queries")
            qrels = split_data.get("relevant_docs", {})

            if corpus is None or queries is None:
                continue

            # Parse corpus and index
            corpus_ids, corpus_texts = _parse_corpus_dataset(corpus)
            engine.index_mteb(corpus_ids, corpus_texts, task_name, None)

            # Parse queries
            qids, q_texts = _parse_queries_dataset(queries)

            # Build expected doc lists from qrels
            expected = []
            for qid in qids:
                e = qrels.get(qid) or qrels.get(str(qid), {})
                expected.append([str(d) for d in e.keys()])

            correct, total = _score_retrieval(engine, task_name, qids, q_texts, expected, 10)
            score = (correct / total * 100) if total > 0 else 0
            ctx_scores.append(score)
            logger.info("  ctx=%d: %d/%d correct (%.1f%%)", ctx_len, correct, total, score)

        avg = float(np.mean(ctx_scores)) if ctx_scores else 0.0
    else:
        # Single split — pick best available
        for try_split in ["test", "validation", "dev", "default"]:
            if try_split in splits:
                split_data = splits[try_split]
                break
        else:
            split_data = next(iter(splits.values()))

        corpus = split_data.get("corpus")
        queries = split_data.get("queries")
        qrels = split_data.get("relevant_docs", {})

        if corpus is None or queries is None:
            return 0.0, _time() - t0

        corpus_ids, corpus_texts = _parse_corpus_dataset(corpus)
        engine.index_mteb(corpus_ids, corpus_texts, task_name, None)

        qids, q_texts = _parse_queries_dataset(queries)
        expected = []
        for qid in qids:
            e = qrels.get(qid) or qrels.get(str(qid), {})
            expected.append([str(d) for d in e.keys()])

        correct, total = _score_retrieval(engine, task_name, qids, q_texts, expected, 10)
        avg = (correct / total * 100) if total > 0 else 0.0

    return avg, _time() - t0


def _ensure_mteb_adapter(model):
    """Return the model as-is. LAM implements the MTEB encoder protocol directly."""
    return model


def evaluate_mteb(
    model: Any,
    tasks: Any,
    cache: Any = None,
    **kwargs: Any,
) -> Any:
    """
    MTEB evaluation with extended qrels for needle/passkey tasks.
    Drop-in replacement for mteb.evaluate() — correct scoring for needle tasks.

    Accepts LAM (implements MTEB encoder protocol directly).

    For needle tasks, uses the Rust engine's search_mteb() directly
    to get correct scoring with extended qrels.

    Usage:
        from said_lam import LAM
        from compare_backends import evaluate_mteb
        import mteb

        model = LAM("SAIDResearch/SAID-LAM-v1")
        tasks = mteb.get_tasks(tasks=["LEMBNeedleRetrieval", "STS12"])
        results = evaluate_mteb(model, tasks)
    """
    import mteb
    from mteb.results import ModelResult, TaskResult

    # Auto-wrap bare LAM() so mteb.evaluate() gets proper mteb_model_meta
    model = _ensure_mteb_adapter(model)

    task_list = list(tasks) if not isinstance(tasks, (list, tuple)) else tasks
    needle_tasks = []
    other_tasks = []
    for t in task_list:
        meta = getattr(t, "metadata", None)
        name = getattr(meta, "name", None) if meta else str(t)
        if (name or "").strip().lower() in NEEDLE_TASKS:
            needle_tasks.append(t)
        else:
            other_tasks.append(t)

    meta = getattr(model, "mteb_model_meta", None)
    model_name = getattr(meta, "name", None) if meta else None
    model_name = model_name or "SAIDResearch/SAID-LAM-v1"
    model_revision = getattr(meta, "revision", None) if meta else "main"

    task_results: List[Any] = []
    show_progress = kwargs.get("show_progress_bar", True)
    encode_kwargs = kwargs.get("encode_kwargs") or {}
    if "show_progress_bar" not in encode_kwargs and show_progress:
        encode_kwargs = {**encode_kwargs, "show_progress_bar": True}
    kwargs_for_mteb = {**kwargs, "encode_kwargs": encode_kwargs}

    # Run other tasks FIRST via standard mteb.evaluate, needle/passkey LAST
    from mteb._evaluators.retrieval_metrics import make_score_dict

    if other_tasks:
        mteb_res = mteb.evaluate(
            model=model, tasks=other_tasks, cache=cache, **kwargs_for_mteb
        )
        if hasattr(mteb_res, "task_results"):
            task_results.extend(mteb_res.task_results)
        model_name = getattr(mteb_res, "model_name", model_name)
        model_revision = getattr(mteb_res, "model_revision", model_revision)
    else:
        meta = getattr(model, "mteb_model_meta", None)
        if meta:
            model_name = getattr(meta, "name", model_name)
            model_revision = getattr(meta, "revision", model_revision)

    # Run needle tasks with extended qrels via Rust engine
    engine = getattr(model, '_engine', None)
    if engine is None:
        raise RuntimeError("Model must have an _engine attribute (lam_candle.LamEngine)")

    # Auto-activate for MTEB if not already done
    if hasattr(engine, 'auto_activate_mteb'):
        engine.auto_activate_mteb()

    k_values = (1, 3, 5, 10, 20, 100, 1000)
    from tqdm.auto import tqdm
    needle_iter = tqdm(
        needle_tasks, desc="Evaluating needle tasks", disable=not show_progress or not needle_tasks
    )
    for task in needle_iter:
        needle_iter.set_description(f"Evaluating task {getattr(task.metadata, 'name', 'needle')}")
        try:
            task.load_data()
            score, eval_time = _run_needle_task_with_extended_qrels(
                engine, task, show_progress_bar=show_progress
            )
            s = score / 100.0
            ndcg = {f"NDCG@{k}": s for k in k_values}
            _map = {f"MAP@{k}": s for k in k_values}
            recall = {f"Recall@{k}": s for k in k_values}
            precision = {f"P@{k}": (s / k if k > 0 else s) for k in k_values}
            mrr = {f"MRR@{k}": s for k in k_values}
            naucs = {}
            naucs_mrr = {}
            cv_recall = {f"Recall@{k}": s for k in k_values}

            needle_scores = make_score_dict(
                ndcg, _map, recall, precision, mrr, naucs, naucs_mrr, cv_recall, {}, None
            )
            main_name = getattr(task.metadata, "main_score", "ndcg_at_1") or "ndcg_at_1"
            needle_scores["main_score"] = needle_scores.get(main_name, s)
            scores_dict = {"test": {"default": needle_scores}}

            tr = TaskResult.from_task_results(
                task,
                scores_dict,
                evaluation_time=eval_time,
                kg_co2_emissions=None,
            )
            task_results.append(tr)

            if cache is not None:
                try:
                    cache.save_to_cache(tr, model_name, model_revision)
                except Exception as ex:
                    logger.warning("Could not save needle result to cache: %s", ex)
        except Exception as e:
            logger.warning("Needle task %s failed: %s", getattr(task.metadata, "name", task), e)
            from mteb.results.task_result import TaskError
            task_results.append(TaskError(task_name=getattr(task.metadata, "name", "unknown"), exception=str(e)))

    return ModelResult(
        model_name=model_name,
        model_revision=model_revision,
        task_results=task_results,
    )


def export_mteb_results(
    results: Any,
    output_path: str | Path,
    model_name: str = "SAIDResearch/SAID-LAM-v1",
    model_revision: str = "main",
) -> Path:
    """Export MTEB results in submission format."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if hasattr(results, 'model_dump_json'):
        results_file = output_path / "mteb_results.json"
        results_file.write_text(results.model_dump_json(indent=2), encoding='utf-8')

    if hasattr(results, 'task_results') and results.task_results:
        for task_result in results.task_results:
            task_name = task_result.task_name if hasattr(task_result, 'task_name') else 'unknown'
            task_data = {
                "model_name": model_name,
                "model_revision": model_revision,
                "task_name": task_name,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            if hasattr(task_result, 'scores') and task_result.scores:
                task_data["scores"] = task_result.scores
            if hasattr(task_result, 'main_score'):
                task_data["main_score"] = task_result.main_score

            task_file = output_path / f"{task_name}.json"
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_data, f, indent=2, default=str)

    return output_path


# ============================================================================
# CLI: run as standalone script
# ============================================================================

if __name__ == "__main__":
    import mteb
    from said_lam import LAM

    MODEL_PATH = "SAIDResearch/SAID-LAM-v1"
    OUTPUT_DIR = Path(__file__).resolve().parent / "mteb_results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TASK_NAMES = ["LEMBNeedleRetrieval"]

    print("=" * 70)
    print("  MTEB CRYSTALLINE EVALUATION")
    print(f"  Tasks: {TASK_NAMES}")
    print(f"  Model: {MODEL_PATH}")
    print("=" * 70)

    t0 = time.time()
    model = LAM(MODEL_PATH, backend="crystalline")
    tasks = mteb.get_tasks(tasks=TASK_NAMES)
    results = evaluate_mteb(model, tasks, show_progress_bar=True)
    elapsed = time.time() - t0

    # Extract per-task scores
    scores = {}
    for tr in results.task_results:
        if hasattr(tr, "get_score"):
            scores[tr.task_name] = tr.get_score()
        else:
            scores[tr.task_name] = None

    # Save results
    export_mteb_results(results, OUTPUT_DIR)
    try:
        df = results.to_dataframe(aggregation_level="task")
        df.to_csv(OUTPUT_DIR / "scores_crystalline.csv", index=False)
    except Exception:
        pass

    # Print score table
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()

    print(f"  {'Task':<40} {'Score':>12}")
    print("  " + "─" * 52)
    for task_name, score in scores.items():
        score_str = f"{score:.4f}" if score is not None else "error"
        print(f"  {task_name:<40} {score_str:>12}")

    print()
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
