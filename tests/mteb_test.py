#!/usr/bin/env python3
"""
MTEB Standard Evaluation for SAID-LAM
======================================

Demonstrates how to evaluate SAID-LAM using the official MTEB 3-step pattern:

    1. Define model
    2. Select tasks (or a benchmark)
    3. Run evaluation

Follows the MTEB API standards:
    - https://embeddings-benchmark.github.io/mteb/api/evaluation/
    - https://embeddings-benchmark.github.io/mteb/api/task/
    - https://embeddings-benchmark.github.io/mteb/api/benchmark/
    - https://embeddings-benchmark.github.io/mteb/api/model/
    - https://embeddings-benchmark.github.io/mteb/api/results/

SAID-LAM implements both MTEB protocols:
    - EncoderProtocol: encode() for STS, Classification, Clustering, etc.
    - SearchProtocol:  index() + search() for LongEmbed retrieval tasks.

BETA NOTE: LongEmbed retrieval uses an in-memory index rebuilt each run.

Usage:
    python mteb_test.py                              # Default: needle + passkey
    python mteb_test.py --tasks STS12 STS13          # Specific task names
    python mteb_test.py --task-types STS             # All STS tasks (MTEB filter)
    python mteb_test.py --task-types Retrieval STS   # Multiple task types
    python mteb_test.py --tasks LEMBNeedleRetrieval   # Single LongEmbed task
    python mteb_test.py --all-longembed              # All 6 LongEmbed tasks
    python mteb_test.py --benchmark "MTEB(eng, v2)"  # Full English benchmark
    python mteb_test.py --output-dir ./results       # Save results to disk
    python mteb_test.py --smoke                      # Smoke test: STS12/STS13 + all example_* paths
    python mteb_test.py --no-cache                   # Run without cache (all tasks from scratch)
    python mteb_test.py --smoke --no-cache          # Smoke test, no cache
    python mteb_test.py --device cpu               # Run on CPU instead of GPU

Examples Combined:
# All STS tasks (17 English STS tasks)
python mteb_test.py --task-types STS --no-cache --output-dir ./smoke_results

# Multiple types
python mteb_test.py --task-types Retrieval STS --no-cache --output-dir ./results

# Filter a benchmark by type (only STS tasks from MTEB(eng, v2))
python mteb_test.py --benchmark "MTEB(eng, v2)" --task-types STS --no-cache --output-dir ./results

# Specific task names (unchanged)
python mteb_test.py --tasks STS12 STS13 --no-cache --output-dir ./smoke_results

# GPU (default)
python mteb_test.py --tasks STS12

# CPU
python mteb_test.py --device cpu --tasks STS12

# Hugging Face token (avoids rate-limit warning; or export HF_TOKEN)
python mteb_test.py --hf-token YOUR_HF_TOKEN --tasks STS12

# Long EXAMPLE
python mteb_test.py --benchmark "MTEB(eng, v2)" --task-types STS --no-cache --hf-token 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 --output-dir ./smoke_results

python mteb_test.py --tasks MindSmallReranking --no-cache --hf-token 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 --output-dir ./resultscompare
"""

from __future__ import annotations

import os

# Pin to a single GPU for consistent device context
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Use /workspace for Hugging Face cache when available (avoids "No space left on device" on small root)
if os.path.exists("/workspace") and not os.environ.get("HF_HOME"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import mteb

# Repo root first so "said_lam" is the package; append said_lam dir for native extension (must not shadow package)
_tests_dir = Path(__file__).resolve().parent
_repo_root = _tests_dir.parent
_said_lam_dir = _repo_root / "said_lam"
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if _said_lam_dir.exists() and str(_said_lam_dir) not in sys.path:
    sys.path.append(str(_said_lam_dir))  # native .so (append so said_lam stays the package)

# ---------------------------------------------------------------------------
# Step 1: Define model
# ---------------------------------------------------------------------------
# LAM implements both EncoderProtocol and SearchProtocol (one class for all).
#
# Two ways to load:
#
#   A) Direct instantiation (recommended):
#       from said_lam import LAM
#       model = LAM("SAIDResearch/SAID-LAM-v1")
#
#   B) Via MTEB registry (once model is registered upstream):
#       model = mteb.get_model("SAIDResearch/SAID-LAM-v1")
#
# Both return an object satisfying MTEBModels (EncoderProtocol + SearchProtocol).
# ---------------------------------------------------------------------------

# LongEmbed task names (SearchProtocol — index + search)
LONGEMBED_TASKS = [
    "LEMBNeedleRetrieval",
    "LEMBPasskeyRetrieval",
    "LEMBNarrativeQARetrieval",
    "LEMBQMSumRetrieval",
    "LEMBSummScreenFDRetrieval",
    "LEMBWikimQARetrieval",
]

# Default tasks to run (fast)
DEFAULT_TASKS = ["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"]


def load_model(device: str = "cuda", output_dim: int | None = None) -> mteb.MTEBModels:
    """Load SAID-LAM as an MTEB-compatible model.

    Uses LAM (one class for all users and MTEB). For GPU, ensure
    the CUDA build is available (CUDA_VISIBLE_DEVICES=0 is set above).

    Args:
        device: "cuda" or "cpu" (use --device on CLI).
        output_dim: Optional Matryoshka dimension (e.g. 128, 256).

    Returns an object implementing both EncoderProtocol and SearchProtocol:
        - EncoderProtocol.encode()  → used for STS, Classification, etc.
        - SearchProtocol.index()   → indexes corpus for retrieval
        - SearchProtocol.search()  → returns ranked results
    """
    from said_lam import LAM

    model = LAM("SAIDResearch/SAID-LAM-v1", device=device, output_dim=output_dim)
    try:
        import lam_candle as lc
        mod_path = getattr(lc, "__file__", None)
        if mod_path:
            print(f"  extension: {mod_path}")
    except Exception:
        pass
    return model


def select_tasks(
    task_names: list[str] | None = None,
    benchmark_name: str | None = None,
    task_types: list[str] | None = None,
    all_longembed: bool = False,
) -> list[mteb.AbsTask]:
    """Select MTEB tasks to evaluate.

    Selection methods (aligned with MTEB API):

        1. By name:       mteb.get_tasks(tasks=["STS12", "STS13"])
        2. By benchmark:  mteb.get_benchmark("MTEB(eng, v2)")
        3. By task type:  mteb.get_tasks(task_types=["STS"], languages=["eng"])
        4. Or:            benchmark = mteb.get_benchmark(...); mteb.filter_tasks(benchmark.tasks, task_types=[...])

    Args:
        task_names:     Specific task names (e.g. ["STS12", "LEMBNeedleRetrieval"])
        benchmark_name: Predefined benchmark (e.g. "MTEB(eng, v2)")
        task_types:     Filter by type (e.g. ["STS"], ["Retrieval", "Reranking"]) — all matching English tasks
        all_longembed:  Shortcut for all 6 LongEmbed retrieval tasks

    Returns:
        List of AbsTask objects ready for mteb.evaluate()
    """
    if benchmark_name:
        benchmark = mteb.get_benchmark(benchmark_name)
        if task_types:
            tasks = mteb.filter_tasks(benchmark.tasks, task_types=task_types, languages=["eng"])
            return list(tasks)
        return list(benchmark.tasks)

    if task_types:
        # All tasks of given type(s), English (same as MTEB filter pattern)
        tasks = mteb.get_tasks(task_types=task_types, languages=["eng"])
        return list(tasks)

    if all_longembed:
        task_names = LONGEMBED_TASKS

    if task_names is None:
        task_names = DEFAULT_TASKS

    tasks = mteb.get_tasks(tasks=task_names, languages=["eng"])
    return list(tasks)


def _save_results_to_disk(result: mteb.ModelResult, output_dir: Path) -> None:
    """Write each TaskResult to output_dir as JSON (optional, for --output-dir)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for task_result in result.task_results:
        path = output_dir / f"{task_result.task_name}.json"
        task_result.to_disk(path)
        print(f"  Saved: {path}")


def print_results(result: mteb.ModelResult) -> None:
    """Print a summary table of evaluation results.

    Uses TaskResult.get_score() for the main metric and
    TaskResult.evaluation_time for timing.
    """
    print("\n" + "=" * 70)
    print("SAID-LAM MTEB Evaluation Results")
    print("=" * 70)
    print(f"{'Task':<35} {'Score':>10}")
    print("-" * 70)

    scores = []
    for tr in result.task_results:
        score = tr.get_score()
        scores.append(score)
        print(f"{tr.task_name:<35} {score:>10.4f}")

    if scores:
        avg = sum(scores) / len(scores)
        print("-" * 70)
        print(f"{'Average':<35} {avg:>10.4f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Quick examples (for documentation / copy-paste)
# ---------------------------------------------------------------------------

def example_basic(task_names: list[str] | None = None, device: str = "cuda"):
    """Example 1: Basic 3-step MTEB evaluation.

    Follows the official MTEB pattern:
        https://embeddings-benchmark.github.io/mteb/api/evaluation/

    Args:
        task_names: Optional list (e.g. ["STS12"]) for smoke tests; default ["LEMBNeedleRetrieval"].
        device: "cuda" or "cpu".
    """
    from said_lam import LAM

    tasks_to_use = task_names or ["LEMBNeedleRetrieval"]
    model = LAM("SAIDResearch/SAID-LAM-v1", device=device)
    tasks = mteb.get_tasks(tasks=tasks_to_use, languages=["eng"])
    results = mteb.evaluate(model=model, tasks=tasks)
    for tr in results.task_results:
        print(f"{tr.task_name}: {tr.get_score():.4f}")
    return results


def example_benchmark(task_names: list[str] | None = None, device: str = "cuda"):
    """Example 2: Run a predefined benchmark (or a task list for smoke).

    MTEB provides named benchmarks like "MTEB(eng, v2)" (41 English tasks).
    For smoke tests pass task_names e.g. ["STS12", "STS13"] to avoid full benchmark.
    """
    from said_lam import LAM

    model = LAM("SAIDResearch/SAID-LAM-v1", device=device)
    if task_names:
        tasks = mteb.get_tasks(tasks=task_names, languages=["eng"])
    else:
        benchmark = mteb.get_benchmark("MTEB(eng, v2)")
        tasks = benchmark.tasks
    results = mteb.evaluate(model=model, tasks=tasks)
    for tr in results.task_results:
        print(f"{tr.task_name}: {tr.get_score():.4f}")
    return results


def example_task_filtering(
    task_types: list[str] | None = None,
    languages: list[str] | None = None,
    domains: list[str] | None = None,
):
    """Example 3: Filter tasks by type, language, or domain.

    Only non-empty lists are passed to mteb.get_tasks() (never pass empty).

    mteb.get_tasks() supports:
        - task_types: ["Retrieval", "STS", "Classification", "Clustering", ...]
        - languages:  ["eng", "deu", "fra", ...]
        - domains:    ["Legal", "Medical", "Academic", ...]
    """
    # Only pass non-empty filters to get_tasks (never pass empty lists)
    types = task_types if task_types else None
    langs = languages if languages else None
    doms = domains if domains else None

    if types or langs or doms:
        # Call with only the filters that were provided (non-empty)
        params: dict[str, Any] = {}
        if types:
            params["task_types"] = types
        if langs:
            params["languages"] = langs
        if doms:
            params["domains"] = doms
        tasks = mteb.get_tasks(**params)
        print(f"Filter: task_types={types}, languages={langs}, domains={doms}")
    else:
        # No filters: show Retrieval and STS examples (do not pass empty lists)
        retrieval_tasks = mteb.get_tasks(task_types=["Retrieval"], languages=["eng"])
        print(f"Found {len(retrieval_tasks)} English retrieval tasks")
        sts_tasks = mteb.get_tasks(task_types=["STS"], languages=["eng"])
        print(f"Found {len(sts_tasks)} English STS tasks")
        return retrieval_tasks, sts_tasks

    print(f"Found {len(tasks)} task(s)")
    for t in tasks[:5]:
        name = getattr(getattr(t, "metadata", t), "name", str(t))
        print(f"  - {name}")
    if len(tasks) > 5:
        print(f"  ... and {len(tasks) - 5} more")
    return tasks


def example_save_results(
    task_names: list[str] | None = None,
    output_dir: Path | str | None = None,
    device: str = "cuda",
):
    """Example 4: Save results to disk.

    TaskResult.to_disk(path) saves JSON files compatible with the
    MTEB leaderboard format.

    Args:
        task_names: Optional list (e.g. ["STS12"]) for smoke tests; default ["LEMBNeedleRetrieval"].
        output_dir: Optional directory; default "./mteb_results".
        device: "cuda" or "cpu".
    """
    from said_lam import LAM

    tasks_to_use = task_names or ["LEMBNeedleRetrieval"]
    out = Path(output_dir) if output_dir else Path("./mteb_results")
    model = LAM("SAIDResearch/SAID-LAM-v1", device=device)
    tasks = mteb.get_tasks(tasks=tasks_to_use, languages=["eng"])
    results = mteb.evaluate(model=model, tasks=tasks)
    out.mkdir(parents=True, exist_ok=True)
    for tr in results.task_results:
        path = out / f"{tr.task_name}.json"
        tr.to_disk(path)
        print(f"Saved {tr.task_name} → {path}")
    return results


# ---------------------------------------------------------------------------
# Smoke test: main flow + all example_* with STS12 / STS12+STS13
# ---------------------------------------------------------------------------

def _run_smoke(
    output_dir: str | None = None,
    overwrite: str = "only-missing",
    no_cache: bool = False,
    device: str = "cuda",
) -> None:
    """Run full smoke: load_model, select_tasks, run_evaluation, print_results, then all examples."""
    smoke_tasks = ["STS12", "STS13"]
    out_dir = Path(output_dir) if output_dir else Path("./smoke_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SAID-LAM — Smoke test (STS12 + STS13, then all example_* paths)")
    print("=" * 70)
    print(f"  Device: {device}")

    # Main flow: load_model
    print("\n[Smoke] Step 1: load_model()")
    t0 = time.perf_counter()
    model = load_model(device=device)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    # Main flow: select_tasks (bulk)
    print("\n[Smoke] Step 2: select_tasks(STS12, STS13)")
    tasks = select_tasks(task_names=smoke_tasks, benchmark_name=None, all_longembed=False)
    task_names = [getattr(getattr(t, "metadata", t), "name", str(t)) for t in tasks]
    print(f"  Selected {len(tasks)} task(s): {', '.join(task_names)}")

    # Main flow: mteb.evaluate() + optional save
    print("\n[Smoke] Step 3: mteb.evaluate()")
    t0 = time.perf_counter()
    if no_cache:
        cache = None
        print("  Cache: disabled (--no-cache).")
    else:
        cache = mteb.ResultCache() if hasattr(mteb, "ResultCache") else None
        if cache is not None:
            print("  Using MTEB result cache (skip re-run for unchanged tasks).")
    result = mteb.evaluate(
        model=model,
        tasks=tasks,
        cache=cache,
        overwrite_strategy=overwrite,
        show_progress_bar=True,
    )
    _save_results_to_disk(result, out_dir)
    print(f"  Evaluation completed in {time.perf_counter() - t0:.1f}s")

    # Main flow: print_results
    print("\n[Smoke] Step 4: print_results()")
    print_results(result)

    # Example paths (each loads model internally to match doc examples)
    print("\n[Smoke] example_basic(tasks=['STS12'])")
    example_basic(task_names=["STS12"], device=device)

    print("\n[Smoke] example_benchmark(tasks=['STS12', 'STS13'])")
    example_benchmark(task_names=["STS12", "STS13"], device=device)

    print("\n[Smoke] example_task_filtering()")
    example_task_filtering()

    print("\n[Smoke] example_save_results(tasks=['STS12'], output_dir=...)")
    example_save_results(task_names=["STS12"], output_dir=out_dir, device=device)

    print("\n" + "=" * 70)
    print("Smoke test finished.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MTEB Standard Evaluation for SAID-LAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mteb_test.py                              # needle + passkey (fast)
  python mteb_test.py --tasks STS12 STS13          # specific task names
  python mteb_test.py --task-types STS             # all STS tasks (MTEB filter)
  python mteb_test.py --task-types Retrieval STS   # multiple types
  python mteb_test.py --all-longembed              # all 6 LongEmbed tasks
  python mteb_test.py --benchmark "MTEB(eng, v2)"  # full English benchmark
  python mteb_test.py --output-dir ./results       # save results as JSON
  python mteb_test.py --smoke                      # smoke: main flow + all example_* (STS12/STS13)
  python mteb_test.py --device cpu                # run on CPU
  python mteb_test.py --hf-token YOUR_TOKEN       # HF token for faster downloads (or set HF_TOKEN)

MTEB protocols used by SAID-LAM:
  EncoderProtocol  → encode()        → STS, Classification, Clustering
  SearchProtocol   → index()+search() → LongEmbed retrieval

BETA: LongEmbed index is in-memory; re-indexes every run.
        """,
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Task names to evaluate (e.g. STS12 STS13 LEMBNeedleRetrieval)",
    )
    parser.add_argument(
        "--task-types", nargs="+", default=None,
        metavar="TYPE",
        help='Filter by task type: STS, Retrieval, Classification, Clustering, Reranking, etc. (MTEB get_tasks/filter_tasks)',
    )
    parser.add_argument(
        "--benchmark", type=str, default=None,
        help='Predefined benchmark name (e.g. "MTEB(eng, v2)"); use with --task-types to filter benchmark',
    )
    parser.add_argument(
        "--all-longembed", action="store_true",
        help="Run all 6 LongEmbed retrieval tasks",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save result JSON files",
    )
    parser.add_argument(
        "--overwrite", type=str, default="only-missing",
        choices=["always", "never", "only-missing", "only-cache"],
        help="Result overwrite strategy (default: only-missing)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: main flow with STS12+STS13, then all example_* with STS12 for fast coverage",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Do not use MTEB result cache; run all tasks from scratch",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device to run the model on (default: cuda)",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="Hugging Face token (or set HF_TOKEN env var) for higher rate limits and faster downloads",
    )
    parser.add_argument(
        "--output-dim", type=int, default=None,
        help="Optional output embedding dimension for Matryoshka truncation (e.g. 384, 256, 128, 64).",
    )
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Smoke test: run main flow with STS12+STS13, then all example functions
    if args.smoke:
        _run_smoke(
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            no_cache=args.no_cache,
            device=args.device,
        )
        return

    print("=" * 70)
    print("SAID-LAM — MTEB Standard Evaluation")
    print("=" * 70)

    # Step 1: Load model
    print("\n[Step 1] Loading SAID-LAM model...")
    print(f"  Device: {args.device}")
    if args.output_dim:
        print(f"  Output dim: {args.output_dim}")
    t0 = time.perf_counter()
    model = load_model(device=args.device, output_dim=args.output_dim)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    # Step 2: Select tasks
    print("\n[Step 2] Selecting tasks...")
    tasks = select_tasks(
        task_names=args.tasks,
        benchmark_name=args.benchmark,
        task_types=args.task_types,
        all_longembed=args.all_longembed,
    )
    task_names = [getattr(t, "metadata", t).name if hasattr(getattr(t, "metadata", t), "name") else str(t) for t in tasks]
    print(f"  Selected {len(tasks)} task(s): {', '.join(task_names)}")

    # Step 3: Evaluate (mteb.evaluate() only; no wrapper)
    print(f"\n[Step 3] Running mteb.evaluate()...")
    t0 = time.perf_counter()
    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.no_cache:
        cache = None
        print("  Cache: disabled (--no-cache).")
    else:
        cache = mteb.ResultCache() if hasattr(mteb, "ResultCache") else None
        if cache is not None:
            print("  Using MTEB result cache (skip re-run for unchanged tasks).")
    result = mteb.evaluate(
        model=model,
        tasks=tasks,
        cache=cache,
        overwrite_strategy=args.overwrite,
        show_progress_bar=True,
    )
    if output_dir is not None:
        _save_results_to_disk(result, output_dir)
    total_time = time.perf_counter() - t0
    print(f"  Evaluation completed in {total_time:.1f}s")

    # Print results
    print_results(result)

    return result


if __name__ == "__main__":
    main()
