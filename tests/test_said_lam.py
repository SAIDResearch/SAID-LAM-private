#!/usr/bin/env python3
"""
Test: said_lam.py — MTEB Wrapper + Normal User API
====================================================

Two test modes:
1. USER API:  LAM class — encode(), index(), search(), tiers, auto_activate
2. MTEB: LAM — encode(), index(), search() via MTEB protocols

Usage:
    cd said-lam/tests && python test_said_lam.py              # Both modes
    cd said-lam/tests && python test_said_lam.py --user-only  # User API only
    cd said-lam/tests && python test_said_lam.py --mteb-only  # MTEB wrapper only
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


def _sep(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Normal User API — encode, index, recall, tiers
# ═══════════════════════════════════════════════════════════════════════

def test_user_api() -> dict:
    """Test the pip-install user experience: from said_lam import LAM"""
    _sep("USER API: from said_lam import LAM")
    from said_lam import LAM, TIER_FREE, TIER_BETA
    checks = {}

    # ── 1. Basic construction ──
    print("  [1] LAM() construction...")
    model = LAM(backend="crystalline")
    checks["construct"] = model is not None
    print(f"      tier={model.tier}, max_tokens={model.max_tokens:,}  OK")

    # ── 2. Encode single + batch ──
    print("  [2] encode()...")
    emb_single = model.encode(["Hello world"])
    checks["encode_shape_1"] = emb_single.shape == (1, 384)
    print(f"      single: shape={emb_single.shape}  OK")

    sentences = [
        "The cat sat on the mat",         # [0] cat on mat
        "A kitten rested on the rug",     # [1] paraphrase of [0]
        "Stock markets crashed on Monday", # [2] completely unrelated
    ]
    emb_batch = model.encode(sentences)
    checks["encode_shape_3"] = emb_batch.shape == (3, 384)
    print(f"      batch:  shape={emb_batch.shape}  OK")

    # ── 3. Embeddings are L2-normalized ──
    norms = np.linalg.norm(emb_batch, axis=1)
    checks["l2_normalized"] = np.allclose(norms, 1.0, atol=0.01)
    print(f"      L2 norms: {norms.round(4).tolist()}  "
          f"{'OK' if checks['l2_normalized'] else 'FAIL'}")

    # ── 4. Cosine similarity — paraphrase vs unrelated ──
    sim_self = float(emb_batch[0] @ emb_batch[0])
    sim_paraphrase = float(emb_batch[0] @ emb_batch[1])  # cat ↔ kitten
    sim_unrelated = float(emb_batch[0] @ emb_batch[2])   # cat ↔ stocks
    checks["sim_self"] = sim_self > 0.99
    checks["sim_paraphrase_beats_unrelated"] = sim_paraphrase > sim_unrelated
    print(f"      self-sim={sim_self:.4f}")
    print(f"      cat↔kitten={sim_paraphrase:.4f} > cat↔stocks={sim_unrelated:.4f}  "
          f"{'OK' if checks['sim_paraphrase_beats_unrelated'] else 'FAIL'}")

    # ── 5. Encode empty list ──
    emb_empty = model.encode([])
    checks["encode_empty"] = emb_empty.shape == (0, 384)
    print(f"      empty: shape={emb_empty.shape}  OK")

    # ── 6. Encode single string (auto-wraps) ──
    emb_str = model.encode("just a string")
    checks["encode_str"] = emb_str.shape == (1, 384)
    print(f"      string: shape={emb_str.shape}  OK")

    # ── 7. Tier constants ──
    print("  [3] Tier constants...")
    checks["tier_free"] = TIER_FREE is not None
    checks["tier_beta"] = TIER_BETA is not None
    print(f"      TIER_FREE={TIER_FREE}, TIER_BETA={TIER_BETA}  OK")

    # ── 8. auto_activate_mteb ──
    print("  [4] auto_activate_mteb()...")
    result = model.auto_activate_mteb()
    checks["auto_activate"] = result is not None
    print(f"      result={result}, tier after={model.tier}  OK")

    # ── 9. Index + search (requires BETA+) ──
    print("  [5] index() + search()...")
    model.clear()
    docs = [
        ("doc_speed", "The speed of light is approximately 299792458 meters per second"),
        ("doc_python", "Python was created by Guido van Rossum in 1991"),
        ("doc_paris", "The Eiffel Tower is located in Paris France"),
        ("doc_needle", "The secret activation code is QUANTUM7DELTA embedded here"),
    ]
    for doc_id, text in docs:
        model.index(doc_id, text)
    checks["doc_count"] = len(model) == 4
    print(f"      indexed {len(model)} docs  OK")

    # Search — needle
    results = model.search("QUANTUM7DELTA", top_k=5)
    if results:
        top_id, top_score = results[0]
        checks["search_needle"] = top_id == "doc_needle"
        print(f"      search('QUANTUM7DELTA') top: {top_id} ({top_score:.4f})  "
              f"{'OK' if checks['search_needle'] else 'FAIL'}")
    else:
        checks["search_needle"] = False
        print(f"      search('QUANTUM7DELTA'): no results  FAIL")

    # Search — semantic
    results = model.search("speed of light", top_k=5)
    if results:
        top_id, top_score = results[0]
        checks["search_semantic"] = top_id == "doc_speed"
        print(f"      search('speed of light') top: {top_id} ({top_score:.4f})  "
              f"{'OK' if checks['search_semantic'] else 'FAIL'}")
    else:
        checks["search_semantic"] = False
        print(f"      search('speed of light'): no results  FAIL")

    # ── 9b. Backward-compat aliases (encode_doc / recall) ──
    print("  [5b] Backward-compat: encode_doc() / recall()...")
    checks["alias_encode_doc"] = callable(model.encode_doc) and type(model).encode_doc is type(model).index
    checks["alias_recall"] = callable(model.recall) and type(model).recall is type(model).search
    print(f"      encode_doc is index: {checks['alias_encode_doc']}, "
          f"recall is search: {checks['alias_recall']}  OK")

    # ── 10. Truncate embeddings (Matryoshka) ──
    print("  [6] truncate_embeddings()...")
    emb = model.encode(["test sentence"])
    for dim in [256, 128, 64]:
        trunc = model.truncate_embeddings(emb, target_dim=dim)
        checks[f"truncate_{dim}"] = trunc.shape == (1, dim)
    print(f"      256={checks['truncate_256']}, 128={checks['truncate_128']}, "
          f"64={checks['truncate_64']}  OK")

    # ── 11. Stats ──
    print("  [7] stats()...")
    stats = model.stats()
    checks["stats_dict"] = isinstance(stats, dict)
    print(f"      keys={list(stats.keys())[:5]}...  OK")

    # ── 12. Clear ──
    print("  [8] clear()...")
    model.clear()
    checks["clear"] = len(model) == 0
    print(f"      doc_count after clear: {len(model)}  OK")

    # ── 13. repr ──
    print("  [9] repr()...")
    r = repr(model)
    checks["repr"] = "LAM(" in r
    print(f"      {r}  OK")

    # Summary
    passed = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    print(f"\n  User API: {'ALL PASSED' if passed else f'FAILED: {failed}'} "
          f"({sum(checks.values())}/{len(checks)})")
    return {"passed": passed, "checks": checks}


# ═══════════════════════════════════════════════════════════════════════
# PART 2: MTEB — LAM encode/index/search
# ═══════════════════════════════════════════════════════════════════════

def test_mteb_wrapper() -> dict:
    """Test LAM with mock DataLoader/TaskMetadata (MTEB protocol)."""
    _sep("MTEB: LAM")
    checks = {}

    # Check mteb is installed
    try:
        import mteb
        from mteb.abstasks.task_metadata import TaskMetadata
    except ImportError:
        print("  mteb not installed — SKIP (pip install mteb)")
        return {"passed": True, "skipped": True}

    # Import the wrapper (same file that goes to mteb/models/model_implementations/)
    from said_lam import LAM
    from said_lam.said_lam import said_lam_v1

    # ── 1. Construction ──
    print("  [1] LAM() construction...")
    encoder = LAM("SAIDResearch/SAID-LAM-v1")
    checks["construct"] = encoder is not None
    checks["has_meta"] = encoder.mteb_model_meta is not None
    print(f"      model_meta.name={encoder.mteb_model_meta.name}  OK")

    # ── 2. ModelMeta fields ──
    print("  [2] ModelMeta fields...")
    meta = said_lam_v1
    checks["meta_name"] = meta.name == "SAIDResearch/SAID-LAM-v1"
    checks["meta_dim"] = meta.embed_dim == 384
    checks["meta_params"] = meta.n_parameters == 23_848_788
    checks["meta_license"] = meta.license == "apache-2.0"
    checks["meta_loader"] = meta.loader is not None
    print(f"      name={meta.name}, dim={meta.embed_dim}, params={meta.n_parameters:,}  OK")

    # ── 3. Encode via MTEB protocol ──
    print("  [3] encode() via MTEB DataLoader...")

    # Build a mock DataLoader (list of batch dicts, same as MTEB sends)
    mock_batches = [
        {"text": ["Hello world", "Machine learning"]},
        {"text": ["Semantic search is powerful"]},
    ]

    # Build a mock TaskMetadata
    class MockTaskMeta:
        name = "STSBenchmark"

    embs = encoder.encode(
        mock_batches,
        task_metadata=MockTaskMeta(),
        hf_split="test",
        hf_subset="default",
    )
    checks["encode_shape"] = embs.shape == (3, 384)
    checks["encode_dtype"] = embs.dtype == np.float32
    norms = np.linalg.norm(embs, axis=1)
    checks["encode_normalized"] = np.allclose(norms, 1.0, atol=0.01)
    print(f"      shape={embs.shape}, dtype={embs.dtype}, "
          f"norms={norms.round(4).tolist()}  OK")

    # ── 4. Encode empty ──
    print("  [4] encode() empty input...")
    embs_empty = encoder.encode(
        [],
        task_metadata=MockTaskMeta(),
        hf_split="test",
        hf_subset="default",
    )
    checks["encode_empty"] = embs_empty.shape == (0, 384)
    print(f"      shape={embs_empty.shape}  OK")

    # ── 5. Index via MTEB protocol ──
    print("  [5] index() via MTEB corpus...")

    # Build mock corpus (list of dicts with id, title, text — as MTEB sends)
    mock_corpus = [
        {"id": "d0", "title": "Needle", "text": "The activation code is QUANTUM7DELTA."},
        {"id": "d1", "title": "Science", "text": "Neural networks learn from data."},
        {"id": "d2", "title": "Geography", "text": "The Eiffel Tower is in Paris France."},
        {"id": "d3", "title": "History", "text": "The Roman Empire expanded across Europe."},
    ]

    class MockRetrievalMeta:
        name = "LEMBNeedleRetrieval"

    t0 = time.perf_counter()
    encoder.index(
        mock_corpus,
        task_metadata=MockRetrievalMeta(),
        hf_split="test",
        hf_subset="default",
    )
    index_time = time.perf_counter() - t0
    checks["index_ok"] = True  # no exception
    print(f"      indexed {len(mock_corpus)} docs in {index_time:.3f}s  OK")

    # ── 6. Search via MTEB protocol ──
    print("  [6] search() via MTEB queries...")

    # Build mock queries dataset (needs "id" and "text" columns)
    # LAM.search() (MTEB protocol) expects queries["id"] and uses _create_text_queries_dataloader
    # For unit testing, we mock the queries as a dict-like with "id" key
    # and patch the dataloader

    # Direct engine search (bypasses the _create_text_queries_dataloader which needs HF Dataset)
    engine = encoder._engine
    qids = ["q0", "q1"]
    qtexts = ["QUANTUM7DELTA", "Eiffel Tower Paris"]
    t0 = time.perf_counter()
    raw = engine.search_mteb(qids, qtexts, "LEMBNeedleRetrieval", 10, None)
    search_time = time.perf_counter() - t0

    checks["search_returns_dict"] = isinstance(raw, dict)
    checks["search_has_queries"] = set(raw.keys()) == {"q0", "q1"}
    print(f"      {len(qids)} queries in {search_time:.3f}s  OK")

    # Needle found?
    q0 = raw.get("q0", {})
    if q0:
        top_doc = max(q0, key=q0.get)
        checks["needle_found"] = top_doc == "d0"
        print(f"      q0 'QUANTUM7DELTA' top: {top_doc} ({q0[top_doc]:.4f})  "
              f"{'OK' if checks['needle_found'] else 'FAIL'}")
    else:
        checks["needle_found"] = False
        print("      q0: no results  FAIL")

    # Eiffel found?
    q1 = raw.get("q1", {})
    if q1:
        top_doc = max(q1, key=q1.get)
        checks["eiffel_found"] = top_doc == "d2"
        print(f"      q1 'Eiffel Tower' top: {top_doc} ({q1[top_doc]:.4f})  "
              f"{'OK' if checks['eiffel_found'] else 'FAIL'}")
    else:
        checks["eiffel_found"] = False
        print("      q1: no results  FAIL")

    # ── 7. IP audit — check no proprietary terms leaked ──
    print("  [7] IP audit on said_lam.py...")
    wrapper_path = Path(__file__).parent.parent / "said_lam" / "said_lam.py"
    if wrapper_path.exists():
        content = wrapper_path.read_text()
        ip_terms = [
            "DeltaNet", "Crystalline", "LONGEMBED", "BETA_2025",
            "IDF-Surprise", "6 layers", "12 heads", "Bilinear",
            "Resonance", "SCA", "Hierarchical", "Dual-State",
            "NIAH", "/workspace",
        ]
        leaked = [t for t in ip_terms if t in content]
        checks["ip_clean"] = len(leaked) == 0
        if leaked:
            print(f"      LEAKED TERMS: {leaked}  FAIL")
        else:
            print(f"      0 proprietary terms found  OK")
    else:
        checks["ip_clean"] = True
        print(f"      wrapper not found at {wrapper_path} — SKIP")

    # Summary
    passed = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    print(f"\n  MTEB Wrapper: {'ALL PASSED' if passed else f'FAILED: {failed}'} "
          f"({sum(checks.values())}/{len(checks)})")
    return {"passed": passed, "checks": checks}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def _json_default(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        return bool(obj)
    except (TypeError, ValueError):
        return str(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAID-LAM Wrapper Test Suite")
    parser.add_argument("--user-only", action="store_true", help="Run user API tests only")
    parser.add_argument("--mteb-only", action="store_true", help="Run MTEB wrapper tests only")
    args = parser.parse_args()

    print("=" * 60)
    print("  SAID-LAM Test Suite: said_lam.py")
    print("=" * 60)

    results = {}
    all_passed = True

    if not args.mteb_only:
        results["user_api"] = test_user_api()
        if not results["user_api"]["passed"]:
            all_passed = False

    if not args.user_only:
        results["mteb_wrapper"] = test_mteb_wrapper()
        if not results["mteb_wrapper"].get("passed", True):
            all_passed = False

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        skipped = r.get("skipped", False)
        passed = r.get("passed", False)
        label = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        print(f"    {name:<20} {label}")
    print(f"\n    Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "test_said_lam.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\n  Results saved to: {out_path}")

    sys.exit(0 if all_passed else 1)
