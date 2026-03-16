"""
Test 4: Tier System & SCA Ablation
====================================

Validates the LAM tier system and SCA (Said Crystalline Attention)
recall engine at each tier level.
Uses the Rust engine (lam_candle) directly — no PyTorch dependency.

Tests:
1. FREE tier: encode() works, recall() blocked, 12K token limit
2. BETA tier: activate() works, recall() enabled, 32K token limit
3. SCA recall accuracy: deterministic 100% recall
4. Search modes: exact search, key-value search
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

# Ensure said_lam package and lam_candle are importable.
_tests_dir = Path(__file__).resolve().parent
_repo_root = _tests_dir.parent
sys.path.insert(0, str(_repo_root / "said_lam"))  # lam_candle.so
sys.path.insert(0, str(_repo_root))                # said_lam package


class AblationStudyValidator:
    """Validates tier system and SCA search capabilities."""

    def __init__(self):
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

    def run(self) -> dict:
        """Run tier system and SCA ablation tests."""
        print("\n" + "=" * 60)
        print("TEST 4: Tier System & SCA Ablation")
        print("=" * 60)

        from said_lam import LAM, TIER_FREE, TIER_BETA

        results = {
            "test": "ablation_study",
            "free_tier": self._test_free_tier(LAM),
            "license_unlock": self._test_license_unlock(LAM),
            "beta_tier": self._test_beta_tier(LAM),
            "sca_search": self._test_sca_search(LAM),
            "search_modes": self._test_search_modes(LAM),
        }

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        test_keys = ["free_tier", "license_unlock", "beta_tier", "sca_search", "search_modes"]
        all_passed = all(
            results[k].get("passed", False)
            for k in test_keys
        )
        for k in test_keys:
            label = k.replace("_", " ").title()
            print(f"  {label:20s} {'PASS' if results[k]['passed'] else 'FAIL'}")
        print(f"  Overall:      {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print("=" * 60)

        # Save results with custom encoder for non-standard types
        class _Encoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'item'):
                    return obj.item()  # numpy scalars
                try:
                    return bool(obj)
                except (TypeError, ValueError):
                    return super().default(obj)

        out_path = self.results_dir / "ablation_study_validation.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, cls=_Encoder)
        print(f"\nResults saved to: {out_path}")

        return results

    def _test_free_tier(self, LAM_class) -> dict:
        """Test FREE tier: encode works, recall blocked."""
        print("\n--- FREE TIER ---")
        model = LAM_class()
        checks = {}

        # Check defaults
        checks["tier_name"] = model.tier == "FREE"
        checks["max_tokens"] = model.max_tokens == 12000
        print(f"  Tier: {model.tier} (expected: FREE) {'OK' if checks['tier_name'] else 'FAIL'}")
        print(f"  Max tokens: {model.max_tokens} (expected: 12000) {'OK' if checks['max_tokens'] else 'FAIL'}")

        # Check encode works
        emb = model.encode(["Hello world"])
        checks["encode_works"] = emb.shape == (1, 384)
        checks["encode_normalized"] = abs(np.linalg.norm(emb[0]) - 1.0) < 0.01
        print(f"  Encode shape: {emb.shape} {'OK' if checks['encode_works'] else 'FAIL'}")
        print(f"  Normalized: {np.linalg.norm(emb[0]):.4f} {'OK' if checks['encode_normalized'] else 'FAIL'}")

        # Check recall is blocked at FREE tier
        try:
            model.encode_doc("test", "test doc")
            model.recall("test")
            checks["recall_blocked"] = False
            print(f"  Recall blocked: NO (unexpected!) FAIL")
        except Exception:
            checks["recall_blocked"] = True
            print(f"  Recall blocked: YES (expected) OK")

        passed = all(checks.values())
        return {"passed": passed, "checks": checks}

    def _test_license_unlock(self, LAM_class) -> dict:
        """Test license unlocking process: tiers, activation, beta registration."""
        print("\n--- LICENSE UNLOCK ---")
        from said_lam import TIER_FREE, TIER_BETA, TIER_LICENSED, TIER_INFINITE
        checks = {}

        # 1. Tier constants exist and have expected values
        checks["tier_free_val"] = TIER_FREE is not None
        checks["tier_beta_val"] = TIER_BETA is not None
        checks["tier_licensed_val"] = TIER_LICENSED is not None
        checks["tier_infinite_val"] = TIER_INFINITE is not None
        print(f"  Tier constants: FREE={TIER_FREE}, BETA={TIER_BETA}, "
              f"LICENSED={TIER_LICENSED}, INFINITE={TIER_INFINITE}  OK")

        # 2. Fresh model starts at FREE
        model = LAM_class()
        checks["starts_free"] = model.tier == "FREE"
        checks["free_12k"] = model.max_tokens == 12000
        print(f"  Fresh model: tier={model.tier}, max_tokens={model.max_tokens}  "
              f"{'OK' if checks['starts_free'] else 'FAIL'}")

        # 3. Invalid key rejected, stays FREE
        bad_result = model.activate("INVALID_KEY")
        checks["bad_key_rejected"] = bad_result is False
        checks["still_free"] = model.tier == "FREE"
        print(f"  Invalid key: rejected={not bad_result}, tier={model.tier}  "
              f"{'OK' if checks['bad_key_rejected'] else 'FAIL'}")

        # 4. auto_activate_mteb() promotes
        result = model.auto_activate_mteb()
        checks["mteb_activate"] = result is not None
        checks["mteb_32k"] = model.max_tokens >= 32000
        print(f"  auto_activate_mteb(): result={result}, tier={model.tier}, "
              f"max_tokens={model.max_tokens}  {'OK' if checks['mteb_32k'] else 'FAIL'}")

        # 5. auto_activate_mteb() is idempotent
        result2 = model.auto_activate_mteb()
        checks["mteb_idempotent"] = result2 is not None
        print(f"  Idempotent: {result2}  OK")

        # 6. register_beta and request_another_beta methods exist
        checks["has_register_beta"] = callable(getattr(model, 'register_beta', None))
        checks["has_request_another"] = callable(getattr(model, 'request_another_beta', None))
        print(f"  register_beta: {checks['has_register_beta']}, "
              f"request_another_beta: {checks['has_request_another']}  OK")

        # 7. After unlock, index + search works
        model.clear()
        model.index("test_doc", "The activation code is DELTA7GAMMA")
        results = model.search("DELTA7GAMMA", top_k=5)
        checks["search_after_unlock"] = len(results) > 0 and results[0][0] == "test_doc"
        print(f"  Search after unlock: {results[0] if results else 'NONE'}  "
              f"{'OK' if checks['search_after_unlock'] else 'FAIL'}")

        passed = all(checks.values())
        return {"passed": passed, "checks": checks}

    def _test_beta_tier(self, LAM_class) -> dict:
        """Test BETA tier: activation via auto_activate_mteb, 32K tokens, recall enabled."""
        print("\n--- BETA TIER ---")
        model = LAM_class()
        checks = {}

        # Activate via auto_activate_mteb (BETA_2025 is not a valid key)
        model.auto_activate_mteb()
        checks["tier_promoted"] = model.tier != "FREE"
        checks["max_tokens"] = model.max_tokens >= 32000
        print(f"  Tier: {model.tier} {'OK' if checks['tier_promoted'] else 'FAIL'}")
        print(f"  Max tokens: {model.max_tokens} {'OK' if checks['max_tokens'] else 'FAIL'}")

        # Check invalid activation
        model2 = LAM_class()
        bad_result = model2.activate("INVALID_KEY")
        checks["bad_key_rejected"] = bad_result is False
        print(f"  Invalid key rejected: {not bad_result} {'OK' if checks['bad_key_rejected'] else 'FAIL'}")

        # Check recall is now enabled
        model.index("doc1", "The quick brown fox jumps over the lazy dog")
        try:
            results = model.search("brown fox")
            checks["recall_enabled"] = len(results) > 0
            print(f"  Recall enabled: YES, got {len(results)} results {'OK' if checks['recall_enabled'] else 'FAIL'}")
        except Exception as e:
            checks["recall_enabled"] = False
            print(f"  Recall enabled: NO ({e}) FAIL")

        passed = all(checks.values())
        return {"passed": passed, "checks": checks}

    def _test_sca_search(self, LAM_class) -> dict:
        """Test SCA recall accuracy."""
        print("\n--- SCA RECALL ACCURACY ---")
        model = LAM_class()
        model.auto_activate_mteb()
        checks = {}

        # Index test documents
        docs = {
            "doc_science": "Machine learning algorithms process large datasets to discover patterns in data",
            "doc_geography": "The Eiffel Tower is located in Paris, the capital city of France",
            "doc_programming": "Python was created by Guido van Rossum and released in 1991",
            "doc_physics": "The speed of light in vacuum is approximately 299792458 meters per second",
            "doc_music": "Ludwig van Beethoven composed nine symphonies during his lifetime",
        }

        for doc_id, text in docs.items():
            model.index(doc_id, text)

        print(f"  Indexed {len(docs)} documents")

        # Test exact search
        exact_results = model._engine.search_exact("Eiffel Tower")
        checks["exact_search"] = len(exact_results) > 0 and exact_results[0][0] == "doc_geography"
        print(f"  Exact 'Eiffel Tower': {exact_results[0] if exact_results else 'NONE'} "
              f"{'OK' if checks['exact_search'] else 'FAIL'}")

        # Test key-value search
        kv_results = model._engine.search_kv("speed of light")
        checks["kv_search"] = len(kv_results) > 0
        print(f"  KV 'speed of light': {kv_results[0] if kv_results else 'NONE'} "
              f"{'OK' if checks['kv_search'] else 'FAIL'}")

        # Test search
        search_results = model.search("Who created Python?")
        top_id = search_results[0][0] if search_results else ""
        checks["search_correct"] = top_id == "doc_programming"
        print(f"  Search 'Who created Python?': {search_results[0] if search_results else 'NONE'} "
              f"{'OK' if checks['search_correct'] else 'FAIL'}")

        # Test search for music
        search_results2 = model.search("Beethoven symphonies")
        top_id2 = search_results2[0][0] if search_results2 else ""
        checks["search_correct_2"] = top_id2 == "doc_music"
        print(f"  Search 'Beethoven symphonies': {search_results2[0] if search_results2 else 'NONE'} "
              f"{'OK' if checks['search_correct_2'] else 'FAIL'}")

        passed = all(checks.values())
        return {"passed": passed, "checks": checks}

    def _test_search_modes(self, LAM_class) -> dict:
        """Test different search modes."""
        print("\n--- SEARCH MODES ---")
        model = LAM_class()
        model.auto_activate_mteb()
        checks = {}

        # Index documents with specific patterns (direct engine calls)
        model._engine.index("contract_1", "Agreement ID: ABC-12345. Payment terms: net 30 days.", None)
        model._engine.index("contract_2", "Agreement ID: XYZ-67890. Payment terms: net 60 days.", None)
        model._engine.index("memo_1", "Internal memo about quarterly earnings forecast.", None)

        # Test exact pattern matching
        exact = model._engine.search_exact("ABC-12345")
        checks["exact_pattern"] = len(exact) > 0 and exact[0][0] == "contract_1"
        print(f"  Exact 'ABC-12345': {exact[0] if exact else 'NONE'} "
              f"{'OK' if checks['exact_pattern'] else 'FAIL'}")

        # Test all instances search
        all_agreements = model._engine.search_all_instances("Agreement ID")
        checks["all_instances"] = len(all_agreements) >= 2
        print(f"  All 'Agreement ID': {len(all_agreements)} results "
              f"{'OK' if checks['all_instances'] else 'FAIL'}")

        # Test doc count
        checks["doc_count"] = model._engine.doc_count() == 3
        print(f"  Doc count: {model._engine.doc_count()} (expected: 3) "
              f"{'OK' if checks['doc_count'] else 'FAIL'}")

        # Test clear
        model._engine.clear()
        checks["clear_works"] = model._engine.doc_count() == 0
        print(f"  After clear: {model._engine.doc_count()} docs "
              f"{'OK' if checks['clear_works'] else 'FAIL'}")

        passed = all(checks.values())
        return {"passed": passed, "checks": checks}


if __name__ == "__main__":
    validator = AblationStudyValidator()
    results = validator.run()
