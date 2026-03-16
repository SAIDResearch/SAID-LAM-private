"""
Test 3: Long Context Processing (SCA + Rust Core)
===================================================

Validates LAM's long-document handling using three indexing paths:

1. encode()  — embeds text to 384-dim vector.
   Limited by tokenizer truncation (max_length=128 in tokenizer.json).
   Model MAX_POSITION=512. Tier limits (12K/32K) declared but not enforced.

2. index()   — Crystalline's standard indexing. Two sub-indexes:
   - BERT token index: single tokenizer.encode(full_text) -> truncated to 128 tokens.
   - Word-level index: simple_tokenize(full_text) -> NO truncation, covers full doc.
   So search() works on long docs via the word path; BERT path is truncated.
   search_exact() and search_kv() operate on stored full text (text_store).

3. stream_index()  — Crystalline's chunked indexing for long documents.
   - BERT token index: chunks text by chars, tokenizes each chunk -> full-doc BERT coverage.
   - Word-level index: NOT updated (only BERT tokens indexed per chunk).
   This is the proper path for long-document BERT token coverage.

This test exercises all three paths and documents the differences:
- encode():       stability, token audit, embedding differentiation
- index():        needle-in-haystack, multi-needle, KV extraction, exact search
                  (works via word path + text_store substring match)
- stream_index(): scaling test with full BERT coverage, comparison vs index()
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

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class LongContextValidator:
    """Validates long context handling via SCA (Rust CrystallineCore) and encode()."""

    def __init__(self):
        self.results_dir = Path(__file__).parent / "results"
        self.viz_dir = Path(__file__).parent / "visualizations"
        self.results_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)
        self._tokenizer = None

    def _load_tokenizer(self):
        """Load the BERT tokenizer from weights/ (same one the Rust engine uses)."""
        if self._tokenizer is None:
            try:
                from tokenizers import Tokenizer
                tok_path = _repo_root / "weights" / "tokenizer.json"
                if tok_path.exists():
                    self._tokenizer = Tokenizer.from_file(str(tok_path))
            except ImportError:
                pass
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer (or estimate)."""
        tok = self._load_tokenizer()
        if tok is not None:
            return len(tok.encode(text).ids)
        return int(len(text.split()) * 1.3)

    def _generate_noise(self, n_words: int, topic: str = "legal") -> str:
        """Generate filler text. Different topics = different vocabulary."""
        topics = {
            "legal": (
                "Whereas the parties hereto agree to the provisions set forth in this "
                "agreement notwithstanding any prior obligations or representations. "
                "The indemnification clause shall apply to all liabilities thereof. "
                "Furthermore the covenant restricts assignment without written consent. "
            ),
            "science": (
                "Machine learning algorithms process data to find patterns. "
                "Neural networks simulate biological neurons to learn representations. "
                "Gradient descent optimizes model parameters through backpropagation. "
                "Regularization techniques prevent overfitting on training data. "
            ),
            "finance": (
                "The stock market fluctuated sharply during the economic downturn. "
                "Investors diversified their portfolios to minimize risk exposure. "
                "Bond yields inversely correlate with prevailing interest rates. "
                "Quarterly earnings exceeded analyst expectations by a wide margin. "
            ),
        }
        base = topics.get(topic, topics["legal"])
        base_words = len(base.split())
        repeats = max(1, n_words // base_words)
        return (base * repeats).strip()

    def _build_haystack(self, needle: str, n_words: int, position: float = 0.5) -> str:
        """Build a document with needle at a specific position in noise."""
        words_before = int(n_words * position)
        words_after = n_words - words_before
        before = self._generate_noise(words_before, "legal")
        after = self._generate_noise(words_after, "finance")
        return f"{before} {needle} {after}"

    def run(self) -> dict:
        """Run all long context validation tests."""
        from said_lam import LAM

        print("\n" + "=" * 70)
        print("TEST 3: Long Context Processing (SCA + Rust Core)")
        print("=" * 70)

        model = LAM()
        _ = model.encode(["warmup"])

        results = {}
        all_passed = True

        # ── encode() path tests ──
        print(f"\n--- ENCODE STABILITY (tier={model.tier}) ---")
        results["encode_lengths"] = self._test_encode_lengths(model)

        print("\n--- TOKEN COUNT AUDIT ---")
        results["token_audit"] = self._test_token_audit(model)

        print("\n--- EMBEDDING DIFFERENTIATION ---")
        results["differentiation"] = self._test_embedding_differentiation(model)
        if not results["differentiation"]["passed"]:
            all_passed = False

        # ── Activate BETA for SCA tests ──
        print(f"\n--- ACTIVATING BETA FOR SCA TESTS ---")
        model.auto_activate_mteb()
        print(f"  Tier: {model.tier}, Max tokens: {model.max_tokens}")

        # ── SCA search tests (Rust CrystallineCore) ──
        print("\n--- SCA NEEDLE-IN-HAYSTACK ---")
        results["needle_haystack"] = self._test_needle_haystack(model)
        if not results["needle_haystack"]["passed"]:
            all_passed = False

        print("\n--- SCA MULTI-NEEDLE ---")
        results["multi_needle"] = self._test_multi_needle(model)
        if not results["multi_needle"]["passed"]:
            all_passed = False

        print("\n--- SCA KEY-VALUE EXTRACTION ---")
        results["kv_extraction"] = self._test_kv_extraction(model)
        if not results["kv_extraction"]["passed"]:
            all_passed = False

        print("\n--- SCA EXACT SEARCH ---")
        results["exact_search"] = self._test_exact_search(model)
        if not results["exact_search"]["passed"]:
            all_passed = False

        print("\n--- INDEX vs STREAM_INDEX COMPARISON ---")
        results["index_comparison"] = self._test_index_vs_stream_index(model)
        if not results["index_comparison"]["passed"]:
            all_passed = False

        print("\n--- STREAM_INDEX SCALING (increasing doc sizes) ---")
        results["scaling"] = self._test_scaling(model)
        if not results["scaling"]["passed"]:
            all_passed = False

        # ── BETA encode lengths ──
        print(f"\n--- BETA ENCODE LENGTHS ---")
        results["beta_encode"] = self._test_encode_lengths(model)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for name, r in results.items():
            if isinstance(r, dict) and "passed" in r:
                status = "PASS" if r["passed"] else "FAIL"
                print(f"  {name:30s} {status}")
            elif isinstance(r, list):
                errors = sum(1 for x in r if x.get("status") == "error")
                print(f"  {name:30s} {len(r)} lengths, {errors} errors")
        overall = "ALL PASSED" if all_passed else "SOME FAILED"
        print(f"  Overall: {overall}")
        print("=" * 70)

        out_path = self.results_dir / "long_context_validation.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved to: {out_path}")

        return results

    # ════════════════════════════════════════════════════════════════════════
    # ENCODE PATH TESTS
    # ════════════════════════════════════════════════════════════════════════

    def _test_encode_lengths(self, model) -> list:
        """Encode texts at various lengths — reports words AND tokens."""
        test_results = []
        header = f"{'Words':>8} | {'Tokens':>8} | {'Shape':>12} | {'Time':>8} | {'Norm':>8} | {'OK?':>4}"
        print(f"  {header}")
        print("  " + "-" * len(header))

        for n_words in [100, 500, 1000, 5000, 10000, 20000]:
            text = self._generate_noise(n_words, "science")
            actual_words = len(text.split())
            actual_tokens = self._count_tokens(text)

            try:
                t0 = time.perf_counter()
                emb = model.encode([text])
                elapsed = time.perf_counter() - t0
                norm = float(np.linalg.norm(emb[0]))
                shape = str(emb.shape)
                status = "OK"
            except Exception as e:
                elapsed = 0
                norm = 0
                shape = "—"
                status = "ERR"
                test_results.append({"words": actual_words, "tokens": actual_tokens,
                                     "status": "error", "error": str(e)})
                print(f"  {actual_words:>8} | {actual_tokens:>8} | {'—':>12} | {'—':>8} | {'—':>8} | {status}")
                continue

            print(f"  {actual_words:>8} | {actual_tokens:>8} | {shape:>12} | {elapsed:>7.3f}s | {norm:>8.4f} | {status}")
            test_results.append({"words": actual_words, "tokens": actual_tokens,
                                 "shape": shape, "time_s": round(elapsed, 3),
                                 "norm": round(norm, 4), "status": status})
        return test_results

    def _test_token_audit(self, model) -> dict:
        """Report how many tokens the tokenizer produces vs its truncation limit."""
        tok = self._load_tokenizer()
        if tok is None:
            print("  tokenizers library not installed — SKIP")
            return {"passed": True, "skipped": True}

        print(f"  {'Label':>12} | {'Words':>7} | {'Raw tokens':>11} | {'Max':>5} | {'Truncated?':>10}")
        print("  " + "-" * 55)

        cases = [
            ("6 words", "The cat sat on the mat"),
            ("90 words", " ".join(["The quick brown fox jumps over the lazy dog."] * 10)),
            ("900 words", " ".join(["The quick brown fox jumps over the lazy dog."] * 100)),
        ]
        for label, text in cases:
            words = len(text.split())
            raw = len(tok.encode(text).ids)
            trunc = "no" if raw <= 128 else f"yes ({raw}->128)"
            print(f"  {label:>12} | {words:>7} | {raw:>11} | {'128':>5} | {trunc:>10}")

        print("\n  NOTE: tokenizer truncation (128) affects encode() and index()'s BERT path.")
        print("  index()'s word path covers full doc. stream_index() chunks for full BERT coverage.")
        return {"passed": True}

    def _test_embedding_differentiation(self, model) -> dict:
        """Different topics should produce different embeddings."""
        checks = {}
        texts = [
            "The tabby cat purred softly on the warm windowsill watching birds.",
            "Stock markets crashed after the central bank unexpectedly raised rates.",
            "The astronaut floated weightlessly inside the International Space Station.",
            "She baked a rich chocolate cake with vanilla frosting for the party.",
        ]
        labels = ["cat", "stocks", "space", "cake"]
        emb = model.encode(texts)

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = float(emb[i] @ emb[j])
                ok = sim < 0.95
                pair = f"{labels[i]}↔{labels[j]}"
                checks[pair] = ok
                print(f"  {pair:>15}: {sim:.4f}  {'OK' if ok else 'FAIL'}")

        sim_self = float(emb[0] @ emb[0])
        checks["self_sim"] = sim_self > 0.99
        print(f"  {'self':>15}: {sim_self:.4f}  {'OK' if checks['self_sim'] else 'FAIL'}")
        return {"passed": all(checks.values()), "checks": checks}

    # ════════════════════════════════════════════════════════════════════════
    # SCA SEARCH TESTS (Rust CrystallineCore)
    # ════════════════════════════════════════════════════════════════════════

    def _test_needle_haystack(self, model) -> dict:
        """Hide a unique needle in a long document, retrieve via SCA search.

        Tests three search modes:
        1. Keyword search via model.search() — hybrid lexical+semantic
        2. Exact substring search via search_exact() — pure lexical
        3. The needle is "The purple elephant danced gracefully on the frozen
           lake at midnight" — unique vocabulary that won't appear in legal/finance noise.
        """
        checks = {}
        model.clear()

        needle = "The purple elephant danced gracefully on the frozen lake at midnight."
        doc = self._build_haystack(needle, n_words=5000, position=0.5)
        actual_words = len(doc.split())
        actual_tokens = self._count_tokens(doc)

        model.index("doc_haystack", doc)
        print(f"  Indexed 1 doc: {actual_words} words, ~{actual_tokens} tokens")

        # 1. Keyword search
        results = model.search("purple elephant frozen lake", top_k=5)
        found = len(results) > 0 and results[0][0] == "doc_haystack"
        score = results[0][1] if results else 0
        checks["keyword_search"] = found
        print(f"  search('purple elephant frozen lake'): "
              f"{'FOUND' if found else 'MISS'} (score={score:.4f})  "
              f"{'OK' if found else 'FAIL'}")

        # 2. Exact search
        exact = model._engine.search_exact("purple elephant")
        found_exact = len(exact) > 0 and exact[0][0] == "doc_haystack"
        checks["exact_search"] = found_exact
        print(f"  search_exact('purple elephant'): "
              f"{'FOUND' if found_exact else 'MISS'}  "
              f"{'OK' if found_exact else 'FAIL'}")

        # 3. Exact search for full phrase
        exact_full = model._engine.search_exact("danced gracefully on the frozen lake")
        found_full = len(exact_full) > 0 and exact_full[0][0] == "doc_haystack"
        checks["exact_phrase"] = found_full
        print(f"  search_exact('danced gracefully...'): "
              f"{'FOUND' if found_full else 'MISS'}  "
              f"{'OK' if found_full else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_multi_needle(self, model) -> dict:
        """Index 5 documents, each with a unique needle in noise. Search for each.

        Each needle has topically distinct vocabulary:
        - cat/fireplace, astronaut/spacewalk, risotto/truffle, cellist/Bach, goalkeeper/penalty

        Uses keyword queries to test SCA's ability to route to the correct document.
        """
        checks = {}
        model.clear()

        needles = {
            "doc_cat": "The orange tabby cat napped on the velvet cushion by the fireplace.",
            "doc_space": "The astronaut repaired the solar panel during the third spacewalk.",
            "doc_cooking": "She garnished the risotto with shaved truffle and fresh microgreens.",
            "doc_music": "The cellist performed a haunting rendition of Bach Cello Suite No 1.",
            "doc_sports": "The goalkeeper saved the penalty kick in the final minute of overtime.",
        }

        for doc_id, needle in needles.items():
            doc = self._build_haystack(needle, n_words=2000, position=0.4)
            model.index(doc_id, doc)
        print(f"  Indexed {len(needles)} documents (each ~2000 words with unique needle)")

        queries = {
            "doc_cat": "tabby cat velvet cushion fireplace",
            "doc_space": "astronaut solar panel spacewalk",
            "doc_cooking": "risotto truffle microgreens",
            "doc_music": "cellist Bach Cello Suite",
            "doc_sports": "goalkeeper penalty kick overtime",
        }

        for expected_id, query in queries.items():
            results = model.search(query, top_k=5)
            top_id = results[0][0] if results else ""
            top_score = results[0][1] if results else 0
            ok = top_id == expected_id
            checks[expected_id] = ok
            print(f"  '{query[:35]:35s}' -> {top_id:12s} ({top_score:.4f})  "
                  f"{'OK' if ok else f'FAIL (got {top_id})'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_kv_extraction(self, model) -> dict:
        """Extract values from KV pairs hidden deep in a long document.

        Uses SCA's search_kv() — the Rust CrystallineCore finds the key
        and extracts the value after it (skips separators like ':', '=').
        """
        checks = {}
        model.clear()

        kv_section = (
            "CRITICAL DATA: "
            "The activation code is: DELTA7BRAVO. "
            "The serial number is: SN-9847-XQ. "
            "The authorization key is: AUTH-2024-GAMMA. "
            "END OF DATA."
        )
        before = self._generate_noise(3000, "legal")
        after = self._generate_noise(3000, "finance")
        doc = f"{before} {kv_section} {after}"

        model.index("doc_kv", doc)
        actual_words = len(doc.split())
        print(f"  Indexed 1 doc: {actual_words} words with 3 KV pairs")

        kv_tests = [
            ("activation code", "DELTA7BRAVO"),
            ("serial number", "SN-9847-XQ"),
            ("authorization key", "AUTH-2024-GAMMA"),
        ]

        for key, expected_value in kv_tests:
            kv_results = model._engine.search_kv(key)
            if kv_results:
                found_value = kv_results[0][0]
                found_doc = kv_results[0][1]
                ok = expected_value in found_value and found_doc == "doc_kv"
                checks[f"kv_{key.replace(' ', '_')}"] = ok
                print(f"  search_kv('{key}'): '{found_value}' from {found_doc}  "
                      f"{'OK' if ok else 'FAIL'}")
            else:
                checks[f"kv_{key.replace(' ', '_')}"] = False
                print(f"  search_kv('{key}'): no results  FAIL")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_exact_search(self, model) -> dict:
        """Find exact codes/identifiers buried in long documents.

        Uses SCA's search_exact() — two-stage: token filter (fast rejection)
        then substring match for 100% precision.
        """
        checks = {}
        model.clear()

        codes = {
            "doc_alpha": "QUANTUM7DELTA",
            "doc_beta": "NEXUS3OMEGA",
            "doc_gamma": "CIPHER9ECHO",
        }

        for doc_id, code in codes.items():
            noise = self._generate_noise(5000, "legal")
            words = noise.split()
            mid = len(words) // 2
            words.insert(mid, f"The secret code is {code} embedded here.")
            doc = " ".join(words)
            model.index(doc_id, doc)

        print(f"  Indexed {len(codes)} documents (~5000 words each, 1 unique code per doc)")

        for expected_id, code in codes.items():
            exact_results = model._engine.search_exact(code)
            if exact_results:
                found_id = exact_results[0][0]
                found_score = exact_results[0][1]
                ok = found_id == expected_id
                checks[f"exact_{code}"] = ok
                print(f"  search_exact('{code}'): {found_id} ({found_score:.1f})  "
                      f"{'OK' if ok else f'FAIL (expected {expected_id})'}")
            else:
                checks[f"exact_{code}"] = False
                print(f"  search_exact('{code}'): no results  FAIL")

        # Negative: code that doesn't exist
        phantom = model._engine.search_exact("NONEXISTENT99PHANTOM")
        checks["negative_exact"] = len(phantom) == 0
        print(f"  search_exact('NONEXISTENT99PHANTOM'): {len(phantom)} results  "
              f"{'OK' if checks['negative_exact'] else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_index_vs_stream_index(self, model) -> dict:
        """Compare index() vs stream_index() on the same long document.

        index():
          - BERT tokens: truncated (tokenizer max_length=128) -> only first 128 tokens indexed
          - Word-level: full doc (simple_tokenize covers everything)
          - search_exact() works via text_store substring match (full text stored)

        stream_index():
          - BERT tokens: chunked -> full doc coverage (each chunk tokenized separately)
          - Word-level: NOT updated (only BERT inverted index + token sets)
          - search_exact() works via same text_store + better BERT candidate filtering

        Both should find the needle, but stream_index() returns chunk/token stats.
        """
        checks = {}

        needle = "ZEPHYR9CRYSTAL"
        needle_sentence = f"The unique code {needle} was embedded in this long document."
        doc = self._build_haystack(needle_sentence, n_words=10000, position=0.5)
        actual_words = len(doc.split())
        actual_tokens = self._count_tokens(doc)
        print(f"  Document: {actual_words} words, ~{actual_tokens} tokens")

        # Test with index() (standard)
        model.clear()
        model.index("doc_standard", doc)
        exact_std = model._engine.search_exact(needle)
        found_std = len(exact_std) > 0 and exact_std[0][0] == "doc_standard"
        checks["index_finds_needle"] = found_std
        print(f"  index() + search_exact('{needle}'): "
              f"{'FOUND' if found_std else 'MISS'}  "
              f"{'OK' if found_std else 'FAIL'}")

        # Test with stream_index() (chunked BERT)
        model.clear()
        stats = model._engine.stream_index("doc_streamed", doc, 2000)
        chunks = stats.get("chunks", 0)
        indexed_tokens = stats.get("tokens", 0)
        print(f"  stream_index(): {chunks} chunks, {indexed_tokens} unique BERT tokens indexed")

        exact_stream = model._engine.search_exact(needle)
        found_stream = len(exact_stream) > 0 and exact_stream[0][0] == "doc_streamed"
        checks["stream_index_finds_needle"] = found_stream
        print(f"  stream_index() + search_exact('{needle}'): "
              f"{'FOUND' if found_stream else 'MISS'}  "
              f"{'OK' if found_stream else 'FAIL'}")

        # stream_index should have more BERT tokens indexed than index()
        # (index truncates to ~128; stream_index covers full doc)
        checks["stream_has_more_tokens"] = indexed_tokens > 128
        print(f"  stream_index BERT tokens ({indexed_tokens}) > 128: "
              f"{'OK' if checks['stream_has_more_tokens'] else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}

    def _test_scaling(self, model) -> dict:
        """Needle retrieval as document size scales from 1K to 50K words.

        Uses stream_index() for full BERT coverage at all document sizes.
        The needle stays the same; noise grows. SCA should maintain perfect
        recall at all sizes.
        """
        checks = {}
        needle = "SCALING_TEST_NEEDLE_X7Q9Z"
        needle_sentence = f"The unique identifier {needle} was recorded in the database."

        sizes = [1000, 5000, 10000, 20000, 50000]
        print(f"  {'Words':>8} | {'Tokens':>8} | {'Method':>14} | {'Index':>8} | {'Search':>8} | {'OK?':>4}")
        print("  " + "-" * 65)

        for n_words in sizes:
            model.clear()
            doc = self._build_haystack(needle_sentence, n_words=n_words, position=0.5)
            actual_words = len(doc.split())
            actual_tokens = self._count_tokens(doc)

            t0 = time.perf_counter()
            stats = model._engine.stream_index("doc_scale", doc, 2000)
            index_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            exact_results = model._engine.search_exact(needle)
            search_time = time.perf_counter() - t1

            found = len(exact_results) > 0 and exact_results[0][0] == "doc_scale"
            chunks = stats.get("chunks", 0)
            checks[f"scale_{n_words}"] = found
            print(f"  {actual_words:>8} | {actual_tokens:>8} | "
                  f"{'stream('+str(chunks)+'ch)':>14} | {index_time:>7.3f}s | "
                  f"{search_time:>7.4f}s | {'OK' if found else 'FAIL'}")

        return {"passed": all(checks.values()), "checks": checks}


if __name__ == "__main__":
    validator = LongContextValidator()
    results = validator.run()
    all_passed = True
    for name, r in results.items():
        if isinstance(r, dict) and "passed" in r:
            if not r["passed"]:
                all_passed = False
    sys.exit(0 if all_passed else 1)
