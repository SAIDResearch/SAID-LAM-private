# SAID-LAM Test Suite

## Test Overview

| # | Test File | What It Validates |
|---|-----------|-------------------|
| 1 | `test_pearson_score.py` | STS-B Pearson correlation (semantic quality) |
| 2 | `test_linear_scaling.py` | O(n) time complexity (R^2 > 0.90) |
| 3 | `test_long_context.py` | 12K/32K token processing without OOM |
| 4 | `test_ablation_study.py` | Tier system, license unlock, SCA search |
| 5 | `test_crystalline_mteb.py` | MTEB pipeline (index_mteb + search_mteb) |
| 6 | `test_said_lam.py` | User API (encode/index/search) + MTEB wrapper |
| 7 | `test_matryoshka.py` | Matryoshka truncation (64/128/256 dims) |
| 8 | `test_embeddings.py` | sentence-transformers parity (drop-in proof) |

Run all:
```bash
cd said-lam/tests && python run_all_tests.py
python run_all_tests.py --skip-pearson   # skip STS-B (needs dataset download)
```

---

## Part 1: sentence-transformers Parity

LAM's `encode()` matches the `SentenceTransformer.encode()` contract:

| Check | sentence-transformers | LAM | Test |
|-------|----------------------|-----|------|
| `encode(["text"])` -> `(1, 384)` ndarray | Yes | Yes | `test_embeddings` |
| `encode(["a","b","c"])` -> `(3, 384)` | Yes | Yes | `test_embeddings` |
| `encode([])` -> `(0, 384)` | Yes | Yes | `test_embeddings` |
| `encode("string")` auto-wraps | Yes | Yes | `test_embeddings` |
| `dtype == float32` | Yes | Yes | `test_embeddings` |
| L2-normalized (norm ~1.0) | Yes (default) | Yes (default) | `test_embeddings` |
| Cosine sim = dot product | Yes (when normalized) | Yes | `test_embeddings` |
| `batch_size=32` param | Yes | Yes | `__init__.py` |
| Deterministic output | Yes | Yes | `test_embeddings` |
| Similar texts score higher | Yes | Yes | `test_embeddings` |
| Self-similarity ~1.0 | Yes | Yes | `test_embeddings` |
| Large batch (50+) no OOM | Yes | Yes | `test_embeddings` |
| Unicode/special chars | Yes | Yes | `test_embeddings` |
| `output_dim` (Matryoshka) | No (separate lib) | Yes (built-in) | `test_matryoshka` |
| Similarity matrix `emb_q @ emb_c.T` | Yes | Yes | `test_embeddings` |
| RAG pattern: dot product search | Yes | Yes | `test_embeddings` |
| Truncate + re-normalize | No | Yes | `test_matryoshka` |

**Vector store compatibility** (encode output works directly with):

| Store | Method | Works with LAM? |
|-------|--------|----------------|
| FAISS IndexFlatIP | `index.add(emb)` | Yes (L2-normalized -> dot product = cosine) |
| ChromaDB | `collection.add(embeddings=emb.tolist())` | Yes |
| Pinecone | `index.upsert(vectors=[(id, emb)])` | Yes |
| Qdrant | `client.upsert(points=[...])` | Yes |
| Weaviate | `client.data_object.create(vector=emb)` | Yes |
| NumPy dot product | `query @ corpus.T` | Yes |

---

## Part 2: What sentence-transformers Cannot Do

These tests validate LAM capabilities that sentence-transformers physically cannot match:

| Test | LAM Result | sentence-transformers |
|------|-----------|----------------------|
| 12K token document | Encodes in O(n) | 512 token limit, truncated |
| 32K token document (BETA SCA index/search) | Streams in O(n) | OOM or truncated |
| 50K+ word document | ~150 MB memory | OOM (40+ GB attention matrix) |
| O(n) linear scaling R^2 | 0.991 | O(n^2) quadratic |
| 100% needle recall (SCA) | Perfect via index+search | No equivalent feature |
| Matryoshka built-in | `output_dim=128` | Requires separate `MatryoshkaLoss` |
| Memory at 10K tokens | ~120 MB | ~12 GB |

---

## Part 3: LongEmbed Benchmarks

LongEmbed is a **separate leaderboard** from MTEB(eng, v2). These 6 tasks test
retrieval over long documents (256 to 32K+ tokens per document).

| Task | NDCG@10 | Notes |
|------|---------|-------|
| LEMBNeedleRetrieval | **100.0%** | Perfect needle-in-haystack at all context lengths |
| LEMBPasskeyRetrieval | **100.0%** | Perfect passkey retrieval at all context lengths |
| LEMBNarrativeQARetrieval | 70.88% | Long narrative comprehension |
| LEMBQMSumRetrieval | 88.95% | Query-based summarization retrieval |
| LEMBSummScreenFDRetrieval | 92.38% | Screenplay/dialogue retrieval |
| LEMBWikimQARetrieval | 94.06% | Wikipedia long-form QA |
| **Average** | **91.0%** | |

sentence-transformers on LongEmbed: **Cannot run** (512 token limit).

---

## Part 4: MTEB(eng, v2) — 41 Tasks

`MTEB(eng, v2)` is the standard English benchmark with **41 tasks** across 7 categories.
This is a separate leaderboard from LongEmbed.

| Category | Tasks | LAM Avg |
|----------|-------|---------|
| STS (Semantic Textual Similarity) | 9 | 78.7% |
| Retrieval (short-text) | 10 | 32.5% |
| Clustering | 8 | 26.8% |
| Classification | 8 | 63.2% |
| Pair Classification | 3 | 80.4% |
| Reranking | 2 | 52.1% |
| Summarization | 1 | 29.3% |

**Where LAM excels**: STS (78.7%), Pair Classification (80.4%), LongEmbed (91.0%).

**Where LAM trades off**: Short-text retrieval, clustering. LAM is a 24M param model
distilled from MiniLM — it's not designed to compete with 300M+ parameter models on
short-text retrieval. Its differentiator is **long context** and **perfect recall**.

### Rust vs PyTorch Engine

| Engine | MTEB Version | Tasks Run | LongEmbed Avg |
|--------|-------------|-----------|---------------|
| Rust (Candle) — current | v2.9.0 | 22 | **91.0%** |
| PyTorch — legacy | v2.4.2 | 61 | 74.9% |

LongEmbed improved from 74.9% (PyTorch) to 91.0% (Rust) — this is the SCA
Crystalline integration making the difference.

---

## Part 5: The Case for Switching

### This is NOT about speed

LAM does not claim to be faster than sentence-transformers at short texts. A 512-token
sentence-transformers model on GPU is very fast. LAM's advantage is elsewhere.

### This IS about accuracy and perfect recall

| Scenario | sentence-transformers | LAM (BETA) |
|----------|----------------------|------------|
| 500-word document | Works fine | Works fine |
| 10K-word contract | Truncated at 512 tokens | Full document, O(n) |
| 50K-word legal filing | Cannot process | Full document, ~150 MB |
| "Find code QUANTUM7DELTA in 100 docs" | ~60% recall (chunking) | **100% recall** (SCA) |
| Needle in 32K token haystack | Cannot test | **100% NDCG@10** |

### Tier progression (sales funnel)

1. **FREE** (12K tokens, encode only) — Proves semantic quality.
   User evaluates embeddings, similarity, RAG with numpy/FAISS.
   No commitment required.

2. **BETA** (12K encode, 32K SCA index/search, 30-day trial) — Proves perfect recall.
   `register_beta("you@email.com")` unlocks `index()` + `search()` via SCA.
   User sees 100% recall on their own data. MAC-locked, auto-expires.
   **No persistent storage — must re-index every session. Proof-of-concept only.**

3. **LICENSED** (32K+, production) — The sellable solution.
   Everything proven in BETA, now with support and cloud persistence.
   Visit https://saidhome.ai/upgrade.

### What the tests prove

| Test | What it proves to users |
|------|------------------------|
| `test_embeddings.py` | "I can replace sentence-transformers today" |
| `test_pearson_score.py` | "Semantic quality is production-grade (r=0.817)" |
| `test_linear_scaling.py` | "O(n) scaling is real, not marketing" |
| `test_long_context.py` | "My 10K-word documents actually work" |
| `test_ablation_study.py` | "License unlock is painless, recall is 100%" |
| `test_crystalline_mteb.py` | "MTEB scores are reproducible and verifiable" |
| `test_matryoshka.py` | "I can reduce dims for faster similarity search" |
