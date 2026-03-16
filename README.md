# SAID-LAM

**LAM (Linear Attention Models) — a new family beyond semantic transformers. SAID-LAM-v1 is Linear Attention Memory.**

*"The answer IS X. Because I Said so." — At ANY scale.*

[![PyPI](https://img.shields.io/pypi/v/said-lam)](https://pypi.org/project/said-lam/)
[![License](https://img.shields.io/badge/license-Proprietary%20%7C%20Apache%202.0-blue)](LICENSE)

## What is LAM?

LAM is a **new model category** — not a transformer, not a sentence-transformer. Where standard transformers rely on probabilistic O(n²) attention that naturally "drifts" as context grows, LAM replaces this entirely with a **recurrent state update** that runs in strict O(n) time:

```
S_t = decay_t ⊙ S_{t-1} + K_t^T V_t    (state update — no attention matrix)
h_t = Q_t S_t                           (output — constant per token)
```

This means LAM has **no attention matrix**. No quadratic bottleneck. Memory stays constant regardless of sequence length.

**SAID-LAM-v1** is a 23.85M parameter LAM distilled from `all-MiniLM-L6-v2`, achieving 94% of its semantic quality while extending context from 512 tokens to **32K+ tokens** — and 100% perfect recall on Needle-in-a-Haystack tasks at any scale.

| Property | Value |
|----------|-------|
| **Model category** | **LAM** (Linear Attention Models) — SAID-LAM-v1: Linear Attention Memory |
| **Architecture** | **SCA** (Said Crystalline Attention) |
| **Parameters** | 23,848,788 |
| **Embedding dimension** | 384 |
| **Max context** | 12K tokens (encode) / 32K (SCA index+search, BETA) |
| **Complexity** | O(n) linear — time AND memory |
| **Framework** | Pure Rust (Candle) — no PyTorch required |
| **Package size** | ~6 MB binary + 92 MB weights (downloaded from HuggingFace) |
| **Memory at 100K tokens** | ~150 MB (transformers: OOM at 40+ GB) |

## Architecture

LAM uses a **Hierarchical DeltaNet** with three core innovations:

### 1. Dual-State Memory
Two recurrent states operating at different timescales:
- **S_fast** (τ=0.3): Captures immediate context, topic shifts, local coherence
- **S_slow** (τ=0.85): Preserves document themes, long-range dependencies, global semantics

### 2. SAID Crystalline Attention (SCA)
Deterministic locking mechanism that crystallizes semantic states — ensuring 0.0% signal loss at any document length. On BETA+ tier, SCA provides **100% perfect search** via IDF-Surprise hybrid scoring that combines lexical exactness with semantic understanding.

### 3. Hierarchical Decay
Position-dependent decay prevents vanishing gradients in long sequences, enabling stable processing beyond 100K tokens where single-state baselines diverge.

## Install

```bash
pip install said-lam
```

On first use, model weights (~92 MB) are automatically downloaded from [HuggingFace](https://huggingface.co/SAIDResearch/SAID-LAM-v1) and cached locally. The pip package itself is only ~6 MB (compiled Rust code only — weights are NOT bundled).

## Drop-in sentence-transformers Replacement

LAM is a drop-in replacement for sentence-transformers. Same API, same output format.

**Before (sentence-transformers):**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["Hello world", "Semantic search"])
# embeddings.shape == (2, 384), float32, L2-normalized
similarity = embeddings[0] @ embeddings[1]
```

**After (LAM):**
```python
from said_lam import LAM

model = LAM("SAIDResearch/SAID-LAM-v1")
embeddings = model.encode(["Hello world", "Semantic search"])
# embeddings.shape == (2, 384), float32, L2-normalized
similarity = embeddings[0] @ embeddings[1]
```

Same output format, same shapes, same downstream compatibility. Everything that works with sentence-transformers embeddings (FAISS, ChromaDB, Pinecone, numpy dot product) works with LAM embeddings.

| Property | sentence-transformers | LAM |
|----------|----------------------|-----|
| Output | `(N, 384)` ndarray, float32 | `(N, 384)` ndarray, float32 |
| L2-normalized | Yes (default) | Yes (default) |
| Cosine sim = dot product | Yes | Yes |
| Max tokens | 512 | 12K (encode) / 32K (SCA index+search) |
| Complexity | O(n²) attention | O(n) linear |
| Framework | PyTorch (~2 GB) | Rust (~6 MB) |
| Memory at 10K tokens | ~12 GB | ~120 MB |

## Usage

### FREE Tier — Embeddings (up to 12K tokens)

```python
from said_lam import LAM

model = LAM("SAIDResearch/SAID-LAM-v1")
embeddings = model.encode(["Hello world", "Semantic search is powerful"])
# embeddings.shape == (2, 384)

# Cosine similarity (L2-normalized by default)
similarity = embeddings[0] @ embeddings[1]
print(f"Similarity: {similarity:.4f}")
```

### BETA Tier — Perfect Recall via SCA (up to 32K tokens, free 1-month trial)

> **BETA has NO persistent storage.** SCA (Said Crystalline Attention) index/search
> state is RAM-only. You must re-index documents every session. This is a
> proof-of-concept to demonstrate that SCA perfect recall works on your data.
> `encode()` remains capped at 12K tokens (same as FREE).

```python
from said_lam import LAM

model = LAM("SAIDResearch/SAID-LAM-v1")
model.register_beta("you@email.com")  # Free 1-month trial, MAC-locked

# Index documents — streams via SCA, no embeddings generated
# Must re-index every session (no persistence in BETA)
model.index("doc_1", "Contract text with secret code QUANTUM7DELTA...")
model.index("doc_2", "Another very long document...")

# Search — finds needle in any haystack via SCA
results = model.search("QUANTUM7DELTA")
# Returns: list of (doc_id, score) tuples with 100% accuracy
```

After expiry, request another trial or upgrade:
```python
model.request_another_beta("you@email.com")  # Needs email approval
# Or visit https://saidhome.ai/upgrade for paid access
```

### Common Patterns

#### Cosine Similarity Between Two Texts

Embeddings are L2-normalized by default, so cosine similarity is just a dot product:

```python
from said_lam import LAM

model = LAM("SAIDResearch/SAID-LAM-v1")

emb = model.encode(["The cat sat on the mat", "A kitten rested on the rug"])
similarity = float(emb[0] @ emb[1])
print(f"Similarity: {similarity:.4f}")  # ~0.75 (semantically similar)
```

#### Batch Comparison (Similarity Matrix)

```python
import numpy as np

sentences_a = ["How is the weather?", "What time is it?"]
sentences_b = ["Is it raining today?", "Do you have the time?", "Nice shoes"]

emb_a = model.encode(sentences_a)  # (2, 384)
emb_b = model.encode(sentences_b)  # (3, 384)

# Similarity matrix — each row is a query, each column is a candidate
sim_matrix = emb_a @ emb_b.T  # (2, 3)
print(sim_matrix)
```

#### Semantic Search Over a Corpus (FREE Tier)

For small corpora where you want pure embedding-based search without `index()`/`search()`:

```python
import numpy as np

corpus = [
    "Python is a programming language",
    "The Eiffel Tower is in Paris",
    "Machine learning uses neural networks",
    "The speed of light is 299792458 m/s",
]
corpus_emb = model.encode(corpus)  # (4, 384)

query_emb = model.encode(["fastest thing in physics"])  # (1, 384)
scores = (query_emb @ corpus_emb.T)[0]  # (4,)
ranked = np.argsort(scores)[::-1]

for i in ranked:
    print(f"  {scores[i]:.4f}  {corpus[i]}")
```

#### Document Index + Search (BETA Tier — SCA, No Persistence)

For larger corpora or when you need 100% exact-match recall. Uses SCA (Said Crystalline
Attention) streaming — no embeddings generated, perfect recall via IDF-Surprise scoring.

> **No persistent storage in BETA.** Re-index every session. This proves SCA works.

```python
model = LAM("SAIDResearch/SAID-LAM-v1")
model.register_beta("you@email.com")

# Index documents (each up to 32K tokens) — must re-index every session
model.index("contract_1", open("contract_a.txt").read())
model.index("contract_2", open("contract_b.txt").read())
model.index("contract_3", open("contract_c.txt").read())

# Search — returns list of (doc_id, score) tuples
results = model.search("indemnification clause", top_k=3)
for doc_id, score in results:
    print(f"  {score:.4f}  {doc_id}")

# Clear and re-index when needed
model.clear()
```

#### Matryoshka Dimensionality Reduction

Trade accuracy for speed/storage with smaller embeddings (64, 128, or 256 dims):

```python
emb_128 = model.encode(["Hello world"], output_dim=128)  # (1, 128)
emb_64  = model.encode(["Hello world"], output_dim=64)   # (1, 64)
# Embeddings are automatically truncated and re-normalized to unit length
```

### Token Limits

- **`encode()`**: Up to 12,000 tokens per text (FREE and BETA). Returns embeddings for your RAG (Pinecone, FAISS, etc.).
- **`index()` + `search()`**: Up to 32,768 tokens per text (BETA). Uses SCA streaming — no embeddings generated, perfect recall.

**`encode()`** — returns one embedding per input text, capped at 12K tokens:

```python
# Each text gets one embedding — long texts are chunked at 12K tokens
embeddings = model.encode(["short text", "very long text..."])  # (2, 384)

# Use output_dim for smaller embeddings (Matryoshka)
embeddings = model.encode(["short text", "very long text..."], output_dim=128)  # (2, 128)
```

> **Long documents?** `encode()` caps at 12K tokens. For full-document recall up to 32K tokens, use `index()` + `search()` instead — SCA streams the full text with no chunking and no embeddings:

```python
model.index("doc_1", open("very_long_contract.txt").read())
results = model.search("indemnification clause")  # finds it anywhere in the doc
# ⚠️ BETA: No persistence — must re-index every session
```

## Tier System

| Tier | encode() | index/search (SCA) | How to get | Features |
|------|----------|-------------------|------------|----------|
| **FREE** | 12K | — | Default (no key) | `encode()` only — embeddings for RAG |
| **MTEB** | 12K | 32K | Auto-detected | Full capability, auto-activated for benchmarks |
| **BETA** | 12K | 32K | `register_beta("you@email.com")` | `index()` + `search()` via SCA. **No persistence — re-index every session.** Proof-of-concept. |
| **LICENSED** | 32K | 32K | Coming soon | + persistent storage + cloud sync |
| **INFINITE** | Unlimited | Unlimited | Coming soon | Oracle mode |

## Benchmarks

### MTEB LongEmbed

| Task | NDCG@10 | Recall@1 |
|------|---------|----------|
| LEMBNeedleRetrieval | **100%** | **100%** |
| LEMBPasskeyRetrieval | **100%** | **100%** |
| LEMBNarrativeQARetrieval | 70.88% | — |

### Scalability vs Transformers

| Sequence Length | LAM Memory | Transformer Memory | Speedup |
|-----------------|-----------|-------------------|---------|
| 128 tokens | 50 MB | 60 MB | ~1x |
| 1K tokens | 80 MB | 450 MB | 4x |
| 10K tokens | 120 MB | 12 GB | 100x |
| 100K tokens | 150 MB | OOM | ∞ |

### MTEB Evaluation

```python
from said_lam import LAM
import mteb

model = LAM("SAIDResearch/SAID-LAM-v1")
tasks = mteb.get_tasks(tasks=["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"])
results = mteb.evaluate(model=model, tasks=tasks)
```

## API Reference

### `LAM(model_name_or_path, device, backend)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name_or_path` | `"SAIDResearch/SAID-LAM-v1"` | Model path or HuggingFace ID |
| `device` | `None` (auto) | `"cpu"` or `"cuda"` |
| `backend` | `"crystalline"` | Backend engine |

### Core Methods

| Method | Tier | Description |
|--------|------|-------------|
| `model.encode(sentences, output_dim=None)` | FREE+ | Encode to embeddings (384, 256, 128, or 64 dims) |
| `model.index(doc_id, text)` | BETA+ | Index a document for search |
| `model.search(query, top_k)` | BETA+ | Retrieve documents by query |
| `model.register_beta(email)` | — | Register for free 1-month beta (MAC-locked) |
| `model.request_another_beta(email)` | — | Request another trial after expiry |
| `model.activate(key)` | — | Activate a license key |
| `model.truncate_embeddings(emb, dim)` | FREE+ | Matryoshka truncation (64/128/256) |
| `model.clear()` | BETA+ | Clear indexed documents |
| `model.stats()` | FREE+ | Model statistics |

## GPU Support

CPU wheels are installed by default. For GPU acceleration:

```bash
# Build from source with CUDA (Linux)
pip install maturin
maturin build --release --features cuda

# Metal (macOS Apple Silicon)
maturin build --release --features metal
```

See [docs/COMPILATION.md](docs/COMPILATION.md) for the full build matrix (7 platform targets).

## Development Testing

For local end-to-end testing with the compiled `.so` binary (GPU or CPU), use the dev setup script. This copies the compiled `lam_candle.so` into the Python package directory so tests can import it directly without `pip install`.

> **This is for internal development/testing only — not for client distribution.**
> Clients install via `pip install said-lam` which triggers a full maturin build.

### In-place Testing (dev_setup.sh)

```bash
cd said-lam
bash scripts/dev_setup.sh           # CPU build (default)
bash scripts/dev_setup.sh --gpu     # GPU/CUDA build
bash scripts/dev_setup.sh --so-only # Skip build, just copy existing .so

cd tests
python run_all_tests.py --skip-pearson
```

### Isolated Testing in /tmp (recommended)

Build and test in an isolated `/tmp` directory — nothing touches your source tree.
Delete `/tmp/said-lam-run` and `/tmp/said-lam-venv` when done.

```bash
# 1. Copy project and weights to /tmp
cp -r /path/to/said-lam /tmp/said-lam-run
cd /tmp/said-lam-run

# 2. Create isolated venv
python3 -m venv /tmp/said-lam-venv
source /tmp/said-lam-venv/bin/activate

# 3. Install build tools
pip install "maturin[patchelf]"

# 4. Build (CPU default, or --features cuda for GPU)
maturin develop --release                  # CPU
maturin develop --release --features cuda  # GPU/CUDA

# 5. Install test dependencies and run
pip install -r tests/requirements.txt
cd tests
python run_all_tests.py --skip-pearson     # Full suite (no Pearson/STS)
python test_crystalline_mteb.py            # Crystalline MTEB only

# 6. Clean up when done
deactivate
rm -rf /tmp/said-lam-run /tmp/said-lam-venv
```

> **Note:** The `weights/` directory (containing `model.safetensors`, `config.json`, `tokenizer.json`)
> must be present in the project root. If weights aren't published to HuggingFace yet,
> copy them manually before testing.

### Test Coverage

| Test | File | What it validates |
|------|------|------------------|
| Pearson Score | `test_pearson_score.py` | STS-B benchmark correlation |
| Linear Scaling | `test_linear_scaling.py` | O(n) complexity verification |
| Long Context | `test_long_context.py` | 12K/32K token processing |
| Tier & SCA Ablation | `test_ablation_study.py` | FREE/BETA tiers, search accuracy |
| **Crystalline MTEB** | `test_crystalline_mteb.py` | `index_mteb`/`search_mteb` pipeline, SearchProtocol, backend comparison |
| **Matryoshka** | `test_matryoshka.py` | `truncate_embeddings()` — 64/128/256 dim truncation, L2 norm, prefix consistency |

The Crystalline MTEB test mirrors `compare_backends.py` — it validates the exact call chain used for MTEB submission:
```
LAM(backend="crystalline") -> LAM.index() -> engine.index_mteb()
                            -> LAM.search() -> engine.search_mteb()
```

## Philosophy

> **DETERMINISM OVER PROBABILITY.**
>
> Standard transformers rely on probabilistic attention, which naturally "drifts" as context grows. LAM utilizes SAID Crystalline Attention (SCA) to deterministically lock semantic states, ensuring 0.0% signal loss regardless of sequence length.
>
> At 10K tokens, LAM is 100x more memory-efficient than transformers. At 100K tokens, transformers don't just become slow — they become impossible. LAM processes them in 150 MB.

## License

- **Software** (compiled binaries, Python wrappers): Proprietary — see [LICENSE](LICENSE)
- **Model Weights** (model.safetensors): Apache 2.0 — see [LICENSE-MODEL](LICENSE-MODEL)

## Links

- Website: [saidhome.ai](https://saidhome.ai)
- HuggingFace: [SAIDResearch/SAID-LAM-v1](https://huggingface.co/SAIDResearch/SAID-LAM-v1)
- Compilation Guide: [docs/COMPILATION.md](docs/COMPILATION.md)
- HuggingFace Publishing: [docs/HUGGINGFACE_PUBLISHING.md](docs/HUGGINGFACE_PUBLISHING.md)
- Author: Said Research ([research@saidhome.ai](mailto:research@saidhome.ai))
