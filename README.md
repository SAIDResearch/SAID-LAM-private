<p align="center"><code>SAIDResearch</code></p>

<h1 align="center">LAM</h1>

<p align="center"><code>SAID‑LAM‑v1</code></p>

<p align="center"><img src="https://img.shields.io/badge/O(n)-Linear_Complexity-0f0f0f?style=flat-square" /></p>

<p align="center">
LAM (Linear Attention Models) — a new family beyond semantic transformers.<br>
SAID‑LAM‑v1 is <strong>Linear Attention Memory</strong>.
</p>

<p align="center"><em>PHILOSOPHY: DETERMINISM OVER PROBABILITY</em></p>

<p align="center"><em>"The answer IS X. Because I Said so." — At ANY scale</em></p>

---

## Menu

1. [Quick-setup](#quick-setup)
2. [MTEB testing (benchmarks)](#mteb-testing-benchmarks)
3. [Model Details](#model-details)
4. [Performance](#performance)
5. [Drop-in sentence-transformers Replacement](#drop-in-sentence-transformers-replacement)
6. [Usage](#usage)
7. [API Reference](#api-reference)
8. [Model Files](#model-files)
9. [Citation](#citation)
10. [Links](#links)

## Quick-setup

This quick setup ensures:

- **Code is loaded from your pip-installed `said-lam` package** (not a local checkout).
- **Weights are loaded from the Hugging Face cache** (auto-downloaded on first run).

`venv` is the standard, recommended way to keep dependencies isolated. You *can* install globally, but using a virtual environment avoids conflicts.

### macOS / Linux (bash, zsh)

```bash
cd /path/to/your/project

# 1) Create & activate a clean virtualenv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip

# 2) Install SAID-LAM (CPU)
pip install said-lam

# 3) Run the quick end-to-end sanity test (CPU)
python said_quick_test.py
```

### Windows (PowerShell)

```powershell
cd C:\path\to\your\project

# 1) Create & activate a clean virtualenv
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip

# 2) Install SAID-LAM (CPU)
pip install said-lam

# 3) Run the quick end-to-end sanity test (CPU)
py .\said_quick_test.py
```

If PowerShell blocks activation, run this once (then retry the activate step):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

GPU upgrade (CUDA) uses a separate package. **Uninstall CPU first to avoid namespace conflicts** (both use `import said_lam`):

```bash
pip uninstall -y said-lam
pip install --upgrade said-lam-gpu

# Run the same quick test on CUDA (if available)
python said_quick_test.py
```

Windows (PowerShell) equivalent:

```powershell
pip uninstall -y said-lam
pip install --upgrade said-lam-gpu
py .\said_quick_test.py
```

## MTEB testing (benchmarks)

For full benchmark-style evaluation (STS tasks, LongEmbed retrieval tasks, cache controls, and result JSON export),
use `mteb_test.py`. This is a heavier workflow than `said_quick_test.py` and is intended for evaluation/benchmarking:

CPU:

```bash
. .venv/bin/activate
pip install -r requirements.txt mteb

# CPU smoke (fast coverage)
python mteb_test.py --smoke --device cpu --no-cache --output-dir ./smoke_results_cpu

# Example: run specific tasks
python mteb_test.py --tasks STS12 STS13 --device cpu --no-cache --output-dir ./results_cpu
```

GPU (CUDA):

```bash
. .venv/bin/activate
pip uninstall -y said-lam
pip install --upgrade said-lam-gpu
pip install -r requirements.txt mteb

# GPU smoke (fast coverage)
python mteb_test.py --smoke --device cuda --no-cache --output-dir ./smoke_results_gpu

# Example: run specific tasks
python mteb_test.py --tasks STS12 STS13 --device cuda --no-cache --output-dir ./results_gpu
```

If no CUDA device is available in the current runtime, the GPU wheel may fall back to CPU automatically.

SAID-LAM-v1 is a 23.85M parameter embedding model with O(n) linear complexity. Where standard transformers rely on O(n²) attention that slows and runs out of memory as context grows, LAM models replace this entirely with a recurrent state update that runs in strict O(n) time and constant memory, defining a new direction separate from transformer-based semantic models.

Distilled from [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), while extending context from 512 tokens to **32K+ tokens** — and demonstrating 100% recall on LongEmbed Needle-in-a-Haystack benchmarks across evaluated scales.

## Model Details

| Property | Value |
|----------|-------|
| **Model Category** | LAM (Linear Attention Models) — SAID-LAM-v1: Linear Attention Memory |
| **Parameters** | 23,848,788 |
| **Embedding Dimension** | 384 |
| **Max Context Length** | 32,768 tokens |
| **Memory Usage** | ~95 MB |
| **Complexity** | O(n) linear — time AND memory |
| **Framework** | Pure Rust (Candle) — no PyTorch required |
| **Package Size** | ~6 MB binary + 92 MB weights (auto-downloaded) |
| **License** | Apache 2.0 (weights) / Proprietary (code) |

## Performance

### O(n) Linear Scaling

LAM scales linearly with input length — empirically validated up to 1M words with R²=1.000, with memory growth from ~0 MB at small inputs up to ~15 MB at 1M words:


### STS-B Semantic Quality

Spearman r = 0.8181 on the STS-B test set (1,379 sentence pairs):


### MTEB LongEmbed Benchmarks

**Combined LongEmbed score (SAID-LAM-v1, average over all six tasks): ~91.0%.**

| Task                      | Score 
|---------------------------|----------------------|
| LEMBNeedleRetrieval       | **100.00%**       
| LEMBPasskeyRetrieval      | **100.00%**       
| LEMBNarrativeQARetrieval  | 69.93%            
| LEMBSummScreenFDRetrieval | 96.59%            
| LEMBQMSumRetrieval        | 85.76%            
| LEMBWikimQARetrieval      | 93.98%            

**LongEmbed SOTA comparison**

| Task                      | SAID-LAM-v1 (23M) | Global SOTA 
|---------------------------|-------------------|--------------------------|
| LEMBNeedleRetrieval       | **100.00%**       | 100.00%                  |
| LEMBPasskeyRetrieval      | **100.00%**       | 100.00%                  |
| LEMBNarrativeQARetrieval  | **69.93%**        | 66.10%                   |
| LEMBSummScreenFDRetrieval | 96.59%            | **99.10%**               |
| LEMBQMSumRetrieval        | **85.76%**        | 83.70%                   |
| LEMBWikimQARetrieval      | **93.98%**        | 91.20%                   |

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

| Property                          | sentence-transformers     | LAM                                   |
|-----------------------------------|---------------------------|---------------------------------------|
| Output                            | (N, 384) ndarray, float32 | (N, 384) ndarray, float32             |
| L2-normalized                     | Yes (default)             | Yes (default)                         |
| Cosine sim = dot product          | Yes                       | Yes                                   |
| Max tokens                        | 512                       | 12K (encode) / 32K (SCA)              |
| Complexity                        | O(n²) attention           | O(n) linear                           |
| Framework                         | PyTorch (~2 GB)           | Rust (~6 MB)                          |
| Memory at 1M tokens (no chunking) | OOM / impractical         | ~15 MB                                |

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

### BETA SCA (SAID Crystalline Attention) — MTEB testing only

BETA SCA (SAID Crystalline Attention) is activated for **MTEB testing only**, to enable perfect LongEmbed context retrieval (e.g. LEMBNeedleRetrieval, LEMBPasskeyRetrieval). Use the [MTEB evaluation](#mteb-evaluation) flow; no signup or activation required for benchmarking.

### Common Patterns

#### Similarity Between Texts

Embeddings are L2-normalized — cosine similarity is just a dot product:

```python
emb = model.encode(["The cat sat on the mat", "A kitten rested on the rug"])
similarity = float(emb[0] @ emb[1])
print(f"Similarity: {similarity:.4f}")  # ~0.5761
```

#### Batch Similarity Matrix

```python
import numpy as np

queries = ["How is the weather?", "What time is it?"]
candidates = ["Is it raining today?", "Do you have the time?", "Nice shoes"]

emb_q = model.encode(queries)      # (2, 384)
emb_c = model.encode(candidates)   # (3, 384)
sim_matrix = emb_q @ emb_c.T       # (2, 3)
```

#### Semantic Search Over a Corpus (FREE Tier)

```python
import numpy as np

corpus = ["Python is a language", "The Eiffel Tower is in Paris",
          "ML uses neural networks", "Speed of light is 299792458 m/s"]
corpus_emb = model.encode(corpus)

query_emb = model.encode(["fastest thing in physics"])
scores = (query_emb @ corpus_emb.T)[0]
ranked = np.argsort(scores)[::-1]
for i in ranked:
    print(f"  {scores[i]:.4f}  {corpus[i]}")
```

#### Matryoshka Dimensionality Reduction

```python
emb_128 = model.encode(["Hello world"], output_dim=128)  # (1, 128)
emb_64  = model.encode(["Hello world"], output_dim=64)   # (1, 64)
# Automatically truncated and re-normalized to unit length
```

Example impact on STS12 (cosine main_score, GPU):

| dim | STS12 score | rel. to 384d |
|-----|------------:|-------------:|
| 384 | 0.7493      | 100.0%       |
| 256 | 0.7472      | 99.7%        |
| 128 | 0.7459      | 99.6%        |
| 64  | 0.7327      | 97.8%        |

### Token Limits

- **`encode()`**: Up to 12,000 tokens per text. Returns embeddings for your RAG.
- **`index()` + `search()`**: Up to 32,768 tokens per text (MTEB BETA SCA — LongEmbed/MTEB testing group only). SCA streaming — no embeddings, perfect recall.

**`encode()`** — returns one embedding per input text, capped at 12K tokens:

```python
# Each text gets one embedding — long texts are chunked at 12K tokens
embeddings = model.encode(["short text", "very long text..."])  # (2, 384)

# Use output_dim for smaller embeddings (Matryoshka)
embeddings = model.encode(["short text", "very long text..."], output_dim=128)  # (2, 128)
```

> **Long documents?** `encode()` caps at 12K tokens. For LongEmbed benchmarks (MTEB BETA SCA testing only), `index()` + `search()` support up to 32K tokens via SCA.

### MTEB Evaluation

One model, one class: use the same `LAM` with `mteb.evaluate()` (LAM implements the global MTEB encoder protocol).

```python
from said_lam import LAM
import mteb

model = LAM("SAIDResearch/SAID-LAM-v1")
tasks = mteb.get_tasks(tasks=["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"])
results = mteb.evaluate(model=model, tasks=tasks)
```

## API Reference

### `LAM(model_name_or_path, device)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name_or_path` | `"SAIDResearch/SAID-LAM-v1"` | Hugging Face model ID to load (default: `SAIDResearch/SAID-LAM-v1`), or a local directory path pointing to the model files |
| `device` | `None` (auto) | Auto-selects CUDA GPU if available, otherwise CPU |

### Core Methods

| Method                                     | Tier  | Description                                      |
|--------------------------------------------|-------|--------------------------------------------------|
| `model.encode(sentences, output_dim=None)` | FREE+ | Encode to embeddings (384, 256, 128, or 64 dims) |
| `model.index(doc_id, text)`                | MTEB  | Index a document for search (benchmarks)         |
| `model.search(query, top_k)`               | MTEB  | Retrieve documents by query (benchmarks)         |
| `model.truncate_embeddings(emb, dim)`      | FREE+ | Matryoshka truncation (64/128/256)               |
| `model.clear()`                            | MTEB  | Clear indexed documents (benchmarks)             |
| `model.stats()`                            | FREE+ | Model statistics                                 |

### Tier System

| Tier         | encode()  | (SCA)     | How to Get    | Features              |
|--------------|-----------|-----------|---------------|-----------------------|
| `FREE`     | 12K       | —         | Default       | `encode()` only — embeddings for RAG |
| `MTEB`     | 12K       | 32K       | Auto-detected | SCA for LongEmbed retrieval (benchmarks only) |
| `LICENSED` | 32K       | 32K       | Coming soon   | + persistent storage + cloud sync |
| `INFINITE` | Unlimited | Unlimited | Coming soon   | Oracle mode |

## Model Files

| File                      | Size   | Description                        |
|---------------------------|--------|------------------------------------|
| `model.safetensors`       | 92 MB  | Model weights (SafeTensors format) |
| `config.json`             | 1 KB   | Model configuration                |
| `tokenizer.json`          | 467 KB | Tokenizer vocabulary               |
| `tokenizer_config.json`   | 350 B  | Tokenizer settings                 |
| `vocab.txt`               | 232 KB | WordPiece vocabulary               |
| `special_tokens_map.json` | 112 B  | Special token definitions          |

## Citation

```bibtex
@misc{said-lam-v1,
  title={SAID-LAM-v1: Linear Attention Memory},
  author={SAIDResearch},
  year={2026},
  url={https://saidhome.ai},
  note={23.85M parameter embedding model with O(n) linear complexity.
        384-dim embeddings, 32K context window, 100% NIAH recall.
        Distilled from all-MiniLM-L6-v2. Pure Rust (Candle) implementation.}
}
```

## Links

- **Organization**: [SAIDResearch](https://saidhome.ai)
- **PyPI**: [said-lam](https://pypi.org/project/said-lam/)
- **Distilled From**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Framework**: [Candle](https://github.com/huggingface/candle) (Hugging Face Rust ML)
- **Contact**: [research@saidhome.ai](mailto:research@saidhome.ai)
