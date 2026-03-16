# SAID-LAM Complete Submission Guide

**End-to-end: HuggingFace weights → PyPI package → GitHub releases (.so binaries) → MTEB leaderboard**

All commands reference the canonical source at `said-lam/`.

---

## Table of Contents

1. [Overview: What Goes Where](#1-overview-what-goes-where)  
   - [Step-by-step: Get the private repo uploaded so you can start wheel builds](#step-by-step-get-the-private-repo-uploaded-so-you-can-start-wheel-builds)
2. [Step 1: HuggingFace Hub — Model Weights](#2-step-1-huggingface-hub--model-weights)
3. [Step 2: PyPI — pip install said-lam](#3-step-2-pypi--pip-install-said-lam)
4. [Step 3: GitHub Releases — Compiled .so Binaries](#4-step-3-github-releases--compiled-so-binaries)
5. [Step 4: MTEB Leaderboard — Model Wrapper PR + Results PR](#5-step-4-mteb-leaderboard--model-wrapper-pr--results-pr)
6. [Licensing Summary](#6-licensing-summary)
7. [Pre-Flight Checklist](#7-pre-flight-checklist)
8. [Release Sequence (Recommended Order)](#8-release-sequence-recommended-order)

---

## 1. Overview: What Goes Where

| Destination | What Ships | Contains Source? | License |
|-------------|-----------|-----------------|---------|
| **HuggingFace Hub** | `model.safetensors`, `config.json`, tokenizer files, model card | No Rust source | Apache 2.0 (weights) |
| **PyPI** (`pip install said-lam`) | Compiled `.so`/`.pyd`/`.dylib` binary + Python wrapper | No — wheels only, no sdist | Proprietary (code) |
| **GitHub Releases** | Pre-compiled `.so` binaries per platform (CPU + GPU) | No — stripped binaries | Proprietary (code) |
| **MTEB repo** (PR) | `ModelMeta` + loader function (Python file) | Minimal Python wrapper | Apache 2.0 |
| **MTEB results repo** (PR) | JSON result files per benchmark task | No code | N/A |

### Proprietary source: two-repo model (aligned with your setup)

Your **original code is proprietary**. You do **not** submit the full `said-lam` folder to a public repo. Only **compiled outputs** go to PyPI; a **separate, public repo** holds only the shareable parts.

| Location | What it is | What it contains |
|----------|------------|-------------------|
| **Private (your machine or private repo)** | Full codebase | `src/` (Rust), `docs/`, `scripts/`, `Cargo.toml`, `said_lam/`, `lam/`, `pyproject.toml`, etc. You **build** here and **upload wheels to PyPI** from here (or from private CI). |
| **PyPI** | Package users install | **Only** the built **wheels** (each wheel contains the compiled `.so` + Python wrapper). No source, no sdist. |
| **Public GitHub repo** (e.g. SAIDResearch/SAID-LAM) | “Other information” only | Everything **except** proprietary parts: Python wrapper (`said_lam/`, `lam/`), `pyproject.toml`, `README.md`, `LICENSE`, optional `tests/` or `hf_model_card/`. **No** `src/`, **no** `docs/`, **no** `scripts/`, **no** `Cargo.toml` (so the public repo **cannot** build the extension). |

So: **build the `.so` (wheels) in private → upload only wheels to PyPI.** Then create a **new** public repo with only the shareable content; that repo does not build the binary, it just documents the API and points users to `pip install said-lam`.

### Step-by-step: Get the private repo uploaded so you can start wheel builds

Follow these steps once so your **full** (proprietary) tree lives in a **private** GitHub repo and the wheel-build workflow can run there.

**Step 1 — Create a private GitHub repo**

- Go to [GitHub New Repository](https://github.com/new) (or your org: `https://github.com/organizations/YOUR_ORG/repositories/new`).
- Repository name: e.g. `SAID-LAM-private` or `said-lam`.
- Set visibility to **Private**.
- Do **not** add a README, .gitignore, or license (you already have them locally).
- Click **Create repository**.

**Step 2 — Push your full said-lam tree to it**

From your machine, in the directory that contains your **full** said-lam project (including `src/`, `docs/`, `scripts/`, `Cargo.toml`, `.github/workflows/release.yml`):

```bash
cd /path/to/said-lam

# If this is not yet a git repo
git init
git add .
# Project .gitignore already ignores target/, dist/, __pycache__/, .venv/
git commit -m "Private repo: full tree for wheel builds"

# Add the private repo as origin (replace with your repo URL)
git remote add origin https://github.com/YOUR_ORG/SAID-LAM-private.git
 git remote set-url origin "https://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@github.com/SAIDResearch/SAID-LAM-private.git"
# Or SSH: git remote add origin git@github.com:YOUR_ORG/SAID-LAM-private.git

git branch -M main
git push -u origin main
```

Your private code is now uploaded; only people with access to that repo can see it.

**Step 3 — Add PyPI token (so CI can publish wheels)**

- On PyPI: [Account → API tokens](https://pypi.org/manage/account/token/) → Create token (scope: entire account or just this project).
- In GitHub: open your **private** repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**.
- Name: `PYPI_API_TOKEN`, Value: the token you copied from PyPI.
- Save.

**Step 4 — Trigger the wheel builds**

You can either use a version tag (and publish to PyPI) or run the workflow manually without publishing.

**Option A — Build and publish to PyPI (push a tag):**

```bash
cd /path/to/said-lam
# Bump version in Cargo.toml and pyproject.toml if needed, then:
git add Cargo.toml pyproject.toml
git commit -m "Release v1.0.0"
git tag v1.0.0
git push origin main --tags
```

The `release.yml` workflow will run: build CPU/CUDA/Metal wheels, run smoke tests, publish to PyPI, and (if configured) create a GitHub Release with assets.

**Option B — Build only (no PyPI publish):**

- In the private repo: **Actions** → **Build and Release** → **Run workflow** → choose branch `main` → **Run workflow**.

This runs the build and smoke-test jobs only. The publish job will be skipped if the run was not triggered by a tag (see `if: startsWith(github.ref, 'refs/tags/')` in the workflow).

After this, your private repo is uploaded and you can start wheel builds either by pushing a tag (Option A) or by running the workflow manually (Option B).

---

## 2. Step 1: HuggingFace Hub — Model Weights

### What to Upload

The `hf_model_card/` directory (already prepared at `said-lam/hf_model_card/`):

```
hf_model_card/
├── README.md                  # Model card (YAML frontmatter + description)
├── config.json                # Model configuration (384 dim, 6 layers, 23.85M params)
├── model.safetensors          # Model weights (~92 MB)
├── tokenizer.json             # Tokenizer vocabulary
├── tokenizer_config.json      # Tokenizer settings
├── vocab.txt                  # WordPiece vocabulary
├── special_tokens_map.json    # Special tokens
├── scaling_plot.png           # (optional) Scaling comparison chart
└── stsb_scatter.png           # (optional) STS-B correlation plot
```

### Upload Commands

```bash
# 1. Install and login
pip install huggingface_hub
hf auth login   # Enter token from https://huggingface.co/settings/tokens

# 2. Create repo (first time only) — use full repo ID (org/repo-name)
hf repos create SAIDResearch/SAID-LAM-v1 --type model

# 3. Upload all files
cd said-lam/
hf upload SAIDResearch/SAID-LAM-v1 ./hf_model_card/ .

# 4. Verify
hf models info SAIDResearch/SAID-LAM-v1
# If "card_data" is {} in the output, that's normal — the Hub may populate it after
# processing the README; your README.md YAML frontmatter is what matters for the model card.
```

### Python API Alternative

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("SAIDResearch/SAID-LAM-v1", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="./hf_model_card",
    repo_id="SAIDResearch/SAID-LAM-v1",
    repo_type="model",
)
```

### Model Card README.md Requirements

The YAML frontmatter MUST include:

```yaml
---
license: apache-2.0
language:
  - en
library_name: said-lam
tags:
  - embeddings
  - linear-attention
  - mteb
  - sentence-similarity
  - retrieval
pipeline_tag: sentence-similarity
model-index:
  - name: SAID-LAM-v1
    results: []   # MTEB results go here after benchmarking
---
```

### Verify model is live on Hugging Face

```python
# Optional: confirm the model repo is online and has required files
from huggingface_hub import list_repo_files
files = list_repo_files("SAIDResearch/SAID-LAM-v1")
assert "model.safetensors" in files and "config.json" in files
print("Model is live on HF:", "model.safetensors" in files)
```

### Verify Auto-Download Works

```python
from said_lam import LAM
model = LAM("SAIDResearch/SAID-LAM-v1")  # Downloads from HF if not cached locally
emb = model.encode(["Hello world"])
print(f"Shape: {emb.shape}")  # (1, 384)
```

---

## 3. Step 2: PyPI — pip install said-lam

### How the Package Works

- `pip install said-lam` installs a **pre-compiled wheel** (no source)
- The wheel contains: `lam_candle.so` (Rust binary) + `said_lam/__init__.py` + `lam/__init__.py`
- Model weights are NOT bundled — downloaded from HuggingFace at first use
- No sdist (source distribution) is published — Rust source stays private

### Version Sync Requirement

Both files MUST have the same version:

| File | Field |
|------|-------|
| `Cargo.toml` | `version = "1.0.0"` |
| `pyproject.toml` | `version = "1.0.0"` |

Maturin uses `Cargo.toml` for the wheel filename. Mismatched versions will cause confusion.

### Manual Upload (Single Platform)

```bash
cd said-lam/

# Build wheel for your current platform
maturin build --release --out dist/

# Upload to TestPyPI first (recommended)
pip install twine
twine upload --repository testpypi dist/*

# Test from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            said-lam

# If good, upload to real PyPI
twine upload dist/*
```

### Create GitHub repository (SAIDResearch) — shareable content only

Create a **new** repo under [https://github.com/SAIDResearch](https://github.com/SAIDResearch) and push **only** the parts you want to share. Do **not** push `src/`, `docs/`, `scripts/`, or `Cargo.toml` (proprietary / required only for building the `.so`).

**Include in the public repo (examples):**

- `said_lam/` — Python wrapper
- `lam/` — Python package
- `pyproject.toml` — package metadata (for reference; the public repo will not build the extension)
- `README.md`, `LICENSE`
- Optional: `tests/` (if they don’t reveal proprietary logic), `hf_model_card/` (model card), `compare_backends.py`
- Optional: `.github/workflows/` only if you have a workflow that **does not** build from source (e.g. one that only attaches pre-built wheels to a Release). Do **not** run `maturin build` in the public repo (no Rust source there).

**Exclude from the public repo:**

- `src/` — Rust source (proprietary)
- `docs/` — internal docs (proprietary)
- `scripts/` — internal scripts (proprietary)
- `Cargo.toml`, `Cargo.lock` — Rust build (needed only where you build the wheel)
- `target/` — build artifacts

**One-time setup from your private tree:**

```bash
# 1. Create the repo on GitHub (empty, no README/license)
#    https://github.com/organizations/SAIDResearch/repositories/new
#    Name: SAID-LAM. Visibility: Public (or as needed).

# 2. Create a clean directory for the public repo and copy only shareable content
mkdir -p /tmp/said-lam-public && cd /tmp/said-lam-public
git init

cp -r /path/to/said-lam/said_lam .
cp -r /path/to/said-lam/lam .
cp /path/to/said-lam/pyproject.toml /path/to/said-lam/README.md /path/to/said-lam/LICENSE .
# Optional: cp -r /path/to/said-lam/tests . ; cp -r /path/to/said-lam/hf_model_card .
# Do NOT copy: src/ docs/ scripts/ Cargo.toml target/

git add .
git commit -m "Initial commit: SAID-LAM public package (Python API; install via pip install said-lam)"

# 3. Push to SAIDResearch
git remote add origin https://github.com/SAIDResearch/SAID-LAM.git
git branch -M main
git push -u origin main
```

After the first push, the repo is live at **https://github.com/SAIDResearch/SAID-LAM**. Users install the compiled package with `pip install said-lam` (wheels are published from your **private** build, not from this repo).

### Building wheels and uploading to PyPI (private side)

Because the **public** repo has no `src/` or `Cargo.toml`, the wheel is built only in your **private** copy of the project (or private CI).

**Option A — Build locally and upload to PyPI:**

```bash
cd /path/to/said-lam   # your full, private tree (with src/, Cargo.toml, etc.)
maturin build --release   # or per-platform as in "Building Locally" below
twine upload dist/*.whl
```

**Option B — Automated build in a private repo:**

If your **private** repo has the full tree (including `src/`, `Cargo.toml`, `.github/workflows/release.yml`), you can run the same workflow there: push a version tag and let private GitHub Actions build and publish to PyPI. In that case:

```bash
# In your private repo
git add Cargo.toml pyproject.toml
git commit -m "Release v1.0.0"
git tag v1.0.0
git push origin main --tags
```

Then the workflow (in the **private** repo) will:
1. Build 15 CPU wheels (5 platforms × 3 Python versions)
2. Build 1 CUDA wheel (Linux x86_64)
3. Build 1 Metal wheel (macOS Apple Silicon)
4. Run smoke tests
5. Publish wheels to PyPI (and optionally create a GitHub Release with assets in the **private** repo)

### What PyPI Receives

```
PyPI: said-lam 1.0.0
├── said_lam-1.0.0-cp310-cp310-manylinux_2_17_x86_64.whl     # Linux x86_64
├── said_lam-1.0.0-cp311-cp311-manylinux_2_17_x86_64.whl
├── said_lam-1.0.0-cp312-cp312-manylinux_2_17_x86_64.whl
├── said_lam-1.0.0-cp310-cp310-manylinux_2_17_aarch64.whl    # Linux ARM64
├── said_lam-1.0.0-cp311-cp311-manylinux_2_17_aarch64.whl
├── said_lam-1.0.0-cp312-cp312-manylinux_2_17_aarch64.whl
├── said_lam-1.0.0-cp310-cp310-macosx_10_12_x86_64.whl       # macOS Intel
├── said_lam-1.0.0-cp311-cp311-macosx_10_12_x86_64.whl
├── said_lam-1.0.0-cp312-cp312-macosx_10_12_x86_64.whl
├── said_lam-1.0.0-cp310-cp310-macosx_11_0_arm64.whl         # macOS Apple Silicon
├── said_lam-1.0.0-cp311-cp311-macosx_11_0_arm64.whl
├── said_lam-1.0.0-cp312-cp312-macosx_11_0_arm64.whl
├── said_lam-1.0.0-cp310-none-win_amd64.whl                  # Windows x86_64
├── said_lam-1.0.0-cp311-none-win_amd64.whl
├── said_lam-1.0.0-cp312-none-win_amd64.whl
├── said_lam-1.0.0-cp311-cp311-manylinux_2_17_x86_64.whl     # CUDA (same tag, different binary)
└── said_lam-1.0.0-cp311-cp311-macosx_11_0_arm64.whl         # Metal (same tag, different binary)
```

> **Note on GPU wheels**: PyPI does not distinguish GPU vs CPU wheels by filename.
> Current strategy: CPU wheels on PyPI (universal), GPU users build from source locally.
> Alternative: Separate `said-lam-cuda` package (requires separate config).

### Verify Installation

```bash
pip install said-lam
python -c "
from lam_candle import LamEngine, TIER_FREE, TIER_BETA
engine = LamEngine()
engine.index('test', 'Hello world')
assert engine.doc_count() == 1
print('Import + basic test: OK')
"
```

---

## 4. Step 3: GitHub Releases — Compiled .so Binaries

### Purpose

GitHub Releases provide:
- Direct download of platform-specific compiled binaries (`.so`, `.dylib`, `.pyd`)
- Version-tagged archives for reproducibility
- An alternative to PyPI for users who need raw binaries

### Release Workflow

The same `.github/workflows/release.yml` handles this. When you create a GitHub Release:

```bash
# Option A: Create release via CLI
gh release create v1.0.0 \
    --title "SAID-LAM v1.0.0" \
    --notes "Initial release — 23.85M parameter LAM with O(n) linear attention" \
    dist/*.whl

# Option B: Create release via GitHub UI
# Go to: https://github.com/SAIDResearch/SAID-LAM/releases/new
# Fill in tag (v1.0.0), title, and description
# Attach wheel files as assets
```

### Build Matrix — .so Files per Platform

| Platform | Target | Binary Name | GPU |
|----------|--------|-------------|-----|
| Linux x86_64 | `x86_64-unknown-linux-gnu` | `lam_candle.cpython-3{10,11,12}-x86_64-linux-gnu.so` | CPU |
| Linux x86_64 | `x86_64-unknown-linux-gnu` | `lam_candle.cpython-311-x86_64-linux-gnu.so` | CUDA 12.1 |
| Linux ARM64 | `aarch64-unknown-linux-gnu` | `lam_candle.cpython-3{10,11,12}-aarch64-linux-gnu.so` | CPU |
| macOS Intel | `x86_64-apple-darwin` | `lam_candle.cpython-3{10,11,12}-darwin.so` | CPU |
| macOS Apple Silicon | `aarch64-apple-darwin` | `lam_candle.cpython-3{10,11,12}-darwin.so` | CPU |
| macOS Apple Silicon | `aarch64-apple-darwin` | `lam_candle.cpython-311-darwin.so` | Metal |
| Windows x86_64 | `x86_64-pc-windows-msvc` | `lam_candle.cp3{10,11,12}-win_amd64.pyd` | CPU |

### Building Locally for Each Platform

```bash
cd said-lam/

# CPU (current platform, auto-detected)
maturin build --release --out dist/

# CUDA (Linux x86_64 only — requires CUDA Toolkit 12.x)
maturin build --release --features cuda --out dist/

# Metal (macOS Apple Silicon only — requires Xcode CLT)
maturin build --release --features metal --out dist/

# Cross-compile Linux ARM64 (from Linux x86_64)
sudo apt install gcc-aarch64-linux-gnu
rustup target add aarch64-unknown-linux-gnu
maturin build --release --target aarch64-unknown-linux-gnu --out dist/

# Cross-compile macOS (requires macOS SDK — easier via CI)
rustup target add x86_64-apple-darwin aarch64-apple-darwin
maturin build --release --target x86_64-apple-darwin --out dist/
maturin build --release --target aarch64-apple-darwin --out dist/
```

### Binary Protection

All binaries are built with maximum protection:

```toml
# Cargo.toml [profile.release]
opt-level = 3        # Maximum optimization
lto = true           # Link-Time Optimization (harder to reverse-engineer)
codegen-units = 1    # Single compilation unit
strip = true         # Strip ALL debug symbols
```

### Attaching Binaries to a GitHub Release

After CI/CD builds complete, artifacts are uploaded automatically. For manual attachment:

```bash
# Download CI artifacts
gh run download <run-id> --dir ./release-artifacts/

# Create release with all binaries attached
gh release create v1.0.0 \
    --title "SAID-LAM v1.0.0" \
    --notes-file RELEASE_NOTES.md \
    ./release-artifacts/**/*.whl
```

---

## 5. Step 4: MTEB Leaderboard — Model Wrapper PR + Results PR

### CRITICAL: Both PRs Are Required

The old workflow of adding YAML to a HuggingFace model card has been **deprecated**. You now MUST submit:

1. **Model Wrapper PR** → `github.com/embeddings-benchmark/mteb`
2. **Results PR** → `github.com/embeddings-benchmark/results`

### Part A: Model Wrapper PR to embeddings-benchmark/mteb

#### File to Create: `mteb/models/said_lam.py`

This file already exists in the project at `said-lam/said_lam/mteb_encoder.py`. For the MTEB PR, create a standalone version at `mteb/models/said_lam.py`:

```python
"""SAID-LAM: Linear Attention Memory — 384-dim embeddings, O(n) complexity."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import (
        Array, BatchedInput, CorpusDatasetType, EncodeKwargs,
        PromptType, QueryDatasetType, RetrievalOutputType, TopRankedDocumentsType,
    )

logger = logging.getLogger(__name__)


def said_lam_loader(
    model_name: str, revision: str | None = None, device: str | None = None, **kwargs
) -> "LAM":
    from said_lam import LAM
    return LAM(model_name, device=device, **kwargs)


class LAM(AbsEncoder):
    """SAID-LAM encoder for MTEB — implements EncoderProtocol + SearchProtocol."""

    LONGEMBED_TASKS = {
        'lembneedleretrieval', 'lembpasskeyretrieval',
        'lembnarrativeqaretrieval', 'lembqmsumretrieval',
        'lembwikimqaretrieval', 'lembsummscreenfdretrieval',
    }

    def __init__(self, model_name="SAIDResearch/SAID-LAM-v1", revision=None, device=None, **kwargs):
        self.model_name = model_name
        self.revision = revision or "main"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._embedding_dim = 384
        self._engine = None
        self._corpus_ids = []
        self._corpus_texts = []
        self._corpus_embeddings = None
        self._current_task = ""
        self.mteb_model_meta = said_lam_v1
        self._init_model(model_name)

    def _init_model(self, model_name):
        import lam_candle
        from said_lam import LAM
        lam = LAM(model_name, backend="crystalline")
        self._engine = lam._engine
        if hasattr(self._engine, 'auto_activate_mteb'):
            self._engine.auto_activate_mteb()

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        texts = []
        for batch in inputs:
            if isinstance(batch, dict) and "text" in batch:
                batch_texts = batch["text"]
                texts.extend(batch_texts if isinstance(batch_texts, list) else [str(batch_texts)])
            elif isinstance(batch, (list, tuple)):
                texts.extend([str(s) for s in batch])
            else:
                texts.append(str(batch))
        if not texts:
            return np.array([]).reshape(0, self._embedding_dim)
        with torch.no_grad():
            embs = self._engine.encode(texts, True, 32, None, False)
            return np.array(embs, dtype=np.float32)

    def index(self, corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs={}, num_proc=None):
        self._current_task = str(getattr(task_metadata, 'name', '')).lower()
        self._corpus_ids, self._corpus_texts = [], []
        self._corpus_embeddings = None
        for doc in corpus:
            self._corpus_ids.append(str(doc["id"]))
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            self._corpus_texts.append(f"{title} {text}".strip() if title else text)
        self._engine.index_mteb(self._corpus_ids, self._corpus_texts, self._current_task, None)

    def search(self, queries, *, task_metadata, hf_split, hf_subset, top_k,
               encode_kwargs={}, top_ranked=None, num_proc=None):
        task = str(getattr(task_metadata, 'name', '')).lower()
        qids = list(queries["id"])
        from mteb._create_dataloaders import _create_text_queries_dataloader
        queries_loader = _create_text_queries_dataloader(queries)
        qtexts = [text for batch in queries_loader for text in batch["text"]]
        raw = self._engine.search_mteb(qids, qtexts, task, top_k, None)
        return {qid: {did: float(s) for did, s in docs.items()} for qid, docs in raw.items()}


# ModelMeta registration
said_lam_v1 = ModelMeta(
    name="SAIDResearch/SAID-LAM-v1",
    revision="main",
    release_date="2026-01-01",
    languages=["eng"],
    loader=said_lam_loader,
    n_parameters=23_848_788,
    memory_usage_mb=90,
    max_tokens=32768,
    embed_dim=384,
    license="apache-2.0",
    open_weights=True,
    adapted_from="sentence-transformers/all-MiniLM-L6-v2",
    modalities=["text"],
    framework=["PyTorch"],
    reference="https://saidhome.ai",
    similarity_fn_name="cosine",
    use_instructions=False,
)
```

#### Register in `mteb/models/overview.py`

Add to the imports:

```python
from mteb.models.said_lam import said_lam_v1
```

#### PR Checklist (MTEB requires all of these)

- [ ] `ModelMeta` fully filled out (name, revision, languages, n_parameters, embed_dim, etc.)
- [ ] Model loads via `mteb.get_model("SAIDResearch/SAID-LAM-v1")`
- [ ] Model loads via `mteb.get_model_meta("SAIDResearch/SAID-LAM-v1")`
- [ ] Implementation tested on representative tasks (STS12, LEMBNeedleRetrieval, etc.)
- [ ] Model is publicly accessible (`pip install said-lam` + weights on HuggingFace)
- [ ] Training data contamination annotated (if applicable)
- [ ] PR description includes: model card link, benchmark results, architecture summary

#### How to Submit the PR

```bash
# 1. Fork the MTEB repo
gh repo fork embeddings-benchmark/mteb --clone

# 2. Create branch
cd mteb
git checkout -b add-said-lam-v1

# 3. Add the model file
cp /path/to/said_lam.py mteb/models/said_lam.py

# 4. Register in overview.py
# Add import line to mteb/models/overview.py

# 5. Test locally
python -c "
import mteb
model = mteb.get_model('SAIDResearch/SAID-LAM-v1')
meta = mteb.get_model_meta('SAIDResearch/SAID-LAM-v1')
print(f'Model: {meta.name}, Params: {meta.n_parameters}, Dim: {meta.embed_dim}')
"

# 6. Create PR
git add mteb/models/said_lam.py mteb/models/overview.py
git commit -m "Add SAIDResearch/SAID-LAM-v1 model"
gh pr create \
    --title "Add SAIDResearch/SAID-LAM-v1 (23.85M param Linear Attention Model)" \
    --body "## Model: SAID-LAM-v1

- **Architecture**: 6-layer Hierarchical DeltaNet (Linear Attention, NOT a transformer)
- **Parameters**: 23.85M (384-dim embeddings)
- **Complexity**: O(n) linear — time AND memory
- **Max context**: 32,768 tokens
- **Framework**: Pure Rust (Candle) + Python wrapper via PyO3
- **Base model**: Distilled from sentence-transformers/all-MiniLM-L6-v2
- **Install**: \`pip install said-lam\`
- **HuggingFace**: https://huggingface.co/SAIDResearch/SAID-LAM-v1

### Key Results
- 100% Recall on LEMBNeedleRetrieval and LEMBPasskeyRetrieval
- 0.836 Pearson on STS-B (competitive with transformers at O(n))
- Implements both EncoderProtocol and SearchProtocol
"
```

### Part B: Run MTEB Evaluations

After the model wrapper PR is submitted (can run in parallel):

```python
import mteb
from said_lam import LAM

model = LAM("SAIDResearch/SAID-LAM-v1")

# Run specific benchmarks
tasks = mteb.get_tasks(tasks=[
    # LongEmbed (your strongest results)
    "LEMBNeedleRetrieval",
    "LEMBPasskeyRetrieval",
    "LEMBNarrativeQARetrieval",
    "LEMBWikiMQARetrieval",
    "LEMBQMSumRetrieval",
    "LEMBSummScreenFDRetrieval",
    # STS (semantic similarity)
    "STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark",
    # Classification
    "Banking77Classification",
    # Or run the full English benchmark:
    # tasks = mteb.get_benchmark("MTEB(eng, classic)")
])

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results/SAIDResearch__SAID-LAM-v1")
```

CLI alternative:
```bash
mteb run -m SAIDResearch/SAID-LAM-v1 -t LEMBNeedleRetrieval --output-folder results
```

Results are saved as:
```
results/
  SAIDResearch__SAID-LAM-v1/
    main/                          # revision
      model_meta.json
      LEMBNeedleRetrieval.json
      LEMBPasskeyRetrieval.json
      STSBenchmark.json
      ...
```

### Part C: Results PR to embeddings-benchmark/results

```bash
# 1. Fork the results repo
gh repo fork embeddings-benchmark/results --clone

# 2. Copy results
cd results
cp -r /path/to/results/SAIDResearch__SAID-LAM-v1 .

# 3. Create PR
git checkout -b add-said-lam-v1-results
git add SAIDResearch__SAID-LAM-v1/
git commit -m "Add SAID-LAM-v1 results"
gh pr create \
    --title "Add SAIDResearch/SAID-LAM-v1 results" \
    --body "Results for SAID-LAM-v1 (23.85M param Linear Attention Model).

Model wrapper PR: embeddings-benchmark/mteb#XXXX

Key results:
- LEMBNeedleRetrieval: 100% NDCG@10, 100% Recall@1
- LEMBPasskeyRetrieval: 100% NDCG@10, 100% Recall@1
- STSBenchmark: 0.836 Pearson
"
```

---

## 6. Licensing Summary

### Dual License Structure

| Component | License | File | Allows |
|-----------|---------|------|--------|
| **Model weights** (`model.safetensors`) | Apache 2.0 | `LICENSE-MODEL` | Use, modify, distribute, commercial |
| **Software** (Rust binary, Python wrapper) | Proprietary | `LICENSE` | Use, install, integrate — NO reverse-engineering, NO redistribution |

### What This Means for Each Submission

| Destination | License Shown | Reasoning |
|-------------|--------------|-----------|
| HuggingFace | `apache-2.0` (in YAML frontmatter) | Only weights + config are hosted |
| PyPI | `Proprietary` (in pyproject.toml classifiers) | Compiled binary code is distributed |
| GitHub | Both `LICENSE` + `LICENSE-MODEL` in repo | Both components live here |
| MTEB ModelMeta | `license="apache-2.0"` | MTEB cares about weight availability |

### Source Code Protection Chain

```
Rust source (src/*.rs)
    → NOT published anywhere (private repo only)
    → Compiled by Maturin with strip=true, lto=true
    → Output: lam_candle.so (no debug symbols, optimized, obfuscated via LTO)
    → Distributed as: pre-compiled wheels on PyPI (NO sdist)
```

---

## 7. Pre-Flight Checklist

### Before HuggingFace Upload
- [ ] `hf_model_card/` has all 7+ files (README.md, config.json, model.safetensors, tokenizer.json, tokenizer_config.json, vocab.txt, special_tokens_map.json)
- [ ] `config.json` has `n_parameters: 23848788`
- [ ] `README.md` has `license: apache-2.0` in YAML frontmatter
- [ ] Model card includes architecture description, benchmarks, usage examples
- [ ] HuggingFace token created with write access

### Before PyPI Upload
- [ ] `Cargo.toml` version matches `pyproject.toml` version
- [ ] All tests pass: `cd tests && python run_all_tests.py --skip-pearson`
- [ ] Smoke test passes on at least 2 platforms
- [ ] PyPI API token stored as `PYPI_API_TOKEN` in GitHub Secrets
- [ ] TestPyPI upload tested first (tag with `-rc1` suffix)
- [ ] `said_lam/__init__.py` has correct `__version__`

### Before GitHub Release
- [ ] Git tag matches package version: `v1.0.0`
- [ ] Release notes describe changes since last version
- [ ] CI/CD workflow completes successfully for all 7 platform targets
- [ ] All wheel artifacts are attached to the release

### Before MTEB Submission
- [ ] HuggingFace repo is **public** and accessible
- [ ] `pip install said-lam` works on a clean machine
- [ ] `mteb.get_model("SAIDResearch/SAID-LAM-v1")` returns working encoder
- [ ] Evaluations run successfully on target benchmarks
- [ ] Result JSON files validate (correct schema, mteb_version, dataset_revision)
- [ ] Model wrapper PR references HuggingFace card URL

---

## 8. Release Sequence (Recommended Order)

Execute in this order — each step depends on the previous:

### Phase 1: Weights First
```
1. Upload model weights to HuggingFace Hub
   → Users can now: huggingface_hub.snapshot_download("SAIDResearch/SAID-LAM-v1")
   → The pip package will auto-download from here
```

### Phase 2: Package Next
```
2. Test on TestPyPI
   git tag v1.0.0-rc1
   git push origin v1.0.0-rc1
   → CI builds + publishes to TestPyPI
   → Test: pip install --index-url https://test.pypi.org/simple/ said-lam

3. Publish to real PyPI
   git tag v1.0.0
   git push origin v1.0.0
   → CI builds + publishes to PyPI
   → Test: pip install said-lam

4. Create GitHub Release
   gh release create v1.0.0 --title "SAID-LAM v1.0.0" --generate-notes
   → Attach wheel artifacts from CI
```

### Phase 3: MTEB Last (requires Phase 1 + 2)
```
5. Submit Model Wrapper PR to embeddings-benchmark/mteb
   → Reviewers will test: pip install said-lam && mteb.get_model(...)

6. Run MTEB evaluations (while wrapper PR is in review)
   → Save results to results/ folder

7. Submit Results PR to embeddings-benchmark/results
   → Reference the model wrapper PR number

8. After both PRs are merged → results appear on MTEB leaderboard
```

### Timeline Estimate

| Step | Dependencies | Can Parallelize? |
|------|-------------|------------------|
| HuggingFace upload | None | — |
| TestPyPI | HuggingFace (for auto-download test) | — |
| PyPI | TestPyPI verified | — |
| GitHub Release | PyPI tag | Yes, with MTEB eval |
| MTEB wrapper PR | HuggingFace + PyPI | Yes, with MTEB eval |
| MTEB evaluation | HuggingFace + PyPI | Yes, with wrapper PR |
| MTEB results PR | Evaluation complete | After eval |

---

## Appendix: Quick Reference Commands

```bash
# === HuggingFace ===
hf upload SAIDResearch/SAID-LAM-v1 ./hf_model_card/ .

# === PyPI (automated) ===
git tag v1.0.0 && git push origin v1.0.0

# === PyPI (manual) ===
maturin build --release --out dist/ && twine upload dist/*

# === GitHub Release ===
gh release create v1.0.0 --title "SAID-LAM v1.0.0" --generate-notes

# === MTEB Evaluation ===
mteb run -m SAIDResearch/SAID-LAM-v1 -t LEMBNeedleRetrieval --output-folder results

# === Local Build (dev) ===
maturin develop --release                    # CPU
maturin develop --release --features cuda    # GPU/CUDA
maturin develop --release --features metal   # GPU/Metal

# === Verify everything works ===
python -c "from said_lam import LAM; m = LAM(); print(m.encode(['test']).shape)"
```
