# HuggingFace Publishing Guide

Publish SAID-LAM-v1 to HuggingFace Hub for auto-download — **without sharing Rust source code**.

## Distribution Strategy

| What Ships | Where | Contains Source? |
|------------|-------|-----------------|
| **PyPI** (`pip install said-lam`) | Compiled `.so`/`.pyd` binary + Python wrapper | **No** |
| **HuggingFace** (model weights) | `model.safetensors`, `config.json`, tokenizer files | **No** |
| **GitHub** (optional) | Python wrapper + build config (NOT Rust source) | **No** |

The Rust source code (`src/*.rs`) stays private. Users get:
- Pre-compiled binary via `pip install said-lam`
- Model weights auto-downloaded from HuggingFace at first use

---

## Step 1: Prepare HuggingFace Repository

The `hf_model_card/` directory contains everything needed:

```
hf_model_card/
├── README.md                  # Model card (shown on HF page)
├── config.json                # Model configuration
├── model.safetensors          # Model weights (92 MB)
├── tokenizer.json             # Tokenizer vocabulary
├── tokenizer_config.json      # Tokenizer settings
├── vocab.txt                  # WordPiece vocabulary
└── special_tokens_map.json    # Special tokens
```

---

## Step 2: Create HuggingFace Repository

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login
huggingface-cli login
# Enter your HF token from https://huggingface.co/settings/tokens

# Create the repository
hf repos create SAIDResearch/SAID-LAM-v1 --type model
```

---

## Step 3: Upload Model Files

```bash
# Upload all files from hf_model_card/
hf upload SAIDResearch/SAID-LAM-v1 ./hf_model_card/ .

# Or upload individual files:
hf upload SAIDResearch/SAID-LAM-v1 ./hf_model_card/README.md README.md
hf upload SAIDResearch/SAID-LAM-v1 ./hf_model_card/config.json config.json
hf upload SAIDResearch/SAID-LAM-v1 ./hf_model_card/model.safetensors model.safetensors
hf upload SAIDResearch/SAID-LAM-v1 ./hf_model_card/tokenizer.json tokenizer.json
hf upload SAIDResearch/SAID-LAM-v1 ./hf_model_card/tokenizer_config.json tokenizer_config.json
hf upload SAIDResearch/SAID-LAM-v1 ./hf_model_card/vocab.txt vocab.txt
hf upload SAIDResearch/SAID-LAM-v1 ./hf_model_card/special_tokens_map.json special_tokens_map.json
```

### Alternative: Using Python API

```python
from huggingface_hub import HfApi

api = HfApi()

# Create repo (if not exists)
api.create_repo("SAIDResearch/SAID-LAM-v1", repo_type="model", exist_ok=True)

# Upload entire directory
api.upload_folder(
    folder_path="./hf_model_card",
    repo_id="SAIDResearch/SAID-LAM-v1",
    repo_type="model",
)
```

---

## Step 4: Verify Upload

```bash
# Check repository contents
hf models info SAIDResearch/SAID-LAM-v1

# Test auto-download works
python -c "
from said_lam import LAM
model = LAM('SAIDResearch/SAID-LAM-v1')
emb = model.encode(['Hello world'])
print(f'Embedding shape: {emb.shape}')
print('SUCCESS: Auto-download from HuggingFace works!')
"
```

---

## Step 5: Publish to PyPI

```bash
cd said-lam/

# Build wheel for your platform
maturin build --release --out dist/

# Upload to PyPI
pip install twine
twine upload dist/*.whl

# Or use maturin directly
maturin publish --username __token__ --password <PYPI_TOKEN>
```

### Automated Publishing (CI/CD)

Push a version tag to trigger automatic builds + PyPI publish:

```bash
git tag v1.0.0
git push origin v1.0.0
# GitHub Actions builds 8 wheel variants and publishes to PyPI
```

---

## What Users Experience

### Install
```bash
pip install said-lam
```

### First Use (auto-downloads weights from HuggingFace)
```python
from said_lam import LAM

model = LAM("SAIDResearch/SAID-LAM-v1")  # Downloads weights automatically
embeddings = model.encode(["Hello world"])
```

### What Happens Under the Hood
1. `pip install said-lam` installs the compiled `.so` binary + Python wrapper
2. `LAM("SAIDResearch/SAID-LAM-v1")` downloads weights from HuggingFace to `~/.cache/huggingface/`
3. The Rust engine loads `model.safetensors`, `config.json`, `tokenizer.json`
4. No Rust source code is ever exposed to users

---

## Source Code Protection

| Layer | What Ships | Source Visible? |
|-------|-----------|-----------------|
| **PyPI wheel** | `lam_candle.so` (compiled binary) | No — compiled Rust, stripped symbols |
| **HuggingFace** | `model.safetensors` + tokenizer | No source — weights only |
| **GitHub** (public) | `said_lam/__init__.py`, `pyproject.toml`, README | Python wrapper only |
| **Rust source** (`src/`) | Not distributed anywhere | Private |

The `Cargo.toml` release profile strips all debug info:
```toml
[profile.release]
strip = true         # Remove debug symbols
lto = true           # Link-Time Optimization (harder to reverse-engineer)
opt-level = 3        # Full optimization
codegen-units = 1    # Single compilation unit
```

---

## MTEB Leaderboard Submission

For submitting MTEB benchmark results without hosting weights publicly:

1. Run MTEB evaluation locally:
```python
from said_lam import LAM
import mteb

model = LAM("SAIDResearch/SAID-LAM-v1")
tasks = mteb.get_tasks(tasks=["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"])
results = mteb.evaluate(model=model, tasks=tasks)
```

2. The HuggingFace model card (`hf_model_card/README.md`) includes MTEB-compatible metadata tags
3. Results appear on the MTEB leaderboard once the HF repo is public

---

## Checklist

- [ ] `hf_model_card/` has all 7 files (README.md, config.json, model.safetensors, tokenizer.json, tokenizer_config.json, vocab.txt, special_tokens_map.json)
- [ ] `config.json` has correct `n_parameters: 23848788`
- [ ] `README.md` model card has `license: apache-2.0` in YAML frontmatter
- [ ] PyPI token configured as GitHub secret (`PYPI_API_TOKEN`)
- [ ] HuggingFace token configured for upload
- [ ] Test: `pip install said-lam && python -c "from said_lam import LAM; LAM()"`
- [ ] Rust source (`src/`) is NOT in any public repository
