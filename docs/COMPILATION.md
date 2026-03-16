# SAID-LAM Compilation Guide

Build the `said-lam` Python package from Rust source. The compiled binary (`lam_candle.so` / `lam_candle.pyd`) ships as a wheel — **no Rust source code is distributed**.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Use Case 1: Local Development](#use-case-1-local-development)
3. [Use Case 2: Build a Wheel for Your Machine](#use-case-2-build-a-wheel-for-your-machine)
4. [Use Case 3: Build All Platform Wheels (CI/CD)](#use-case-3-build-all-platform-wheels-cicd)
5. [Use Case 4: Publish to PyPI](#use-case-4-publish-to-pypi)
6. [Use Case 5: GPU Builds (CUDA / Metal)](#use-case-5-gpu-builds-cuda--metal)
7. [How pip Auto-Selects the Correct Wheel](#how-pip-auto-selects-the-correct-wheel)
8. [GitHub Actions: Step-by-Step Setup](#github-actions-step-by-step-setup)
9. [Testing Your Build](#testing-your-build)
10. [Platform Matrix](#platform-matrix)
11. [What Ships in the Wheel](#what-ships-in-the-wheel)
12. [Build Optimization](#build-optimization)
13. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| **Rust** | stable (1.75+) | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| **Python** | 3.8-3.12 | System or pyenv |
| **Maturin** | 1.4+ | `pip install maturin` |

### GPU Prerequisites (optional)

| Backend | Requirement |
|---------|-------------|
| **CUDA** (Linux) | CUDA Toolkit 12.x + `nvcc` in PATH |
| **Metal** (macOS) | Xcode Command Line Tools (pre-installed on macOS) |

---

## Use Case 1: Local Development

**Goal**: Build and install directly into your Python environment for testing.

### Option A: Virtual Environment (Recommended)

```bash
cd said-lam/

# Create and activate virtualenv
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# Build + install in one step
maturin develop --release

# Verify
python -c "from lam_candle import LamEngine; e = LamEngine(); print(e.get_tier_name())"
```

### Option B: Build Wheel + pip install (No venv required)

```bash
cd said-lam/

# Step 1: Build the wheel
maturin build --release --out dist/

# Step 2: Install it
pip install dist/said_lam-*.whl --force-reinstall

# Step 3: Verify (run from a different directory to avoid import confusion)
cd /tmp
python -c "
from lam_candle import LamEngine, TIER_FREE, TIER_BETA
engine = LamEngine()
print(f'Tier: {engine.get_tier_name()}')
print(f'Backend: {engine.get_backend()}')
print('Import successful')
"
```

---

## Use Case 2: Build a Wheel for Your Machine

**Goal**: Create a `.whl` file you can share or install on same-platform machines.

```bash
cd said-lam/

# Build for your current platform (auto-detected)
maturin build --release --out dist/

# The output will be something like:
#   dist/said_lam-1.0.0-cp311-cp311-manylinux_2_39_x86_64.whl
#
# The filename encodes: package-version-pythonversion-abi-platform

ls -lh dist/*.whl
```

### Build for a Specific Python Version

```bash
# Target Python 3.10 specifically
maturin build --release --interpreter python3.10 --out dist/

# Target multiple Python versions
maturin build --release --interpreter python3.9 python3.10 python3.11 python3.12 --out dist/
```

### Install the Wheel

```bash
# On your machine
pip install dist/said_lam-1.0.0-cp311-cp311-manylinux_2_39_x86_64.whl

# On another machine (same platform + Python version)
scp dist/said_lam-*.whl user@server:~/
ssh user@server "pip install ~/said_lam-*.whl"
```

---

## Use Case 3: Build All Platform Wheels (CI/CD)

**Goal**: Build wheels for every supported platform using GitHub Actions.

This is handled automatically by `.github/workflows/release.yml`. See [GitHub Actions: Step-by-Step Setup](#github-actions-step-by-step-setup) for the complete walkthrough.

### Manual Cross-Compilation (Advanced)

If you need to build for other platforms locally:

```bash
# Install cross-compilation targets
rustup target add aarch64-unknown-linux-gnu
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin

# Linux ARM64 (requires cross-compiler)
sudo apt install gcc-aarch64-linux-gnu
maturin build --release --target aarch64-unknown-linux-gnu --out dist/

# macOS targets (requires macOS SDK, easier via CI)
maturin build --release --target x86_64-apple-darwin --out dist/
maturin build --release --target aarch64-apple-darwin --out dist/
```

---

## Use Case 4: Publish to PyPI

**Goal**: Make `pip install said-lam` work for anyone on any platform.

### How It Works

When you upload multiple wheels to PyPI, pip automatically picks the right one:

```
PyPI: said-lam 1.0.0
  ├── said_lam-1.0.0-cp310-cp310-manylinux_2_17_x86_64.whl    (Linux x86_64, Python 3.10)
  ├── said_lam-1.0.0-cp311-cp311-manylinux_2_17_x86_64.whl    (Linux x86_64, Python 3.11)
  ├── said_lam-1.0.0-cp312-cp312-manylinux_2_17_x86_64.whl    (Linux x86_64, Python 3.12)
  ├── said_lam-1.0.0-cp3*-manylinux_2_17_aarch64.whl          (Linux ARM64)
  ├── said_lam-1.0.0-cp3*-macosx_10_12_x86_64.whl             (macOS Intel)
  ├── said_lam-1.0.0-cp3*-macosx_11_0_arm64.whl               (macOS Apple Silicon)
  └── said_lam-1.0.0-cp3*-none-win_amd64.whl                  (Windows x86_64)

User runs: pip install said-lam
  → pip checks OS + arch + Python version
  → Downloads matching pre-compiled wheel automatically
  → No source code is distributed — binary wheels only
```

> **No source fallback (sdist)**: We do NOT publish a source distribution.
> Rust source code is proprietary and never leaves the build server.
> If no wheel matches the user's platform, pip will show an error.
> To support a new platform, add it to the CI/CD matrix.

### Option A: Automated via GitHub Actions (Recommended)

```bash
# Step 1: Bump version in Cargo.toml and pyproject.toml
# Step 2: Commit and tag
git add -A
git commit -m "Release v1.0.1"
git tag v1.0.1
git push origin main --tags

# Step 3: GitHub Actions automatically:
#   - Builds 15 CPU wheels (5 platforms × 3 Python versions) + CUDA + Metal
#   - Uploads all pre-compiled wheels to PyPI (no source code published)
#   - Users can now: pip install said-lam
```

### Option B: Manual Upload

```bash
# Step 1: Build wheel(s) locally
maturin build --release --out dist/

# Step 2: Install twine
pip install twine

# Step 3: Upload to PyPI
twine upload dist/*

# Step 4: Upload to TestPyPI first (recommended for testing)
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ said-lam
```

---

## Use Case 5: GPU Builds (CUDA / Metal)

### CUDA (Linux x86_64)

```bash
# Requires: CUDA Toolkit 12.x with nvcc in PATH
nvcc --version  # Verify CUDA is available

# Build with CUDA support
maturin build --release --features cuda --out dist/

# The wheel filename will be the same as CPU — the CUDA code is compiled in.
# Users install the same way: pip install said_lam-*.whl
```

### Metal (macOS Apple Silicon)

```bash
# Requires: Xcode Command Line Tools (pre-installed on macOS)
xcode-select --version  # Verify

# Build with Metal GPU support
maturin build --release --features metal --out dist/
```

### GPU vs CPU: Which Gets Installed?

PyPI does **not** distinguish GPU wheels by filename (unlike PyTorch). Two strategies:

**Strategy 1: Separate package names** (e.g., `said-lam` for CPU, `said-lam-cuda` for GPU)
- Requires separate Cargo.toml/pyproject.toml per variant

**Strategy 2: CPU on PyPI, GPU via direct install** (current approach)
- PyPI has CPU wheels (works everywhere)
- GPU users install from a direct URL or build locally:
  ```bash
  # GPU users build locally:
  pip install said-lam  # Gets CPU wheel from PyPI
  # Then rebuild with GPU:
  cd said-lam && maturin develop --release --features cuda
  ```

---

## How pip Auto-Selects the Correct Wheel

When you run `pip install said-lam`, pip does the following:

1. **Queries PyPI** for all available files for `said-lam==1.0.0`
2. **Filters by compatibility**:
   - Python version: `cp311` matches Python 3.11
   - ABI tag: `cp311` matches CPython 3.11
   - Platform tag: `manylinux_2_17_x86_64` matches Linux x86_64
3. **Picks the best match** (prefers newer manylinux, exact Python version)
4. **Fails if no wheel matches** — no source distribution is published (proprietary Rust source)

### Platform Tags Explained

```
said_lam-1.0.0-cp311-cp311-manylinux_2_17_x86_64.whl
                │      │     │              │
                │      │     │              └─ CPU architecture
                │      │     └─ OS (Linux with glibc 2.17+)
                │      └─ ABI (CPython 3.11)
                └─ Python version (CPython 3.11)
```

### Ensuring All Platforms Are Covered

Upload wheels for every combination you want to support:

| User's Machine | Wheel They Get |
|----------------|----------------|
| Ubuntu 22.04, Python 3.11 | `*-cp311-cp311-manylinux_2_17_x86_64.whl` |
| macOS M2, Python 3.12 | `*-cp312-cp312-macosx_11_0_arm64.whl` |
| Windows 10, Python 3.10 | `*-cp310-none-win_amd64.whl` |
| Raspberry Pi, Python 3.11 | `*-cp311-cp311-manylinux_2_17_aarch64.whl` |
| Unsupported platform | `pip install` fails — request support via GitHub issue |

---

## GitHub Actions: Step-by-Step Setup

### Step 1: Create the Repository Structure

Your repository needs these files:

```
said-lam/
├── Cargo.toml              # Rust package config
├── pyproject.toml           # Python package config (build-backend = maturin)
├── src/
│   ├── lib.rs               # Rust entry point (PyO3 module)
│   ├── crystalline.rs       # SCA engine
│   ├── engine.rs            # LAM engine
│   ├── model.rs             # Neural network model
│   ├── sca_dropin.rs        # SCA drop-in search
│   └── storage.rs           # Persistence layer
├── said_lam/
│   └── __init__.py          # Python wrapper package
├── lam/
│   └── __init__.py          # Backwards-compatibility shim
├── weights/                 # Model weights (embedded in compiled binary)
└── .github/
    └── workflows/
        └── release.yml      # CI/CD pipeline
```

### Step 2: Set Up PyPI API Token

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token (scope: entire account or project-specific)
3. Copy the token (starts with `pypi-`)
4. In your GitHub repo: **Settings > Secrets and variables > Actions > New repository secret**
5. Name: `PYPI_API_TOKEN`, Value: paste the token

### Step 3: The Release Workflow

The workflow at `.github/workflows/release.yml` builds **pre-compiled wheels only** (no source distribution) across all platforms:

| Job | Platform | GPU | Python Versions | Runner |
|-----|----------|-----|-----------------|--------|
| build-cpu (1/5) | Linux x86_64 | CPU | 3.10, 3.11, 3.12 | `ubuntu-latest` |
| build-cpu (2/5) | Linux ARM64 | CPU | 3.10, 3.11, 3.12 | `ubuntu-latest` (cross-compile) |
| build-cpu (3/5) | macOS Intel | CPU | 3.10, 3.11, 3.12 | `macos-13` |
| build-cpu (4/5) | macOS Apple Silicon | CPU | 3.10, 3.11, 3.12 | `macos-latest` |
| build-cpu (5/5) | Windows x86_64 | CPU | 3.10, 3.11, 3.12 | `windows-latest` |
| build-cuda | Linux x86_64 | CUDA 12.1 | 3.11 | `ubuntu-latest` + CUDA container |
| build-metal | macOS Apple Silicon | Metal | 3.11 | `macos-latest` |

> **No sdist**: Source distributions are NOT published. Rust source is proprietary.

### Step 4: Trigger a Release

```bash
# Option A: Push a version tag (triggers build + publish)
git tag v1.0.0
git push origin v1.0.0

# Option B: Create a GitHub Release (triggers build + publish)
gh release create v1.0.0 --title "v1.0.0" --notes "Initial release"

# Option C: Manual trigger (triggers build only, no publish)
gh workflow run release.yml
```

### Step 5: Verify on PyPI

```bash
# After the workflow completes (~10 min):
pip install said-lam              # Should install the correct wheel
pip install said-lam --upgrade    # Force upgrade

# Verify
python -c "from lam_candle import LamEngine; print('OK')"
```

### Step 6: TestPyPI First (Recommended)

To test without publishing to the real PyPI, add a TestPyPI job:

```yaml
# Add to release.yml, after the publish job:
publish-test:
  name: Publish to TestPyPI
  needs: [build-cpu, build-cuda, build-metal]
  runs-on: ubuntu-latest
  if: startsWith(github.ref, 'refs/tags/v') && contains(github.ref, 'rc')

  steps:
    - name: Download all artifacts
      uses: actions/download-artifact@v4
      with:
        path: dist
        merge-multiple: true

    - name: Publish to TestPyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.TEST_PYPI_API_TOKEN }}
        repository-url: https://test.pypi.org/legacy/
        packages-dir: dist/
```

Then test with:
```bash
git tag v1.0.1-rc1
git push origin v1.0.1-rc1

# After workflow completes:
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ said-lam
```

---

## Testing Your Build

### Quick Smoke Test

```bash
python -c "
from lam_candle import LamEngine, TIER_FREE, TIER_BETA, TIER_LICENSED, TIER_INFINITE

# 1. Check tier constants
assert TIER_FREE == 1
assert TIER_BETA == 2
assert TIER_LICENSED == 3
assert TIER_INFINITE == 4
print('Tier constants: OK')

# 2. Create engine
engine = LamEngine()
print(f'Engine created: tier={engine.get_tier_name()}, backend={engine.get_backend()}')

# 3. Check methods exist
assert hasattr(engine, 'encode')
assert hasattr(engine, 'recall')
assert hasattr(engine, 'index')
assert hasattr(engine, 'activate')
assert hasattr(engine, 'search_kv')
print('Core methods: OK')

# 4. Test document indexing
engine.index('doc1', 'The quick brown fox jumps over the lazy dog')
engine.index('doc2', 'Python is a programming language')
assert engine.doc_count() == 2
print(f'Indexing: OK ({engine.doc_count()} docs)')

# 5. Test search
results = engine.search('brown fox')
print(f'Search: OK (top result: {results[0] if results else \"none\"})')

print('All tests passed')
"
```

### Full Integration Test

```bash
python -c "
from lam_candle import LamEngine
import numpy as np

engine = LamEngine()

# Index documents
docs = [
    'Machine learning is a subset of artificial intelligence',
    'The Eiffel Tower is located in Paris, France',
    'Python was created by Guido van Rossum',
    'The speed of light is approximately 299,792,458 meters per second',
]
for i, doc in enumerate(docs):
    engine.index(f'doc_{i}', doc)

# Test semantic search
results = engine.search('Who created Python?')
print(f'Semantic search: {results[:3]}')

# Test key-value search
results = engine.search_kv('speed of light')
print(f'KV search: {results}')

# Test encoding
if engine.has_model():
    emb = engine.encode(['test sentence'])
    print(f'Encoding shape: {np.array(emb).shape}')

print('Integration tests passed')
"
```

---

## Platform Matrix

| Platform | Target | GPU | Command |
|----------|--------|-----|---------|
| Linux x86_64 | `x86_64-unknown-linux-gnu` | CPU | `maturin build --release` |
| Linux x86_64 | `x86_64-unknown-linux-gnu` | CUDA | `maturin build --release --features cuda` |
| Linux ARM64 | `aarch64-unknown-linux-gnu` | CPU | `maturin build --release --target aarch64-unknown-linux-gnu` |
| macOS Intel | `x86_64-apple-darwin` | CPU | `maturin build --release --target x86_64-apple-darwin` |
| macOS Apple Silicon | `aarch64-apple-darwin` | CPU | `maturin build --release` |
| macOS Apple Silicon | `aarch64-apple-darwin` | Metal | `maturin build --release --features metal` |
| Windows x86_64 | `x86_64-pc-windows-msvc` | CPU | `maturin build --release` |

---

## What Ships in the Wheel

Each `.whl` file contains:

```
said_lam-1.0.0-cp311-cp311-manylinux_2_17_x86_64.whl
├── lam_candle.cpython-311-x86_64-linux-gnu.so   # Compiled Rust binary
├── said_lam/__init__.py                          # Python API
├── lam/__init__.py                               # Backwards-compat shim
└── said_lam-1.0.0.dist-info/                     # Package metadata
```

**NOT included**: Rust source code (`src/`), model weights (`weights/`), or build files.

Model weights are auto-downloaded from HuggingFace at first use.

---

## Build Optimization

The `Cargo.toml` release profile maximizes binary performance:

```toml
[profile.release]
opt-level = 3        # Maximum optimization
lto = true           # Link-Time Optimization (smaller, faster binary)
codegen-units = 1    # Single codegen unit (better optimization)
strip = true         # Strip debug symbols (smaller binary)
```

---

## Version Bumping Checklist

Before each release:

1. Update version in `Cargo.toml`:
   ```toml
   [package]
   version = "1.0.1"
   ```

2. Update version in `pyproject.toml`:
   ```toml
   [project]
   version = "1.0.1"
   ```

3. Both versions **must match** — maturin uses `Cargo.toml` for the wheel filename.

4. Commit, tag, push:
   ```bash
   git add Cargo.toml pyproject.toml
   git commit -m "Bump version to 1.0.1"
   git tag v1.0.1
   git push origin main --tags
   ```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `error: linker 'cc' not found` | Install build essentials: `apt install build-essential` (Linux) or Xcode CLT (macOS) |
| `nvcc not found` (CUDA build) | Install CUDA Toolkit: `apt install nvidia-cuda-toolkit` or download from NVIDIA |
| `aarch64-linux-gnu-gcc not found` | Install cross-compiler: `apt install gcc-aarch64-linux-gnu` |
| `maturin not found` | `pip install maturin` |
| `pyo3 version mismatch` | Ensure Python version matches target: `maturin build --interpreter python3.11` |
| Wheel installs but import fails | Check platform matches: `file lam_candle*.so` should match your arch |
| `Couldn't find a virtualenv` | Use `maturin build` + `pip install` instead of `maturin develop` |
| `maturin develop` fails | Create a venv first: `python -m venv .venv && source .venv/bin/activate` |
| TestPyPI install fails with deps | Use `--extra-index-url https://pypi.org/simple/` to resolve non-test deps |
| Version conflict on PyPI | You cannot re-upload the same version. Bump version and re-tag |

---

## Quick Reference

```bash
# Development (build + install)
maturin build --release --out dist/ && pip install dist/said_lam-*.whl --force-reinstall

# Release (tag triggers CI/CD → PyPI)
git tag v1.0.0 && git push origin v1.0.0

# Users install
pip install said-lam

# Users import
from lam_candle import LamEngine
engine = LamEngine()
```
