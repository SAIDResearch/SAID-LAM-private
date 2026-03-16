#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# SAID-LAM: Development Setup
# ═══════════════════════════════════════════════════════════════════
#
# Builds lam_candle.so (CPU or CUDA) and installs it everywhere
# Python might import it from — project dir, tests dir, venv, AND
# /tmp test environments.
#
# Usage:
#   bash scripts/dev_setup.sh               # CPU build (default)
#   bash scripts/dev_setup.sh --gpu         # GPU/CUDA build
#   bash scripts/dev_setup.sh --so-only     # Skip build, just copy existing .so
#   bash scripts/dev_setup.sh --skip-deps  # Skip pip install of dev deps
#   bash scripts/dev_setup.sh --gpu --run-dir /tmp/said-lam-run
#   bash scripts/dev_setup.sh --gpu --venv /tmp/said-lam-venv
#
# /tmp test environment (common pattern):
#   bash scripts/dev_setup.sh --gpu \
#       --run-dir /tmp/said-lam-run \
#       --venv /tmp/said-lam-venv
#
# After setup, run tests:
#   cd tests && python test_crystalline_mteb.py
#   cd tests && python test_matryoshka.py
#   cd tests && python run_all_tests.py --skip-sts
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PKG_DIR="$PROJECT_DIR/said_lam"
TESTS_DIR="$PROJECT_DIR/tests"

# Find the real source repo (has target/release with built .so, or has cargo available)
# When running from a /tmp copy, PROJECT_DIR has Cargo.toml but no cargo toolchain
# and no target/ dir. We need to find the original source with the built artifacts.
SOURCE_DIR="$PROJECT_DIR"
if [ ! -d "$PROJECT_DIR/target/release" ]; then
    for candidate in \
        "/workspace/LAM/LAM/said-lam" \
        "$HOME/LAM/LAM/said-lam" \
        "/home/user/LAM/LAM/said-lam"; do
        if [ -d "$candidate/target/release" ] || [ -f "$candidate/Cargo.toml" ]; then
            SOURCE_DIR="$candidate"
            break
        fi
    done
fi

# Python extension suffix (e.g. .cpython-311-x86_64-linux-gnu.so)
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SO_NAME="lam_candle${EXT_SUFFIX}"

# Parse flags
BUILD_MODE="cpu"
SO_ONLY=false
SKIP_DEPS=false
EXTRA_RUN_DIR=""
EXTRA_VENV=""
for arg in "$@"; do
    case "$arg" in
        --gpu|--cuda) BUILD_MODE="gpu" ;;
        --so-only)    SO_ONLY=true ;;
        --skip-deps)  SKIP_DEPS=true ;;
        --run-dir)    _next="run-dir" ;;
        --venv)       _next="venv" ;;
        *)
            if [ "${_next:-}" = "run-dir" ]; then
                EXTRA_RUN_DIR="$arg"; _next=""
            elif [ "${_next:-}" = "venv" ]; then
                EXTRA_VENV="$arg"; _next=""
            fi
            ;;
    esac
done

# Auto-detect /tmp test environment if not specified
if [ -z "$EXTRA_RUN_DIR" ] && [ -d "/tmp/said-lam-run" ]; then
    EXTRA_RUN_DIR="/tmp/said-lam-run"
fi
if [ -z "$EXTRA_VENV" ] && [ -d "/tmp/said-lam-venv" ]; then
    EXTRA_VENV="/tmp/said-lam-venv"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  SAID-LAM Development Setup"
echo "═══════════════════════════════════════════════════════════════"
echo "  Project:    $PROJECT_DIR"
echo "  Package:    $PKG_DIR"
echo "  Extension:  $SO_NAME"
echo "  Source:     $SOURCE_DIR"
echo "  Build:      $BUILD_MODE"
if [ -n "$EXTRA_RUN_DIR" ]; then
echo "  Run dir:    $EXTRA_RUN_DIR"
fi
if [ -n "$EXTRA_VENV" ]; then
echo "  Venv:       $EXTRA_VENV"
fi
echo "═══════════════════════════════════════════════════════════════"

# ─── Step 1: Build ────────────────────────────────────────────────
# Ensure cargo is on PATH: source rustup env, then check common locations
if ! command -v cargo &>/dev/null; then
    # Rust installed via rustup puts cargo in ~/.cargo/bin; sourcing env adds it to PATH
    if [ -f "$HOME/.cargo/env" ]; then
        set +u
        source "$HOME/.cargo/env"
        set -u
        echo "  Sourced \$HOME/.cargo/env (Rust)"
    fi
fi

if ! command -v cargo &>/dev/null; then
    for cargo_candidate in \
        "$HOME/.cargo/bin/cargo" \
        "/root/.cargo/bin/cargo" \
        "${EXTRA_VENV:-}/bin/cargo" \
        "${VIRTUAL_ENV:-}/bin/cargo"; do
        if [ -n "$cargo_candidate" ] && [ -x "$cargo_candidate" ]; then
            export PATH="$(dirname "$cargo_candidate"):$PATH"
            echo "  Found cargo at: $cargo_candidate"
            break
        fi
    done
fi

# Still no cargo? Install Rust via rustup, then source env and continue
if ! command -v cargo &>/dev/null && [ "$SO_ONLY" = false ]; then
    echo ""
    echo "--- cargo not found — installing Rust (rustup) ---"
    if [ -f "$HOME/.cargo/env" ]; then
        set +u
        source "$HOME/.cargo/env"
        set -u
        echo "  Sourced \$HOME/.cargo/env (cargo should now be on PATH)"
    else
        echo "  Running: curl ... | sh -s -- -y"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        if [ -f "$HOME/.cargo/env" ]; then
            set +u
            source "$HOME/.cargo/env"
            set -u
            echo "  Sourced \$HOME/.cargo/env"
        fi
    fi
    if ! command -v cargo &>/dev/null; then
        echo ""
        echo "  ERROR: Rust install may have failed. Try manually:"
        echo "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo "    source \"\$HOME/.cargo/env\""
        echo "    then re-run this script."
        exit 1
    fi
    echo "  cargo: $(command -v cargo)"
fi

if [ "$SO_ONLY" = false ]; then
    if ! command -v cargo &>/dev/null; then
        echo ""
        echo "  ERROR: cargo not found. Install Rust: https://rustup.rs"
        exit 1
    elif [ ! -f "$SOURCE_DIR/Cargo.toml" ]; then
        echo ""
        echo "  No Cargo.toml in $SOURCE_DIR — skipping build, will use existing .so"
        SO_ONLY=true
    fi
fi

if [ "$SO_ONLY" = false ]; then
    echo ""
    echo "--- Building lam_candle ($BUILD_MODE) in $SOURCE_DIR ---"
    cd "$SOURCE_DIR"

    if [ "$BUILD_MODE" = "gpu" ]; then
        # Clean previous build to prevent stale CPU .so
        echo "  Cleaning previous build artifacts..."
        cargo clean --release 2>/dev/null || true

        echo "  cargo build --release --features cuda"
        cargo build --release --features cuda 2>&1 | tail -10
    else
        echo "  cargo build --release"
        cargo build --release 2>&1 | tail -10
    fi
    echo "  Build complete."
fi

# ─── Step 2: Find the .so ────────────────────────────────────────
SO_SOURCE=""
for candidate in \
    "$SOURCE_DIR/target/release/liblam_candle.so" \
    "$SOURCE_DIR/target/maturin/liblam_candle.so" \
    "$PROJECT_DIR/target/release/liblam_candle.so" \
    "$PROJECT_DIR/target/maturin/liblam_candle.so" \
    "$SOURCE_DIR/../said-lam-rust/python/lam_candle.so" \
    "$SOURCE_DIR/../lam_package/rust_candle/lam_candle.so"; do
    if [ -f "$candidate" ]; then
        SO_SOURCE="$candidate"
        break
    fi
done

if [ -z "$SO_SOURCE" ]; then
    echo ""
    echo "ERROR: No liblam_candle.so found in target/release/."
    echo "  Run 'cargo build --release' first."
    exit 1
fi

echo ""
echo "--- Installing .so ($BUILD_MODE build) ---"
echo "  Source: $SO_SOURCE"

# ─── Helper: install .so into a directory ─────────────────────────
install_so() {
    local dir="$1"
    local label="${2:-}"
    if [ -d "$dir" ]; then
        cp "$SO_SOURCE" "$dir/$SO_NAME"
        cp "$SO_SOURCE" "$dir/lam_candle.so" 2>/dev/null || true
        echo "  -> $dir/$SO_NAME${label:+ ($label)}"
    fi
}

# ─── Helper: install .so into a venv's site-packages ──────────────
install_so_venv() {
    local venv_path="$1"
    local label="${2:-venv}"
    if [ ! -d "$venv_path" ]; then
        return
    fi
    # Find site-packages inside the venv
    local site_dir
    site_dir=$(find "$venv_path" -type d -name "site-packages" 2>/dev/null | head -1)
    if [ -z "$site_dir" ]; then
        echo "  WARNING: No site-packages found in $venv_path"
        return
    fi
    # Replace .so in any package dirs (lam_candle/, said_lam/)
    for pkg_name in lam_candle said_lam; do
        if [ -d "$site_dir/$pkg_name" ]; then
            cp "$SO_SOURCE" "$site_dir/$pkg_name/$SO_NAME"
            echo "  -> $site_dir/$pkg_name/$SO_NAME ($label)"
        fi
    done
    # Replace bare .so in site-packages (pyo3 flat layout)
    if [ -f "$site_dir/$SO_NAME" ]; then
        cp "$SO_SOURCE" "$site_dir/$SO_NAME"
        echo "  -> $site_dir/$SO_NAME ($label flat)"
    fi
}

# ─── Step 3: Install everywhere ──────────────────────────────────

# Project package dir
install_so "$PKG_DIR" "package"

# Tests dir
install_so "$TESTS_DIR" "tests"

# /tmp run dir (if it exists)
if [ -n "$EXTRA_RUN_DIR" ]; then
    install_so "$EXTRA_RUN_DIR/said_lam" "run-dir package"
    install_so "$EXTRA_RUN_DIR/tests" "run-dir tests"
    install_so "$EXTRA_RUN_DIR" "run-dir root"
fi

# Active venv ($VIRTUAL_ENV)
if [ -n "${VIRTUAL_ENV:-}" ]; then
    install_so_venv "$VIRTUAL_ENV" "active venv"
fi

# Conda
if [ -n "${CONDA_PREFIX:-}" ]; then
    install_so_venv "$CONDA_PREFIX" "conda"
fi

# Explicit --venv or auto-detected /tmp/said-lam-venv
if [ -n "$EXTRA_VENV" ]; then
    install_so_venv "$EXTRA_VENV" "extra venv"
fi

# ─── Step 3b: pip install said-lam into the target venv ──────────
# This is the KEY step — without this, `from said_lam import LAM`
# fails with ModuleNotFoundError in /tmp test environments.
echo ""
echo "--- Installing said-lam package (pip install -e) ---"

# Determine which pip to use
if [ -n "$EXTRA_VENV" ] && [ -x "$EXTRA_VENV/bin/pip" ]; then
    TARGET_PIP="$EXTRA_VENV/bin/pip"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/pip" ]; then
    TARGET_PIP="$VIRTUAL_ENV/bin/pip"
else
    TARGET_PIP="pip3"
fi

# Install the said_lam package from SOURCE_DIR (editable so changes are live)
# Use --no-build-isolation because the .so is already built
echo "  Using: $TARGET_PIP"
echo "  Installing from: $SOURCE_DIR"
$TARGET_PIP install --no-build-isolation -e "$SOURCE_DIR" 2>&1 | tail -5 || {
    # Fallback: if editable install fails (e.g. maturin not available),
    # create a minimal .pth file so Python can find said_lam
    echo "  Editable install failed — falling back to .pth file method"
    SITE_DIR=$($TARGET_PIP show pip 2>/dev/null | grep -i "^Location:" | awk '{print $2}')
    if [ -z "$SITE_DIR" ]; then
        SITE_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
    fi
    if [ -n "$SITE_DIR" ] && [ -d "$SITE_DIR" ]; then
        echo "$SOURCE_DIR" > "$SITE_DIR/said-lam.pth"
        echo "  -> $SITE_DIR/said-lam.pth (pointing to $SOURCE_DIR)"
    else
        echo "  WARNING: Could not determine site-packages dir for .pth fallback"
    fi
}

# Also copy the .so into the venv site-packages after install
if [ -n "$EXTRA_VENV" ]; then
    install_so_venv "$EXTRA_VENV" "extra venv (post-install)"
fi

# ─── Step 3c: Install dev Python dependencies ────────────────────
if [ "$SKIP_DEPS" = false ]; then
    echo ""
    echo "--- Installing dev dependencies (mteb, pytest, psutil) ---"
    echo "  Using: $TARGET_PIP"
    $TARGET_PIP install "mteb>=1.0" "pytest>=7.0" "psutil" 2>&1 | tail -5
    echo "  Dev dependencies installed."
else
    echo "  (--skip-deps: skipping Python dependency install)"
fi

# ─── Step 4: Verify weights ──────────────────────────────────────
# Check project dir first, then source dir for weights
WEIGHTS_DIR="$PROJECT_DIR/weights"
if [ ! -f "$WEIGHTS_DIR/model.safetensors" ] && [ -f "$SOURCE_DIR/weights/model.safetensors" ]; then
    WEIGHTS_DIR="$SOURCE_DIR/weights"
fi
if [ ! -f "$WEIGHTS_DIR/model.safetensors" ]; then
    echo ""
    echo "ERROR: weights/model.safetensors not found."
    echo "  Download from HuggingFace: hf download SAIDResearch/SAID-LAM-v1"
    exit 1
fi
echo "  Weights: $WEIGHTS_DIR/model.safetensors ($(du -h "$WEIGHTS_DIR/model.safetensors" | cut -f1))"

# Also ensure /tmp run dir has weights symlinked
if [ -n "$EXTRA_RUN_DIR" ] && [ ! -e "$EXTRA_RUN_DIR/weights" ]; then
    ln -sf "$WEIGHTS_DIR" "$EXTRA_RUN_DIR/weights"
    echo "  Symlinked weights -> $EXTRA_RUN_DIR/weights"
fi

# ─── Step 5: Verify import and device ────────────────────────────
echo ""
echo "--- Verifying import (build=$BUILD_MODE) ---"
cd "$PROJECT_DIR"
python3 -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'said_lam')
from said_lam import LAM
model = LAM('weights')
tier = getattr(model, 'tier', None) or (getattr(model._engine, 'get_tier_name', lambda: 'unknown')())
max_tok = getattr(model, 'max_tokens', None) or getattr(model._engine, 'get_max_tokens', lambda: 12000)()
print(f'  Tier: {tier}')
print(f'  Max tokens: {max_tok}')
backend = getattr(model._engine, 'get_backend', lambda: 'unknown')()
print(f'  Backend: {backend}')
emb = model.encode(['test'])
print(f'  Encode OK: shape={emb.shape}')
print(f'  index_mteb: {hasattr(model._engine, \"index_mteb\")}')
print(f'  search_mteb: {hasattr(model._engine, \"search_mteb\")}')
print(f'  truncate_embeddings: {hasattr(model._engine, \"truncate_embeddings\")}')
print('  All checks passed')
" 2>&1 | grep -v "^$"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete ($BUILD_MODE build). Ready for testing."
echo ""
echo "  Run tests:"
echo "    cd $TESTS_DIR"
echo "    python test_crystalline_mteb.py        # Crystalline MTEB e2e"
echo "    python test_matryoshka.py              # Matryoshka truncation"
echo "    python run_all_tests.py --skip-sts     # Full suite (no STS)"
echo ""
if [ "$BUILD_MODE" = "gpu" ]; then
echo "  GPU NOTE: The device line above should show 'Cuda'."
echo "  If it still shows 'Cpu', check: nvidia-smi, CUDA_VISIBLE_DEVICES"
fi
echo "  NOTE: This .so is for DEVELOPMENT TESTING ONLY."
echo "  Client distribution uses: pip install said-lam"
echo "═══════════════════════════════════════════════════════════════"
