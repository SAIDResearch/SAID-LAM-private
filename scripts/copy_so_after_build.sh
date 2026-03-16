#!/usr/bin/env bash
# Copy the compiled lam_candle .so into said_lam/ and tests/ after maturin/cargo build.
# Run after: maturin build --release --features cuda  (or cargo build --release --features cuda)
#
# Usage: bash scripts/copy_so_after_build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PKG_DIR="$PROJECT_DIR/said_lam"
TESTS_DIR="$PROJECT_DIR/tests"

# Prefer release build (maturin/cargo)
SO_SOURCE=""
for candidate in \
    "$PROJECT_DIR/target/release/liblam_candle.so" \
    "$PROJECT_DIR/target/maturin/liblam_candle.so" \
    "/workspace/LAM/LAM/said-lam/target/release/liblam_candle.so"; do
    if [ -f "$candidate" ]; then
        SO_SOURCE="$candidate"
        break
    fi
done

if [ -z "$SO_SOURCE" ]; then
    echo "ERROR: No liblam_candle.so found. Run from said-lam: maturin build --release --features cuda"
    exit 1
fi

EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SO_NAME="lam_candle${EXT_SUFFIX}"

echo "Copying $SO_SOURCE -> said_lam/ and tests/"
cp "$SO_SOURCE" "$PKG_DIR/$SO_NAME"
cp "$SO_SOURCE" "$PKG_DIR/lam_candle.so"
echo "  -> $PKG_DIR/$SO_NAME"
echo "  -> $PKG_DIR/lam_candle.so"

if [ -d "$TESTS_DIR" ]; then
    cp "$SO_SOURCE" "$TESTS_DIR/$SO_NAME"
    cp "$SO_SOURCE" "$TESTS_DIR/lam_candle.so"
    echo "  -> $TESTS_DIR/$SO_NAME"
    echo "  -> $TESTS_DIR/lam_candle.so"
fi

# Active venv: update lam_candle / said_lam in site-packages
if [ -n "${VIRTUAL_ENV:-}" ]; then
    SITE=$(find "$VIRTUAL_ENV" -type d -name "site-packages" 2>/dev/null | head -1)
    if [ -n "$SITE" ]; then
        for pkg in lam_candle said_lam; do
            if [ -d "$SITE/$pkg" ]; then
                cp "$SO_SOURCE" "$SITE/$pkg/$SO_NAME"
                echo "  -> $SITE/$pkg/$SO_NAME (venv)"
            fi
        done
    fi
fi

echo "Done."
