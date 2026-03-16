cd /workspace/LAM/LAM/said-lam && CARGO_BUILD_RUSTFLAGS="-C link-arg=-fuse-ld=bfd" maturin build --release --features cuda 2>&1
then run /workspace/LAM/LAM/said-lam/scripts/copy_so_after_build.sh
then test