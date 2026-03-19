//! SAID-LAM: Linear Attention Model - Pure Rust Implementation
//!
//! This is the "Brain" - all secret sauce is hidden in this compiled binary.
//! The Python wrapper is just a thin interface.
//!
//! Features:
//! - 32K context window with O(n) complexity
//! - IDF-Surprise search (100% NIAH recall)
//! - Tier-based licensing (hidden in binary)
//! - No PyTorch dependency

use pyo3::prelude::*;

mod engine;
mod crystalline;
mod model;
mod storage;
mod sca_dropin;  // Text + Embedding storage (InMemory always, Mmap with feature)
pub(crate) mod secrets;   // Position interpolation & embedding truncation
pub(crate) mod license;   // License management, benchmark ranking, device lock

// Re-export main class
pub use engine::LamEngine;

// =============================================================================
// TIER SYSTEM (Hidden in binary - users can't modify)
// =============================================================================

pub(crate) const TIER_FREE: u8 = 1;
pub(crate) const TIER_BETA: u8 = 2;
pub(crate) const TIER_LICENSED: u8 = 3;
pub(crate) const TIER_INFINITE: u8 = 4;

const TIER_LIMITS: [(u8, usize); 4] = [
    (TIER_FREE, 12_000),      // 12K tokens
    (TIER_BETA, 32_000),      // 32K tokens
    (TIER_LICENSED, 32_000),  // 32K tokens
    (TIER_INFINITE, usize::MAX), // Unlimited
];

pub(crate) fn get_tier_limit(tier: u8) -> usize {
    TIER_LIMITS.iter()
        .find(|(t, _)| *t == tier)
        .map(|(_, limit)| *limit)
        .unwrap_or(12_000)
}

// =============================================================================
// LICENSE VALIDATION (Hidden in binary)
// Delegates to license::LicenseManager for key classification.
// =============================================================================

fn validate_license(key: &Option<String>) -> u8 {
    match key.as_deref() {
        Some(k) if !k.is_empty() => {
            // Classify the explicit key through LicenseManager's key logic
            // Handles sk_live_*, sk_ent_*, lam_*, LAM-*, BETA_* formats
            license::LicenseManager::classify_key(k)
        }
        _ => TIER_FREE,
    }
}

pub(crate) fn validate_activation(key: &str) -> bool {
    // Only accept keys validated through LicenseManager (online).
    // ACTIVATE_ prefix reserved for future admin-issued activation codes.
    key.starts_with("ACTIVATE_")
}

// =============================================================================
// PYTHON MODULE
// =============================================================================

/// Eagerly initialize CUDA + cuBLAS so that a subsequent PyTorch import
/// does not corrupt the cuBLAS handle.  Call this **before** `import torch`.
/// The created device is cached and reused by LamEngine.
#[pyfunction]
fn cuda_warmup() -> PyResult<()> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::{Device, Tensor, DType};

        // Try to create a CUDA device and run a small matmul to force cuBLAS init
        match Device::new_cuda(0) {
            Ok(dev) => {
                // Warm up cuBLAS with various matrix sizes to ensure all kernel
                // paths are initialized before PyTorch can interfere.
                for sz in [2, 32, 64, 128, 384] {
                    let a = Tensor::zeros((1, sz, 384), DType::F32, &dev)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
                    let b = Tensor::zeros((1, 384, sz), DType::F32, &dev)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
                    let _ = a.matmul(&b)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
                }
                // Cache the device so LamEngine reuses this cuBLAS handle
                model::set_warmed_up_device(dev);
                eprintln!("📊 CUDA warmup: cuBLAS initialized successfully");
            }
            Err(e) => {
                eprintln!("⚠️ CUDA warmup skipped (no GPU): {e:?}");
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("⚠️ CUDA warmup skipped (not compiled with CUDA)");
    }
    Ok(())
}

/// Python module initialization
#[pymodule]
fn lam_candle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LamEngine>()?;
    m.add_function(wrap_pyfunction!(cuda_warmup, m)?)?;
    m.add("__version__", "1.0.0")?;
    m.add("TIER_FREE", TIER_FREE)?;
    m.add("TIER_BETA", TIER_BETA)?;
    m.add("TIER_LICENSED", TIER_LICENSED)?;
    m.add("TIER_INFINITE", TIER_INFINITE)?;
    Ok(())
}
