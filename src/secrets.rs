//! Embedding truncation (Matryoshka)
//!
//! Ported from _secrets.pyx / lam_package/rust_candle/src/secrets.rs
//! Used internally by LamEngine for truncate_embeddings().

/// Truncate embeddings to target dimension with L2 normalization
///
/// Valid target dimensions: 64, 128, 256.
/// If target_dim >= FULL_DIM (384), returns embeddings unchanged.
pub fn truncate_embeddings(
    embeddings: &[Vec<f32>],
    target_dim: usize,
) -> Result<Vec<Vec<f32>>, String> {
    const FULL_DIM: usize = 384;

    if target_dim == FULL_DIM {
        return Ok(embeddings.to_vec());
    }

    if !matches!(target_dim, 64 | 128 | 256) {
        return Err("INVALID_DIMENSION".to_string());
    }

    let result: Vec<Vec<f32>> = embeddings
        .iter()
        .map(|emb| {
            let truncated: Vec<f32> = emb[..target_dim].to_vec();
            let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                truncated.iter().map(|x| x / norm).collect()
            } else {
                truncated
            }
        })
        .collect();

    Ok(result)
}
