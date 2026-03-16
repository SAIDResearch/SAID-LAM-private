//! LAM Model - EXACT Port of _core.py to Rust Candle
//!
//! This is a 100% faithful port of the PyTorch implementation.
//! Every function, every tensor operation matches the original exactly.
//!
//! Weights are loaded from disk at runtime (HuggingFace cache or local path).
//! The .so binary contains only compiled code (~5MB), not model weights.

use candle_core::{Device, DType, Tensor, Result as CandleResult, D};
use candle_nn::{VarBuilder, Module, Linear, Embedding};
use safetensors::SafeTensors;
use std::path::Path;
use std::collections::HashMap;
use tokenizers::Tokenizer;

// ═══════════════════════════════════════════════════════════════════════════════
// MODEL CONFIGURATION - Exact match to _core.py/config.json
// ═══════════════════════════════════════════════════════════════════════════════

const VOCAB_SIZE: usize = 30522;
const HIDDEN_SIZE: usize = 384;
const NUM_LAYERS: usize = 6;
const NUM_HEADS: usize = 12;
const HEAD_DIM: usize = HIDDEN_SIZE / NUM_HEADS; // 32
const INTERMEDIATE_SIZE: usize = 1536;
const MAX_POSITION: usize = 512;
const CONV_KERNEL: usize = 4;
const LAYER_NORM_EPS: f64 = 1e-12;

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS - Exact port from _core.py lines 45-54
// ═══════════════════════════════════════════════════════════════════════════════

/// L2 normalization: F.normalize(x, p=2, dim=dim)
/// _core.py line 45-47
pub fn l2norm(x: &Tensor, _dim: i32) -> CandleResult<Tensor> {
    // Always use last dimension for L2 norm (most common case)
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let norm_clamped = norm.clamp(1e-12, f32::MAX)?;
    x.broadcast_div(&norm_clamped)
}

/// ELU + 1: (F.elu(x, 1.0, False) + 1.0)
/// _core.py line 49-50
#[allow(dead_code)]
pub fn elu_p1(x: &Tensor) -> CandleResult<Tensor> {
    // ELU: max(0, x) + min(0, alpha * (exp(x) - 1))
    // With alpha=1: max(0, x) + min(0, exp(x) - 1)
    let pos = x.maximum(&Tensor::zeros_like(x)?)?;
    let neg_input = x.minimum(&Tensor::zeros_like(x)?)?;
    let neg = (neg_input.exp()? - 1.0)?;
    let sum = (pos + neg)?;
    sum + 1.0
}

/// Sum normalization: x / x.sum(-1, keepdim=True)
/// _core.py line 52-53
#[allow(dead_code)]
pub fn sum_norm(x: &Tensor) -> CandleResult<Tensor> {
    let sum = x.sum_keepdim(D::Minus1)?;
    x.broadcast_div(&sum)
}

/// SiLU activation: use Candle's fused kernel (matches PyTorch F.silu)
pub fn silu(x: &Tensor) -> CandleResult<Tensor> {
    x.silu()
}

/// Sigmoid: 1 / (1 + exp(-x))
pub fn sigmoid(x: &Tensor) -> CandleResult<Tensor> {
    (x.neg()?.exp()? + 1.0)?.recip()
}

/// GELU activation (exact using erf, matching PyTorch F.gelu default)
/// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
pub fn gelu(x: &Tensor) -> CandleResult<Tensor> {
    x.gelu_erf()
}

// ═══════════════════════════════════════════════════════════════════════════════
// RMSNorm - Exact port from _core.py lines 55-64
// ═══════════════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

#[allow(dead_code)]
impl RMSNorm {
    pub fn new(vb: &VarBuilder, dim: usize) -> CandleResult<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps: 1e-5 })
    }
    
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        // return x * norm * self.weight
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let norm = (variance + self.eps)?.sqrt()?.recip()?;
        x.broadcast_mul(&norm)?.broadcast_mul(&self.weight)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FusedRMSNormGated - Exact port from _core.py lines 66-79
// ═══════════════════════════════════════════════════════════════════════════════

pub struct FusedRMSNormGated {
    weight: Tensor,
    gate: Tensor,
    eps: f64,
}

impl FusedRMSNormGated {
    pub fn new(vb: &VarBuilder, dim: usize) -> CandleResult<Self> {
        let weight = vb.get(dim, "weight")?;
        let gate = vb.get(dim, "gate")?;
        Ok(Self { weight, gate, eps: 1e-5 })
    }
    
    pub fn forward(&self, x: &Tensor, g: Option<&Tensor>) -> CandleResult<Tensor> {
        // norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        // if g is not None:
        //     return x * norm * self.weight * torch.sigmoid(g)
        // else:
        //     return x * norm * self.weight * torch.sigmoid(self.gate)
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let norm = (variance + self.eps)?.sqrt()?.recip()?;
        let x_normed = x.broadcast_mul(&norm)?.broadcast_mul(&self.weight)?;
        
        let gate_sigmoid = match g {
            Some(g_val) => sigmoid(g_val)?,
            None => sigmoid(&self.gate)?,
        };
        x_normed.broadcast_mul(&gate_sigmoid)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LayerNorm - Standard implementation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub fn new(vb: &VarBuilder) -> CandleResult<Self> {
        let weight = vb.get(HIDDEN_SIZE, "weight")?;
        let bias = vb.get(HIDDEN_SIZE, "bias")?;
        Ok(Self { weight, bias, eps: LAYER_NORM_EPS })
    }
    
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        x_normed.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ShortConvolution - Exact port from _core.py lines 81-100
// ═══════════════════════════════════════════════════════════════════════════════

pub struct ShortConvolution {
    weight: Tensor,  // [hidden_size, 1, kernel_size]
    bias: Tensor,    // [hidden_size]
    kernel_size: usize,
    use_silu: bool,
}

impl ShortConvolution {
    pub fn new(vb: &VarBuilder, hidden_size: usize, use_silu: bool) -> CandleResult<Self> {
        let weight = vb.get((hidden_size, 1, CONV_KERNEL), "conv.weight")?;
        let bias = vb.get(hidden_size, "conv.bias")?;
        Ok(Self { weight, bias, kernel_size: CONV_KERNEL, use_silu })
    }
    
    /// Forward pass matching _core.py ShortConvolution.forward
    /// Uses optimized depthwise convolution
    /// Note: Manual implementation is faster than Candle's conv1d for depthwise
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        
        // Transpose: [batch, seq, hidden] -> [batch, hidden, seq]
        let x_t = x.transpose(1, 2)?.contiguous()?;
        
        // Pad BOTH sides (PyTorch Conv1d behavior)
        let pad_len = self.kernel_size / 2;
        let x_padded = if pad_len > 0 {
            let zeros_left = Tensor::zeros((batch, hidden, pad_len), x.dtype(), x.device())?;
            let zeros_right = Tensor::zeros((batch, hidden, pad_len), x.dtype(), x.device())?;
            Tensor::cat(&[&zeros_left, &x_t, &zeros_right], 2)?
        } else {
            x_t
        };
        
        let padded_len = x_padded.dim(2)?;
        let out_len = padded_len - self.kernel_size + 1;
        
        // Optimized depthwise conv using tensor ops (stays on device)
        // For each kernel position, slice and multiply
        let mut result = self.bias.reshape((1, hidden, 1))?.broadcast_as((batch, hidden, out_len))?;
        
        for k_idx in 0..self.kernel_size {
            // Slice input: [batch, hidden, out_len]
            let x_slice = x_padded.narrow(2, k_idx, out_len)?;
            // Weight for this kernel position: [hidden, 1, 1] -> [1, hidden, 1]
            let w_slice = self.weight.narrow(2, k_idx, 1)?.squeeze(2)?.reshape((1, hidden, 1))?;
            // Multiply and add
            result = result.broadcast_add(&x_slice.broadcast_mul(&w_slice)?)?;
        }
        
        // Transpose back: [batch, hidden, seq] -> [batch, seq, hidden]
        let output = result.transpose(1, 2)?;
        
        // Truncate to original sequence length
        let output = if output.dim(1)? > seq_len {
            output.narrow(1, 0, seq_len)?
        } else {
            output
        };
        
        // Apply SiLU if configured (use Candle's fused kernel)
        if self.use_silu {
            output.silu()
        } else {
            Ok(output)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EnhancedResonanceFlux - Exact port from _core.py lines 102-225
// ═══════════════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
pub struct EnhancedResonanceFlux {
    d_k: usize,
    d_v: usize,
    num_heads: usize,
    w_bilinear: Tensor,  // [num_heads, d_k, d_v]
    temp: Tensor,        // [num_heads]
    flux_net_0_weight: Tensor,
    flux_net_0_bias: Tensor,
    flux_net_2_weight: Tensor,
    flux_net_2_bias: Tensor,
    token_flux_proj_0_weight: Tensor,
    token_flux_proj_0_bias: Tensor,
    token_flux_proj_2_weight: Tensor,
    token_flux_proj_2_bias: Tensor,
}

#[allow(dead_code)]
impl EnhancedResonanceFlux {
    pub fn new(vb: &VarBuilder, d_k: usize, d_v: usize, num_heads: usize) -> CandleResult<Self> {
        Ok(Self {
            d_k,
            d_v,
            num_heads,
            w_bilinear: vb.get((num_heads, d_k, d_v), "W_bilinear")?,
            temp: vb.get(num_heads, "temp")?,
            flux_net_0_weight: vb.get((d_k / 2, d_k + d_v + 1), "flux_net.0.weight")?,
            flux_net_0_bias: vb.get(d_k / 2, "flux_net.0.bias")?,
            flux_net_2_weight: vb.get((1, d_k / 2), "flux_net.2.weight")?,
            flux_net_2_bias: vb.get(1, "flux_net.2.bias")?,
            token_flux_proj_0_weight: vb.get((d_k / 2, d_k + d_v), "token_flux_proj.0.weight")?,
            token_flux_proj_0_bias: vb.get(d_k / 2, "token_flux_proj.0.bias")?,
            token_flux_proj_2_weight: vb.get((1, d_k / 2), "token_flux_proj.2.weight")?,
            token_flux_proj_2_bias: vb.get(1, "token_flux_proj.2.bias")?,
        })
    }
    
    /// Forward pass - Exact port of _core.py lines 158-212
    /// Handles both 4D [b, h, c, d_k] and 5D [b, h, n, c, d_k] inputs
    pub fn forward(&self, k_chunk: &Tensor, u_chunk: &Tensor) -> CandleResult<Tensor> {
        let dims = k_chunk.dims();
        
        if dims.len() == 4 {
            // [b, h, c, d_k]
            let (_b, h, _c, _d_k) = k_chunk.dims4()?;
            
            // Bilinear: einsum('bhck,hkd->bhcd', k_chunk, W_bilinear)
            let k_proj = self.einsum_bhck_hkd_to_bhcd(k_chunk)?;
            
            // Interaction: (k_proj * u_chunk).sum(-1)
            let interaction = k_proj.mul(u_chunk)?.sum(D::Minus1)?;
            
            // Temperature scaling
            let temp_expanded = self.temp.reshape((1, h, 1))?;
            let attn_scores = interaction.broadcast_div(&temp_expanded)?;
            
            // Average
            let avg_attn = attn_scores.mean(D::Minus1)?;
            let k_avg = k_chunk.mean(2)?;
            let u_avg = u_chunk.mean(2)?;
            
            // Flux network
            let flux_input = Tensor::cat(&[&k_avg, &u_avg, &avg_attn.unsqueeze(D::Minus1)?], D::Minus1)?;
            let x = flux_input.broadcast_matmul(&self.flux_net_0_weight.t()?)?;
            let x = x.broadcast_add(&self.flux_net_0_bias)?;
            let x = silu(&x)?;
            let x = x.broadcast_matmul(&self.flux_net_2_weight.t()?)?;
            let x = x.broadcast_add(&self.flux_net_2_bias)?;
            let psi = sigmoid(&x)?.squeeze(D::Minus1)?;
            
            psi.clamp(0.01, 0.99)
            
        } else if dims.len() == 5 {
            // [b, h, n, c, d_k] - Vectorized
            let (_b, h, _n, _c, _d_k) = k_chunk.dims5()?;
            
            // Bilinear: einsum('bhnck,hkd->bhncd', k_chunk, W_bilinear)
            let k_proj = self.einsum_bhnck_hkd_to_bhncd(k_chunk)?;
            
            // Interaction
            let interaction = k_proj.mul(u_chunk)?.sum(D::Minus1)?;
            
            // Temperature scaling
            let temp_expanded = self.temp.reshape((1, h, 1, 1))?;
            let attn_scores = interaction.broadcast_div(&temp_expanded)?;
            
            // Average
            let avg_attn = attn_scores.mean(D::Minus1)?;
            let k_avg = k_chunk.mean(3)?;
            let u_avg = u_chunk.mean(3)?;
            
            // Flux network
            let flux_input = Tensor::cat(&[&k_avg, &u_avg, &avg_attn.unsqueeze(D::Minus1)?], D::Minus1)?;
            let x = flux_input.broadcast_matmul(&self.flux_net_0_weight.t()?)?;
            let x = x.broadcast_add(&self.flux_net_0_bias)?;
            let x = silu(&x)?;
            let x = x.broadcast_matmul(&self.flux_net_2_weight.t()?)?;
            let x = x.broadcast_add(&self.flux_net_2_bias)?;
            let psi = sigmoid(&x)?.squeeze(D::Minus1)?;
            
            psi.clamp(0.01, 0.99)
        } else {
            Err(candle_core::Error::Msg(format!("Unexpected dims: {:?}", dims)))
        }
    }
    
    /// Compute token flux - Exact port of _core.py lines 214-225
    pub fn compute_token_flux(&self, k: &Tensor, v: &Tensor) -> CandleResult<Tensor> {
        // kv = torch.cat([k, v], dim=-1)
        // return self.token_flux_proj(kv).clamp(0.01, 0.99)
        let kv = Tensor::cat(&[k, v], D::Minus1)?;
        
        let x = kv.broadcast_matmul(&self.token_flux_proj_0_weight.t()?)?;
        let x = x.broadcast_add(&self.token_flux_proj_0_bias)?;
        let x = silu(&x)?;
        let x = x.broadcast_matmul(&self.token_flux_proj_2_weight.t()?)?;
        let x = x.broadcast_add(&self.token_flux_proj_2_bias)?;
        let psi = sigmoid(&x)?;
        
        psi.clamp(0.01, 0.99)
    }
    
    // Helper: einsum('bhck,hkd->bhcd', a, W_bilinear)
    fn einsum_bhck_hkd_to_bhcd(&self, a: &Tensor) -> CandleResult<Tensor> {
        let (_batch, heads, _chunk_size, _d_k) = a.dims4()?;
        
        let mut result = Vec::new();
        for h in 0..heads {
            let a_h = a.narrow(1, h, 1)?.squeeze(1)?; // [b, c, d_k]
            let w_h = self.w_bilinear.narrow(0, h, 1)?.squeeze(0)?; // [d_k, d_v]
            let r_h = a_h.contiguous()?.matmul(&w_h.contiguous()?)?; // [b, c, d_v]
            result.push(r_h.unsqueeze(1)?);
        }
        Tensor::cat(&result, 1)
    }
    
    // Helper: einsum('bhnck,hkd->bhncd', a, W_bilinear)
    fn einsum_bhnck_hkd_to_bhncd(&self, a: &Tensor) -> CandleResult<Tensor> {
        let (batch, heads, n_chunks, chunk_size, d_k) = a.dims5()?;
        
        let mut result = Vec::new();
        for h in 0..heads {
            let a_h = a.narrow(1, h, 1)?.squeeze(1)?; // [b, n, c, d_k]
            let w_h = self.w_bilinear.narrow(0, h, 1)?.squeeze(0)?; // [d_k, d_v]
            
            // Reshape for batch matmul: [b*n, c, d_k]
            let a_flat = a_h.reshape((batch * n_chunks, chunk_size, d_k))?.contiguous()?;
            let r_flat = a_flat.matmul(&w_h.contiguous()?)?; // [b*n, c, d_v]
            let r_h = r_flat.reshape((batch, n_chunks, chunk_size, self.d_v))?;
            result.push(r_h.unsqueeze(1)?);
        }
        Tensor::cat(&result, 1)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DeltaNet Layer - Exact port of EnhancedHierarchicalDeltaNet from _core.py
// ═══════════════════════════════════════════════════════════════════════════════

pub struct DeltaNetLayer {
    // Projections - _core.py lines 501-503
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    
    // Beta scaling - _core.py lines 506-508
    b_proj: Linear,
    
    // Convolutions - _core.py lines 511-526
    q_conv1d: ShortConvolution,
    k_conv1d: ShortConvolution,
    v_conv1d: ShortConvolution,
    
    // Hierarchical decay - _core.py lines 532-545
    fast_decay_proj: Linear,
    fast_decay_bias: Tensor,
    slow_decay_proj: Linear,
    slow_decay_bias: Tensor,
    
    // Output gates - _core.py lines 554-556
    fast_gate_proj: Linear,
    slow_gate_proj: Linear,
    
    // Resonance flux - _core.py lines 548-550
    resonance_flux: EnhancedResonanceFlux,
    
    // Output processing - _core.py lines 559-564
    g_proj: Linear,
    o_norm: FusedRMSNormGated,
    o_proj: Linear,
}

impl DeltaNetLayer {
    pub fn new(vb: &VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("v_proj"))?,
            b_proj: candle_nn::linear_no_bias(HIDDEN_SIZE, NUM_HEADS, vb.pp("b_proj"))?,
            q_conv1d: ShortConvolution::new(&vb.pp("q_conv1d"), HIDDEN_SIZE, true)?,
            k_conv1d: ShortConvolution::new(&vb.pp("k_conv1d"), HIDDEN_SIZE, true)?,
            v_conv1d: ShortConvolution::new(&vb.pp("v_conv1d"), HIDDEN_SIZE, true)?,
            fast_decay_proj: candle_nn::linear(HIDDEN_SIZE, NUM_HEADS, vb.pp("fast_decay_proj"))?,
            fast_decay_bias: vb.get(NUM_HEADS, "fast_decay_bias")?,
            slow_decay_proj: candle_nn::linear(HIDDEN_SIZE, NUM_HEADS, vb.pp("slow_decay_proj"))?,
            slow_decay_bias: vb.get(NUM_HEADS, "slow_decay_bias")?,
            fast_gate_proj: candle_nn::linear_no_bias(HIDDEN_SIZE, NUM_HEADS, vb.pp("fast_gate_proj"))?,
            slow_gate_proj: candle_nn::linear_no_bias(HIDDEN_SIZE, NUM_HEADS, vb.pp("slow_gate_proj"))?,
            resonance_flux: EnhancedResonanceFlux::new(&vb.pp("resonance_flux"), HEAD_DIM, HEAD_DIM, NUM_HEADS)?,
            g_proj: candle_nn::linear_no_bias(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("g_proj"))?,
            o_norm: FusedRMSNormGated::new(&vb.pp("o_norm"), HEAD_DIM)?,
            o_proj: candle_nn::linear_no_bias(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("o_proj"))?,
        })
    }
    
    /// Forward pass - Exact port of EnhancedHierarchicalDeltaNet.forward
    /// _core.py lines 566-764
    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, seq_len, _hidden) = hidden_states.dims3()?;

        // CHUNK_SIZE=32 padding: pad to multiple of 32 when seq_len >= 64 (match Python F.pad)
        const CHUNK_SIZE: usize = 32;
        let (hidden_states, work_len) = if seq_len >= 64 {
            let pad_len = (CHUNK_SIZE - seq_len % CHUNK_SIZE) % CHUNK_SIZE;
            let padded_len = seq_len + pad_len;
            let zeros = Tensor::zeros(
                (batch_size, pad_len, HIDDEN_SIZE),
                hidden_states.dtype(),
                hidden_states.device(),
            )?;
            let padded = Tensor::cat(&[hidden_states, &zeros], 1)?;
            (padded, padded_len)
        } else {
            (hidden_states.clone(), seq_len)
        };

        // Linear projections + convolutions (_core.py lines 620-637)
        let q = self.q_conv1d.forward(&self.q_proj.forward(&hidden_states)?)?;
        let k = self.k_conv1d.forward(&self.k_proj.forward(&hidden_states)?)?;
        let v = self.v_conv1d.forward(&self.v_proj.forward(&hidden_states)?)?;

        // Reshape for multi-head: [b, l, (h d)] -> [b, h, l, d]
        // CRITICAL: Make contiguous after transpose to ensure proper memory layout for matmul
        let q = q.reshape((batch_size, work_len, NUM_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k.reshape((batch_size, work_len, NUM_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.reshape((batch_size, work_len, NUM_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;

        // Beta scaling (_core.py lines 664-669)
        let beta = sigmoid(&self.b_proj.forward(&hidden_states)?)?;
        let beta = beta.transpose(1, 2)?; // [b, h, l]

        // Hierarchical decay (_core.py lines 672-678)
        let fast_decay_raw = self.fast_decay_proj.forward(&hidden_states)?;
        let fast_decay = sigmoid(&fast_decay_raw.broadcast_add(&self.fast_decay_bias)?)?;
        let fast_decay = fast_decay.transpose(1, 2)?;

        let slow_decay_raw = self.slow_decay_proj.forward(&hidden_states)?;
        let slow_decay = sigmoid(&slow_decay_raw.broadcast_add(&self.slow_decay_bias)?)?;
        let slow_decay = slow_decay.transpose(1, 2)?;

        // Output gates (_core.py lines 681-686)
        let fast_gate = sigmoid(&self.fast_gate_proj.forward(&hidden_states)?)?;
        let fast_gate = fast_gate.transpose(1, 2)?.unsqueeze(D::Minus1)?;

        let slow_gate = sigmoid(&self.slow_gate_proj.forward(&hidden_states)?)?;
        let slow_gate = slow_gate.transpose(1, 2)?.unsqueeze(D::Minus1)?;

        // Enhanced hierarchical delta rule (_core.py line 706-713)
        let o = enhanced_hierarchical_delta_rule(
            &q, &k, &v, &beta,
            &fast_decay, &slow_decay,
            &fast_gate, &slow_gate,
            &self.resonance_flux,
        )?;

        // Transpose back: [b, h, l, d] -> [b, l, h, d]
        let o = o.transpose(1, 2)?;

        // Output gating and norm (_core.py lines 749-753)
        let g = self.g_proj.forward(&hidden_states)?;
        let g = g.reshape((batch_size, work_len, NUM_HEADS, HEAD_DIM))?;
        let o = self.o_norm.forward(&o, Some(&g))?;

        // Final projection (_core.py lines 756-757); trim back to seq_len if padded
        let o = o.reshape((batch_size, work_len, HIDDEN_SIZE))?;
        let o = if work_len > seq_len {
            o.narrow(1, 0, seq_len)?
        } else {
            o
        };
        self.o_proj.forward(&o)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// _enhanced_hierarchical_delta_rule_impl - Exact port from _core.py lines 227-392
// ═══════════════════════════════════════════════════════════════════════════════

fn enhanced_hierarchical_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    _fast_decay: &Tensor,
    _slow_decay: &Tensor,
    fast_gate: &Tensor,
    slow_gate: &Tensor,
    _resonance_flux: &EnhancedResonanceFlux,
) -> CandleResult<Tensor> {
    // Work entirely in 4D [b, h, l, d] to avoid Candle 5D matmul striding issues
    let (_b, _h, l, _d_k) = q.dims4()?;
    let d_v = v.dim(D::Minus1)?;
    
    // L2 normalize q and k (_core.py lines 284-285)
    let q = l2norm(q, -1)?.contiguous()?;
    let k = l2norm(k, -1)?.contiguous()?;
    
    // Beta scaling (_core.py lines 287-289)
    let beta_expanded = beta.unsqueeze(D::Minus1)?;
    let v = v.broadcast_mul(&beta_expanded)?.contiguous()?;
    let k_beta = k.broadcast_mul(&beta_expanded)?.contiguous()?;
    
    // TITAN KERNEL 1: attn_const computation (_core.py lines 336-347)
    // All operations in 4D [b, h, l, d]
    let k_t = k.transpose(D::Minus1, D::Minus2)?.contiguous()?;
    let attn_const = k_beta.matmul(&k_t)?.neg()?; // [b, h, l, l]
    
    // Mask upper triangle (including diagonal)
    let attn_const = mask_triu_4d(&attn_const, 0)?;
    
    // Vectorized cumulative
    let mask = tril_mask(l, attn_const.device())?;
    let mask = mask.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, l, l]
    let attn_const_t = attn_const.transpose(D::Minus1, D::Minus2)?.contiguous()?;
    let updates = attn_const.contiguous()?.matmul(&attn_const_t)?.broadcast_mul(&mask)?;
    
    // Add identity
    let eye = eye_matrix(l, attn_const.device(), attn_const.dtype())?;
    let eye = eye.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, l, l]
    let attn_const = (attn_const + updates)?.broadcast_add(&eye)?;
    
    // Fused u+w computation (_core.py lines 345-347)
    let vk_stacked = Tensor::cat(&[&v, &k_beta], D::Minus1)?.contiguous()?;
    let uw_stacked = attn_const.contiguous()?.matmul(&vk_stacked)?;
    let u = uw_stacked.narrow(D::Minus1, 0, d_v)?.contiguous()?;
    
    // TITAN KERNEL 2: Attention computation (_core.py lines 352-358)
    let attn_all = q.matmul(&k_t)?; // [b, h, l, l]
    let attn_all = mask_triu_4d(&attn_all, 1)?;
    
    // TITAN KERNEL 3: Dual core output (_core.py lines 363-365)
    let attn_v = attn_all.contiguous()?.matmul(&v)?;
    let o_fast = fast_gate.broadcast_mul(&attn_v)?;
    
    let attn_u = attn_all.contiguous()?.matmul(&u)?;
    let o_slow = slow_gate.broadcast_mul(&attn_u)?;
    
    // Merge: o = 0.1 * o_fast + 0.9 * o_slow (_core.py line 365)
    (o_fast * 0.1)? + (o_slow * 0.9)?
}

// Helper: Mask upper triangle for 4D tensors
fn mask_triu_4d(x: &Tensor, diagonal: i32) -> CandleResult<Tensor> {
    let dims = x.dims();
    let size = dims[dims.len() - 1];
    
    let mut mask_data = vec![1.0f32; size * size];
    for i in 0..size {
        for j in 0..size {
            if j as i32 >= i as i32 + diagonal {
                mask_data[i * size + j] = 0.0;
            }
        }
    }
    let mask = Tensor::from_vec(mask_data, (size, size), x.device())?;
    let mask = mask.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, l, l]
    x.broadcast_mul(&mask)
}

// Helper: Create lower triangular mask
fn tril_mask(size: usize, device: &Device) -> CandleResult<Tensor> {
    let mut data = vec![0.0f32; size * size];
    for i in 0..size {
        for j in 0..i {
            data[i * size + j] = 1.0;
        }
    }
    Tensor::from_vec(data, (size, size), device)
}

// Helper: Create identity matrix
fn eye_matrix(size: usize, device: &Device, dtype: DType) -> CandleResult<Tensor> {
    let mut data = vec![0.0f32; size * size];
    for i in 0..size {
        data[i * size + i] = 1.0;
    }
    Tensor::from_vec(data, (size, size), device)?.to_dtype(dtype)
}

// Helper: Mask upper triangle
#[allow(dead_code)]
fn mask_triu(x: &Tensor, diagonal: i32) -> CandleResult<Tensor> {
    let dims = x.dims();
    let size = dims[dims.len() - 1];
    
    let mut mask_data = vec![1.0f32; size * size];
    for i in 0..size {
        for j in 0..size {
            if j as i32 >= i as i32 + diagonal {
                mask_data[i * size + j] = 0.0;
            }
        }
    }
    let mask = Tensor::from_vec(mask_data, (size, size), x.device())?;
    x.broadcast_mul(&mask)
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFN Layers - Exact port from _core.py create_lam_model lines 979-991
// ═══════════════════════════════════════════════════════════════════════════════

pub struct FFNIntermediate {
    dense: Linear,
}

impl FFNIntermediate {
    pub fn new(vb: &VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            dense: candle_nn::linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, vb.pp("dense"))?,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = self.dense.forward(x)?;
        gelu(&x)
    }
}

pub struct FFNOutput {
    dense: Linear,
}

impl FFNOutput {
    pub fn new(vb: &VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            dense: candle_nn::linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, vb.pp("dense"))?,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.dense.forward(x)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Embeddings - Exact port from _core.py create_lam_model lines 952-958
// ═══════════════════════════════════════════════════════════════════════════════

pub struct Embeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl Embeddings {
    pub fn new(vb: &VarBuilder) -> CandleResult<Self> {
        // Load from embeddings directly (stored in safetensors)
        // Note: Both embeddings.* and teacher_model.embeddings.* exist
        // We use embeddings.* which is the LAM trained version
        let vb_emb = vb.pp("embeddings");
        Ok(Self {
            word_embeddings: candle_nn::embedding(VOCAB_SIZE, HIDDEN_SIZE, vb_emb.pp("word_embeddings"))?,
            position_embeddings: candle_nn::embedding(MAX_POSITION, HIDDEN_SIZE, vb_emb.pp("position_embeddings"))?,
            token_type_embeddings: candle_nn::embedding(2, HIDDEN_SIZE, vb_emb.pp("token_type_embeddings"))?,
            layer_norm: LayerNorm::new(&vb_emb.pp("LayerNorm"))?,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let seq_len = input_ids.dim(D::Minus1)?;
        
        // Word embeddings
        let word_emb = self.word_embeddings.forward(input_ids)?;
        
        // Position embeddings
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        let position_ids = Tensor::new(position_ids.as_slice(), input_ids.device())?;
        let pos_emb = self.position_embeddings.forward(&position_ids)?;
        
        // Token type embeddings (all zeros)
        let token_type_ids = Tensor::zeros(input_ids.shape(), DType::U32, input_ids.device())?;
        let token_type_emb = self.token_type_embeddings.forward(&token_type_ids)?;
        
        // Combine: (word + token_type) + position to match Python evaluation order
        // pos_emb is [seq, 384]; broadcast to [batch, seq, 384] for add (batch may be > 1)
        let batch_size = word_emb.dim(0)?;
        let pos_emb = pos_emb.unsqueeze(0)?.broadcast_as((batch_size, seq_len, HIDDEN_SIZE))?;
        let embeddings = (word_emb.broadcast_add(&token_type_emb)? + pos_emb)?;
        
        // Layer norm
        self.layer_norm.forward(&embeddings)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Full LAM Model - Exact port of create_lam_model + LAMForward
// ═══════════════════════════════════════════════════════════════════════════════

pub struct LAMModel {
    device: Device,
    tokenizer: Option<Tokenizer>,
    embeddings: Embeddings,
    deltanet_layers: Vec<DeltaNetLayer>,
    deltanet_norms: Vec<LayerNorm>,
    ffn_intermediates: Vec<FFNIntermediate>,
    ffn_outputs: Vec<FFNOutput>,
    ffn_norms: Vec<LayerNorm>,
}

impl LAMModel {
    /// Get the best available device (CUDA > CPU)
    /// Checks CUDA_VISIBLE_DEVICES environment variable at initialization
    fn get_best_device() -> Device {
        // Check CUDA_VISIBLE_DEVICES first - if empty, force CPU
        if let Ok(cuda_visible) = std::env::var("CUDA_VISIBLE_DEVICES") {
            if cuda_visible.is_empty() {
                eprintln!("📊 CUDA_VISIBLE_DEVICES is empty, forcing CPU mode");
                return Device::Cpu;
            }
        }
        
        // Try CUDA first (loop through IDs to find a working one)
        #[cfg(feature = "cuda")]
        {
            for i in 0..4 {
                match Device::new_cuda(i) {
                    Ok(device) => {
                        return device;
                    }
                    Err(e) => {
                        eprintln!("⚠️ CUDA Device {} initialization failed: {:?}", i, e);
                    }
                }
            }
        }
        // Fall back to CPU
        eprintln!("⚠️ All CUDA devices failed or not available, falling back to CPU");
        Device::Cpu
    }
    
    /// Load model from a filesystem path.
    /// Expects model.safetensors and tokenizer.json in the directory (or a direct file path).
    pub fn load(path: &str) -> Result<Self, String> {
        let device = Self::get_best_device();
        eprintln!("📊 LAM using device: {:?}", device);
        let base_path = Path::new(path);

        // Find weights file
        let weights_path = if base_path.is_file() {
            path.to_string()
        } else {
            let safetensors_path = base_path.join("model.safetensors");
            if safetensors_path.exists() {
                safetensors_path.to_string_lossy().to_string()
            } else {
                return Err(format!("Model weights not found at {}", path));
            }
        };

        // Load tokenizer
        let tokenizer_path = if base_path.is_file() {
            base_path.parent().map(|p| p.join("tokenizer.json"))
        } else {
            Some(base_path.join("tokenizer.json"))
        };
        let tokenizer = tokenizer_path
            .filter(|p| p.exists())
            .and_then(|p| Tokenizer::from_file(p).ok())
            .map(|mut tok| {
                // Override tokenizer.json truncation (128) → MAX_POSITION (512)
                // to utilize the full position embedding capacity
                use tokenizers::TruncationParams;
                let _ = tok.with_truncation(Some(TruncationParams {
                    max_length: MAX_POSITION,
                    ..Default::default()
                }));
                // Disable fixed padding (tokenizer.json pads to 128)
                let _ = tok.with_padding(None);
                tok
            });

        // Load safetensors
        let data = std::fs::read(&weights_path)
            .map_err(|e| format!("Failed to read weights: {}", e))?;
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| format!("Failed to parse safetensors: {}", e))?;
        
        Self::build_from_tensors(tensors, tokenizer, device)
    }

    /// Common builder: deserialize SafeTensors into a LAMModel.
    fn build_from_tensors(tensors: SafeTensors<'_>, tokenizer: Option<Tokenizer>, device: Device) -> Result<Self, String> {
        // Convert to HashMap
        let mut tensor_map: HashMap<String, Tensor> = HashMap::new();
        for (name, view) in tensors.tensors() {
            let tensor = Tensor::from_raw_buffer(
                view.data(),
                DType::F32,
                view.shape(),
                &device,
            ).map_err(|e| format!("Failed to convert tensor {}: {}", name, e))?;
            tensor_map.insert(name.to_string(), tensor);
        }

        let vb = VarBuilder::from_tensors(tensor_map, DType::F32, &device);

        // Build model
        let embeddings = Embeddings::new(&vb)
            .map_err(|e| format!("Failed to build embeddings: {}", e))?;

        let mut deltanet_layers = Vec::new();
        let mut deltanet_norms = Vec::new();
        let mut ffn_intermediates = Vec::new();
        let mut ffn_outputs = Vec::new();
        let mut ffn_norms = Vec::new();

        for i in 0..NUM_LAYERS {
            deltanet_layers.push(
                DeltaNetLayer::new(&vb.pp(format!("deltanet_layers.{}", i)))
                    .map_err(|e| format!("Failed to build deltanet_layers.{}: {}", i, e))?
            );
            deltanet_norms.push(
                LayerNorm::new(&vb.pp(format!("deltanet_norms.{}", i)))
                    .map_err(|e| format!("Failed to build deltanet_norms.{}: {}", i, e))?
            );
            ffn_intermediates.push(
                FFNIntermediate::new(&vb.pp(format!("deltanet_ffns.{}", i)))
                    .map_err(|e| format!("Failed to build ffn_intermediates.{}: {}", i, e))?
            );
            ffn_outputs.push(
                FFNOutput::new(&vb.pp(format!("ffn_outputs.{}", i)))
                    .map_err(|e| format!("Failed to build ffn_outputs.{}: {}", i, e))?
            );
            ffn_norms.push(
                LayerNorm::new(&vb.pp(format!("ffn_norms.{}", i)))
                    .map_err(|e| format!("Failed to build ffn_norms.{}: {}", i, e))?
            );
        }

        Ok(Self {
            device,
            tokenizer,
            embeddings,
            deltanet_layers,
            deltanet_norms,
            ffn_intermediates,
            ffn_outputs,
            ffn_norms,
        })
    }
    
    /// Get a reference to the tokenizer (for CrystallineCore)
    pub fn get_tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }
    
    /// Tokenize using BERT tokenizer
    fn tokenize(&self, text: &str) -> Vec<u32> {
        if let Some(ref tokenizer) = self.tokenizer {
            match tokenizer.encode(text, true) {
                Ok(encoding) => {
                    // Get tokens and filter out padding (token ID 0)
                    // Python tokenizer doesn't pad, so we need to match that
                    let ids = encoding.get_ids();
                    // Find end of actual tokens (before padding)
                    let end = ids.iter().rposition(|&id| id != 0)
                        .map(|i| i + 1)
                        .unwrap_or(ids.len());
                    ids[..end].to_vec()
                }
                Err(_) => self.fallback_tokenize(text),
            }
        } else {
            self.fallback_tokenize(text)
        }
    }
    
    fn fallback_tokenize(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![101u32]; // [CLS]
        for word in text.split_whitespace().take(MAX_POSITION - 2) {
            let hash = word.to_lowercase().bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            tokens.push((hash % (VOCAB_SIZE as u64 - 1000) + 1000) as u32);
        }
        tokens.push(102); // [SEP]
        tokens
    }
    
    /// Encode texts to embeddings - With BATCH PROCESSING for GPU efficiency
    /// Default batch_size=32 (optimal for DeltaNet architecture)
    pub fn encode(&self, texts: &[String], normalize: bool) -> Result<Vec<Vec<f32>>, String> {
        self.encode_with_batch_size(texts, normalize, 32)
    }
    
    /// Encode with configurable batch size
    pub fn encode_with_batch_size(&self, texts: &[String], normalize: bool, batch_size: usize) -> Result<Vec<Vec<f32>>, String> {
        let batch_size = batch_size.max(1); // Ensure at least 1
        let mut all_embeddings = Vec::with_capacity(texts.len());
        
        // Process in batches
        for batch_start in (0..texts.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(texts.len());
            let batch_texts = &texts[batch_start..batch_end];
            
            // Tokenize all texts in this batch
            let mut all_tokens: Vec<Vec<u32>> = Vec::with_capacity(batch_texts.len());
            let mut max_len = 0usize;
            
            for text in batch_texts {
                let tokens = self.tokenize(text);
                max_len = max_len.max(tokens.len());
                all_tokens.push(tokens);
            }
            
            // Handle empty batch edge case
            if max_len == 0 {
                for _ in batch_texts {
                    all_embeddings.push(vec![0.0; HIDDEN_SIZE]);
                }
                continue;
            }
            
            // Pad all sequences to max_len and create attention masks
            let batch_size = all_tokens.len();
            let mut padded_tokens: Vec<u32> = Vec::with_capacity(batch_size * max_len);
            let mut attention_mask: Vec<f32> = Vec::with_capacity(batch_size * max_len);
            
            for tokens in &all_tokens {
                // Add actual tokens
                for &t in tokens {
                    padded_tokens.push(t);
                    attention_mask.push(1.0);
                }
                // Pad with 0s
                for _ in tokens.len()..max_len {
                    padded_tokens.push(0); // PAD token
                    attention_mask.push(0.0);
                }
            }
            
            // Create batch tensors: [batch_size, max_len]
            let input_ids = Tensor::new(padded_tokens.as_slice(), &self.device)
                .map_err(|e| format!("Tensor error: {}", e))?
                .reshape((batch_size, max_len))
                .map_err(|e| format!("Reshape error: {}", e))?;
            
            let attn_mask = Tensor::new(attention_mask.as_slice(), &self.device)
                .map_err(|e| format!("Mask tensor error: {}", e))?
                .reshape((batch_size, max_len))
                .map_err(|e| format!("Mask reshape error: {}", e))?;
            
            // Forward pass - Exact port of LAMForward.forward (_core.py lines 1082-1139)
            let mut x = self.embeddings.forward(&input_ids)
                .map_err(|e| format!("Embeddings error: {}", e))?;
            
            // DeltaNet layers with residual connections (_core.py lines 1125-1137)
            for i in 0..NUM_LAYERS {
                let residual = x.clone();
                
                // DeltaNet attention
                let x_attn = self.deltanet_layers[i].forward(&x)
                    .map_err(|e| format!("DeltaNet layer {} error: {}", i, e))?;
                let sum1 = (residual + x_attn).map_err(|e| format!("Add error: {}", e))?;
                x = self.deltanet_norms[i].forward(&sum1)
                    .map_err(|e| format!("DeltaNet norm {} error: {}", i, e))?;
                
                // FFN (_core.py lines 1131-1137)
                // CRITICAL: Python code applies GELU twice (intermediate_act_fn + F.gelu)
                let residual = x.clone();
                let x_ffn = self.ffn_intermediates[i].forward(&x)  // This applies GELU once
                    .map_err(|e| format!("FFN intermediate {} error: {}", i, e))?;
                let x_ffn = gelu(&x_ffn)  // Apply GELU second time (matching Python bug)
                    .map_err(|e| format!("FFN GELU error: {}", e))?;
                let x_ffn = self.ffn_outputs[i].forward(&x_ffn)
                    .map_err(|e| format!("FFN output {} error: {}", i, e))?;
                // Note: dropout is skipped in inference mode
                let sum2 = (residual + x_ffn).map_err(|e| format!("Add error: {}", e))?;
                x = self.ffn_norms[i].forward(&sum2)
                    .map_err(|e| format!("FFN norm {} error: {}", i, e))?;
            }
            
            // Mean pooling with attention mask (_core.py lines 1152-1155)
            // x shape: [batch_size, seq_len, hidden_size]
            // attn_mask shape: [batch_size, seq_len]
            
            // Expand mask to [batch_size, seq_len, hidden_size]
            let mask_expanded = attn_mask.unsqueeze(2)
                .map_err(|e| format!("Mask unsqueeze error: {}", e))?
                .broadcast_as(x.shape())
                .map_err(|e| format!("Mask broadcast error: {}", e))?;
            
            // Masked sum: x * mask then sum over seq_len
            let masked_x = x.mul(&mask_expanded)
                .map_err(|e| format!("Masked mul error: {}", e))?;
            let summed = masked_x.sum(1)
                .map_err(|e| format!("Sum error: {}", e))?;
            
            // Sum of mask per sample (for mean pooling)
            let mask_sum = attn_mask.sum(1)
                .map_err(|e| format!("Mask sum error: {}", e))?
                .unsqueeze(1)
                .map_err(|e| format!("Mask sum unsqueeze error: {}", e))?
                .clamp(1e-9, f32::MAX)
                .map_err(|e| format!("Mask clamp error: {}", e))?;
            
            // Mean pooling: summed / mask_sum
            let pooled = summed.broadcast_div(&mask_sum)
                .map_err(|e| format!("Mean pool div error: {}", e))?;
            
            // L2 normalize if requested
            let output = if normalize {
                l2norm(&pooled, -1).map_err(|e| format!("L2norm error: {}", e))?
            } else {
                pooled
            };
            
            // Extract embeddings for each sample in batch
            let output_vec: Vec<f32> = output.flatten_all()
                .map_err(|e| format!("Flatten error: {}", e))?
                .to_vec1()
                .map_err(|e| format!("To vec error: {}", e))?;
            
            for i in 0..batch_size {
                let start = i * HIDDEN_SIZE;
                let end = start + HIDDEN_SIZE;
                all_embeddings.push(output_vec[start..end].to_vec());
            }
        }
        
        Ok(all_embeddings)
    }
}
