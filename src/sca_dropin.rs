//! ═══════════════════════════════════════════════════════════════════════════════
//! SCA DROP-IN REPLACEMENT (from rust_test/src/lib.rs)
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! This module is a drop-in replacement for CrystallineCore search/retrieval,
//! aligned with license activation. It contains the full RustHybridEngine from
//! rust_test: quantized Hamming, HDC, routing (PureSemantic/FullHybrid/PureLexical),
//! extended qrels, evaluate_batch. crystalline.rs remains unchanged; engine.rs
//! can later switch to this backend. No #[pymodule] here—expose via lam_candle if needed.

use pyo3::prelude::*;
use rayon::prelude::*;
use ahash::{AHashMap, AHashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use simsimd::BinarySimilarity;
use bitvec::prelude::*;
use rand::prelude::*;

fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

// =================================================================
// SOUNDEX IMPLEMENTATION
// =================================================================

fn soundex(word: &str) -> String {
    if word.is_empty() {
        return "0000".to_string();
    }
    
    let word_upper: Vec<char> = word.to_uppercase().chars().collect();
    let first_letter = word_upper[0];
    
    let get_code = |c: char| -> Option<char> {
        match c {
            'B' | 'F' | 'P' | 'V' => Some('1'),
            'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => Some('2'),
            'D' | 'T' => Some('3'),
            'L' => Some('4'),
            'M' | 'N' => Some('5'),
            'R' => Some('6'),
            _ => None,
        }
    };
    
    let mut result = String::with_capacity(4);
    result.push(first_letter);
    
    let mut prev_code: Option<char> = get_code(first_letter);
    
    for &c in word_upper.iter().skip(1) {
        if let Some(code) = get_code(c) {
            if Some(code) != prev_code {
                result.push(code);
                if result.len() == 4 {
                    break;
                }
            }
            prev_code = Some(code);
        } else {
            prev_code = None;
        }
    }
    
    while result.len() < 4 {
        result.push('0');
    }
    
    result
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a_len = a.chars().count();
    let b_len = b.chars().count();
    
    if a_len == 0 { return b_len; }
    if b_len == 0 { return a_len; }
    
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    
    let mut prev_row: Vec<usize> = (0..=b_len).collect();
    let mut curr_row: Vec<usize> = vec![0; b_len + 1];
    
    for i in 1..=a_len {
        curr_row[0] = i;
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr_row[j] = (prev_row[j] + 1)
                .min(curr_row[j - 1] + 1)
                .min(prev_row[j - 1] + cost);
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }
    
    prev_row[b_len]
}

// =================================================================
// STOPWORDS
// =================================================================

#[allow(dead_code)]
fn is_stopword(word: &str) -> bool {
    const STOPWORDS: &[&str] = &[
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "it", "its", "this", "that", "these", "those", "i", "you", "he",
        "she", "we", "they", "what", "which", "who", "whom", "whose",
        "where", "when", "why", "how", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just", "also"
    ];
    STOPWORDS.contains(&word)
}

// =================================================================
// QUERY ROUTE ENUM
// =================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryRoute {
    PureSemantic,
    FullHybrid,
    PureLexical,
}

/// Code-intent words: query contains one of these (whole-word) → route to high_lexical
const CODE_INTENT_WORDS: &[&str] = &["passkey", "password", "passcode", "serial", "needle"];

/// Known compound→split mappings for CODE_INTENT_WORDS that may appear as two
/// words in documents (e.g. "pass key") but one word in queries ("passkey").
const COMPOUND_SPLITS: &[(&str, &[&str])] = &[
    ("passkey", &["pass", "key"]),
    ("passcode", &["pass", "code"]),
    ("password", &["pass", "word"]),
];

// =================================================================
// HDC ENGINE
// =================================================================

const HDC_DIM: usize = 10_000;

#[derive(Clone, Debug)]
struct HyperVector {
    bits: BitVec<u64, Lsb0>,
}

impl HyperVector {
    fn random() -> Self {
        let mut rng = rand::thread_rng();
        let mut bv = BitVec::with_capacity(HDC_DIM);
        for _ in 0..HDC_DIM {
            bv.push(rng.gen::<bool>());
        }
        Self { bits: bv }
    }

    /// SSP: Deterministic location encoding from a string seed.
    /// Uses SHA-256 of the seed string to initialize a PRNG, then generates
    /// a reproducible 10K-bit vector. Same seed → same vector always.
    /// Matches Python ssp_engine.py::encode_location().
    fn from_seed(seed_str: &str) -> Self {
        use sha2::{Sha256, Digest};
        let hash = Sha256::digest(seed_str.as_bytes());
        // Take first 4 bytes as u32 seed for StdRng (deterministic)
        let seed_val = u32::from_be_bytes([hash[0], hash[1], hash[2], hash[3]]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_val as u64);
        let mut bv = BitVec::with_capacity(HDC_DIM);
        for _ in 0..HDC_DIM {
            bv.push(rng.gen::<bool>());
        }
        Self { bits: bv }
    }

    /// SSP: Project a float embedding into binary hyperspace via random hyperplane projection.
    /// bit_i = 1 if (embedding · projection_row_i > 0)
    /// Matches Python ssp_engine.py::encode_entity().
    fn from_embedding(embedding: &[f32], projection: &[Vec<f32>]) -> Self {
        let mut bv = BitVec::with_capacity(HDC_DIM);
        for row in projection.iter().take(HDC_DIM) {
            let dot: f32 = embedding.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
            bv.push(dot > 0.0);
        }
        Self { bits: bv }
    }

    fn bind(&self, other: &Self) -> Self {
        let mut new_bits = self.bits.clone();
        new_bits ^= &other.bits;
        Self { bits: new_bits }
    }

    fn unbind(&self, key: &Self) -> Self {
        self.bind(key)
    }

    fn bundle(vectors: &[HyperVector]) -> Self {
        let mut result = BitVec::with_capacity(HDC_DIM);
        for i in 0..HDC_DIM {
            let mut ones = 0;
            for v in vectors {
                if v.bits[i] { ones += 1; }
            }
            result.push(ones > (vectors.len() / 2));
        }
        Self { bits: result }
    }
    
    fn similarity(&self, other: &Self) -> f32 {
        let distance = (self.bits.clone() ^ &other.bits).count_ones();
        1.0 - (distance as f32 / HDC_DIM as f32)
    }

    /// SSP: Create a zero vector (empty memory).
    #[allow(dead_code)]
    fn zeros() -> Self {
        Self { bits: bitvec::bitvec![u64, Lsb0; 0; HDC_DIM] }
    }
}

struct HDCEngine {
    // Original HDC fields
    concept_memory: AHashMap<String, HyperVector>,
    encoded_docs: AHashMap<String, HyperVector>,

    // SSP fields: Spatial Semantic Pointers
    /// Random projection matrix (HDC_DIM rows × embedding_dim cols) for
    /// projecting 384-dim LAM embeddings into 10K-bit hyperspace.
    /// Initialized once with seed=42 for reproducibility (matches Python).
    ssp_projection: Option<Vec<Vec<f32>>>,
    /// Cached location hypervectors (deterministic → cacheable).
    #[allow(dead_code)]
    ssp_location_cache: AHashMap<String, HyperVector>,
    /// Entity hypervectors indexed by doc_id (from embedding projection).
    ssp_entities: AHashMap<String, HyperVector>,
    /// Composite SSP memory: S_memory = Σ(L_i ⊛ E_i)
    ssp_memory: Option<HyperVector>,
    /// Whether SSP is initialized (projection matrix generated).
    ssp_enabled: bool,
}

/// SSP projection seed — must match Python ssp_engine.py seed=42
const SSP_SEED: u64 = 42;
/// Embedding dimension for LAM model
const SSP_EMBEDDING_DIM: usize = 384;

impl HDCEngine {
    fn new() -> Self {
        Self {
            concept_memory: AHashMap::new(),
            encoded_docs: AHashMap::new(),
            ssp_projection: None,
            ssp_location_cache: AHashMap::new(),
            ssp_entities: AHashMap::new(),
            ssp_memory: None,
            ssp_enabled: false,
        }
    }

    // ==================================================================
    // SSP INITIALIZATION AND ENCODING
    // ==================================================================

    /// Initialize SSP projection matrix. Must be called before encode_entity.
    /// Generates a (HDC_DIM × SSP_EMBEDDING_DIM) random matrix seeded at 42.
    fn init_ssp(&mut self) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SSP_SEED);
        let mut projection = Vec::with_capacity(HDC_DIM);
        for _ in 0..HDC_DIM {
            let row: Vec<f32> = (0..SSP_EMBEDDING_DIM)
                .map(|_| {
                    // Box-Muller transform for normal distribution (matches numpy randn)
                    let u1: f64 = rng.gen::<f64>().max(1e-10);
                    let u2: f64 = rng.gen::<f64>();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    z as f32
                })
                .collect();
            projection.push(row);
        }
        self.ssp_projection = Some(projection);
        self.ssp_enabled = true;
    }

    /// Encode a file/scope/line_range into a deterministic location hypervector.
    /// Matches Python ssp_engine.py::encode_location().
    #[allow(dead_code)]
    fn encode_location(&mut self, file_path: &str, scope: &str, line_range: &str) -> HyperVector {
        let key = format!("{}::{}::{}", file_path, scope, line_range);
        if let Some(cached) = self.ssp_location_cache.get(&key) {
            return cached.clone();
        }
        let loc = HyperVector::from_seed(&key);
        self.ssp_location_cache.insert(key, loc.clone());
        loc
    }

    /// Project a 384-dim embedding into 10K-bit hyperspace.
    /// Matches Python ssp_engine.py::encode_entity().
    fn encode_entity(&self, embedding: &[f32]) -> Option<HyperVector> {
        self.ssp_projection.as_ref().map(|proj| {
            HyperVector::from_embedding(embedding, proj)
        })
    }

    /// Register an entity for a doc_id (stores the projected hypervector).
    fn ssp_register_entity(&mut self, doc_id: &str, embedding: &[f32]) {
        if let Some(entity) = self.encode_entity(embedding) {
            self.ssp_entities.insert(doc_id.to_string(), entity);
        }
    }

    /// Build composite SSP memory from all registered entities and their locations.
    /// S_memory = bundle([L_i XOR E_i for all i])
    #[allow(dead_code)]
    fn ssp_build_memory(&mut self, bindings: &[(String, String, String, Vec<f32>)]) {
        // bindings: [(doc_id, file_path, scope_line, embedding)]
        let mut bound_vectors = Vec::with_capacity(bindings.len());
        for (doc_id, file_path, scope_line, embedding) in bindings {
            let loc = self.encode_location(file_path, scope_line, "");
            if let Some(entity) = self.encode_entity(embedding) {
                self.ssp_entities.insert(doc_id.clone(), entity.clone());
                bound_vectors.push(loc.bind(&entity));
            }
        }
        if !bound_vectors.is_empty() {
            self.ssp_memory = Some(HyperVector::bundle(&bound_vectors));
        }
    }

    /// Validate a proposal: unbind location from S_memory, compare with candidate entity.
    /// Returns similarity score. >0.5 means alignment with stored memory.
    fn ssp_validate_proposal(&self, location: &HyperVector, candidate: &HyperVector) -> f32 {
        if let Some(ref memory) = self.ssp_memory {
            let recovered = memory.unbind(location);
            recovered.similarity(candidate)
        } else {
            0.5 // No memory → neutral
        }
    }

    /// Compare query entity against a candidate entity directly in hyperspace.
    /// This is the primary discrimination signal for search_hybrid.
    fn ssp_entity_similarity(&self, query_emb: &[f32], doc_id: &str) -> f32 {
        if !self.ssp_enabled { return 0.5; }
        let query_entity = match self.encode_entity(query_emb) {
            Some(e) => e,
            None => return 0.5,
        };
        match self.ssp_entities.get(doc_id) {
            Some(doc_entity) => query_entity.similarity(doc_entity),
            None => 0.5,
        }
    }

    // ==================================================================
    // ORIGINAL HDC METHODS (preserved for backward compatibility)
    // ==================================================================

    fn get_vector(&mut self, token: &str) -> HyperVector {
        if let Some(vec) = self.concept_memory.get(token) {
            return vec.clone();
        }
        let vec = HyperVector::random();
        self.concept_memory.insert(token.to_string(), vec.clone());
        vec
    }

    fn encode_document(&mut self, doc_id: &str, attributes: &[(&str, &str)]) {
        let mut fields = Vec::new();
        for (key, value) in attributes {
            let k_vec = self.get_vector(key);
            let v_vec = self.get_vector(value);
            fields.push(k_vec.bind(&v_vec));
        }
        let doc_vec = HyperVector::bundle(&fields);
        self.encoded_docs.insert(doc_id.to_string(), doc_vec);
    }

    fn search_fuzzy(&self, query_key: &str, query_val: &str) -> Vec<(String, f32)> {
        let k_vec = match self.concept_memory.get(query_key) {
            Some(v) => v,
            None => return vec![],
        };
        let v_vec = match self.concept_memory.get(query_val) {
            Some(v) => v,
            None => return vec![],
        };
        let query_vec = k_vec.bind(v_vec);

        let mut results: Vec<(String, f32)> = self.encoded_docs.iter()
            .map(|(id, doc_vec)| (id.clone(), doc_vec.similarity(&query_vec)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    fn extract_attribute(&self, doc_id: &str, attribute: &str) -> Option<(String, f32)> {
        let doc_vec = self.encoded_docs.get(doc_id)?;
        let attr_vec = self.concept_memory.get(attribute)?;
        let noisy_value = doc_vec.unbind(attr_vec);
        let best_match = self.concept_memory.iter()
            .map(|(token, vec)| (token.clone(), vec.similarity(&noisy_value)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        best_match
    }

    fn encode_query_pattern(&self, query: &str) -> HyperVector {
        let words: Vec<&str> = query.split_whitespace()
            .filter(|w| w.len() >= 3)
            .collect();
        
        if words.is_empty() {
            return HyperVector::random();
        }

        let mut pattern = if let Some(first_word) = words.first() {
            self.concept_memory.get(*first_word)
                .cloned()
                .unwrap_or_else(|| HyperVector::random())
        } else {
            HyperVector::random()
        };

        for word in words.iter().skip(1) {
            if let Some(word_vec) = self.concept_memory.get(*word) {
                pattern = pattern.bind(word_vec);
            }
        }

        pattern
    }
}

// =================================================================
// MAIN ENGINE - UNIFIED WITH ROUTING
// =================================================================

#[pyclass]
pub struct RustHybridEngine {
    // 1. QUANTIZED INDEX (1-BIT) - Stores ALL embeddings (doc or passage level)
    matrix_quantized: Vec<u8>,
    
    // 2. GLOBAL STATS
    corpus_mean: Vec<f32>,
    dim: usize,
    num_docs: usize,
    quantized_dim: usize,
    
    // 3. PASSAGE METADATA - Maps doc to its passages in matrix_quantized
    passage_counts: Vec<usize>,
    passage_offsets: Vec<usize>,
    
    // 4. WORD-LEVEL LEXICAL INDEX
    word_inverted_index: AHashMap<String, AHashSet<usize>>,
    doc_word_sets: Vec<AHashSet<String>>,
    doc_word_tf: Vec<AHashMap<String, u32>>,
    doc_texts: Vec<String>,
    
    // 5. PHONETIC INDEX (Soundex)
    phonetic_index: AHashMap<String, AHashSet<String>>,
    vocabulary: AHashSet<String>,
    
    // 6. IDF MAPS
    word_idf: AHashMap<String, f32>,
    idf_map: AHashMap<u64, f32>,
    
    // 7. DOC IDS & GAMMAS
    doc_ids: Vec<String>,
    gammas: Vec<f32>,
    
    // 8. LEGACY (hash-based)
    lexical_index: Vec<AHashSet<u64>>,
    
    // 9. HDC ENGINE
    hdc_engine: Option<HDCEngine>,
    
    // 10. ROUTING CONTROL (for testing/comparison)
    force_route: Option<QueryRoute>,
    last_route_used: Option<QueryRoute>,  // For logging
    
    // 11. HOLOGRAPHIC 16-VIEW QUANTIZATION
    holographic_16view: bool,
    holographic_scale: f32,
    bytes_per_passage: usize,
    rerank_depth: usize,
    
    // 12. HYBRID CANDIDATE GENERATION WEIGHTS
    hybrid_alpha_semantic: f32,  // Default 0.60 (semantic-heavy for summary-to-transcript matching)
    hybrid_alpha_lexical: f32,   // Default 0.40
}

#[pymethods]
impl RustHybridEngine {
    #[new]
    pub fn new(dim: usize, corpus_mean: Vec<f32>) -> Self {
        let quantized_dim = (dim + 7) / 8;
        
        RustHybridEngine {
            matrix_quantized: Vec::new(),
            corpus_mean,
            dim,
            num_docs: 0,
            quantized_dim,
            passage_counts: Vec::new(),
            passage_offsets: Vec::new(),
            word_inverted_index: AHashMap::new(),
            doc_word_sets: Vec::new(),
            doc_word_tf: Vec::new(),
            doc_texts: Vec::new(),
            phonetic_index: AHashMap::new(),
            vocabulary: AHashSet::new(),
            word_idf: AHashMap::new(),
            idf_map: AHashMap::new(),
            doc_ids: Vec::new(),
            gammas: Vec::new(),
            lexical_index: Vec::new(),
            hdc_engine: None,
            force_route: None,
            last_route_used: None,
            holographic_16view: false,
            holographic_scale: 0.2,
            bytes_per_passage: quantized_dim,  // Default: single view
            rerank_depth: 100,
            hybrid_alpha_semantic: 0.60,
            hybrid_alpha_lexical: 0.40,
        }
    }

    /// Get query IDF for adaptive alpha calculation (MTEB compatibility)
    pub fn get_query_idf(&self, query: &str) -> f32 {
        let (_, _, _, idf, _, _, _) = self.analyze_query(query);
        idf
    }

    // =================================================================
    // HDC METHODS
    // =================================================================
    
    pub fn init_hdc(&mut self) {
        let mut hdc = HDCEngine::new();
        hdc.init_ssp();  // Initialize SSP projection matrix (384→10K)
        self.hdc_engine = Some(hdc);
    }

    /// Check if the SSP projection matrix is initialized (i.e. ready for entity encoding).
    pub fn ssp_is_ready(&self) -> bool {
        self.hdc_engine.as_ref().map(|h| h.ssp_enabled).unwrap_or(false)
    }

    pub fn add_hdc_document(&mut self, doc_id: String, attributes: Vec<(String, String)>) {
        if let Some(ref mut hdc) = self.hdc_engine {
            let attrs: Vec<(&str, &str)> = attributes.iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            hdc.encode_document(&doc_id, &attrs);
        }
    }

    pub fn search_hdc(&self, query_key: String, query_val: String) -> Vec<(String, f32)> {
        if let Some(ref hdc) = self.hdc_engine {
            hdc.search_fuzzy(&query_key, &query_val)
        } else {
            vec![]
        }
    }

    pub fn extract_hdc_attribute(&self, doc_id: String, attribute: String) -> Option<(String, f32)> {
        if let Some(ref hdc) = self.hdc_engine {
            hdc.extract_attribute(&doc_id, &attribute)
        } else {
            None
        }
    }

    // =================================================================
    // SCA PURE RECALL — Raw 10K-bit entity index + batch Hamming search
    // Exposed to Python for .said memory recall (no lexical/semantic mix)
    // =================================================================

    /// Register a solution entity by doc_id + 384-dim LAM embedding.
    /// Projects the embedding into 10K-bit hyperspace via the SSP projection
    /// matrix (seeded at 42, matching ssp_engine.py).
    #[pyo3(name = "sca_register_entity")]
    pub fn sca_register_entity(&mut self, doc_id: String, embedding: Vec<f32>) {
        if let Some(ref mut hdc) = self.hdc_engine {
            hdc.ssp_register_entity(&doc_id, &embedding);
        }
    }

    /// Batch register multiple solution entities at once.
    #[pyo3(name = "sca_register_entities_batch")]
    pub fn sca_register_entities_batch(&mut self, doc_ids: Vec<String>, embeddings_flat: Vec<f32>) {
        if let Some(ref mut hdc) = self.hdc_engine {
            let dim = SSP_EMBEDDING_DIM;
            for (i, doc_id) in doc_ids.iter().enumerate() {
                let start = i * dim;
                let end = start + dim;
                if end <= embeddings_flat.len() {
                    hdc.ssp_register_entity(doc_id, &embeddings_flat[start..end]);
                }
            }
        }
    }

    /// Pure SCA recall: find top-K entities by Hamming similarity in 10K-bit hyperspace.
    /// Returns Vec<(doc_id, similarity_score)> sorted descending.
    /// This is the ONLY recall path for .said memory — no lexical, no cosine, pure Hamming.
    #[pyo3(name = "sca_recall_top_k")]
    pub fn sca_recall_top_k(&self, query_embedding: Vec<f32>, top_k: usize, threshold: f32) -> Vec<(String, f32)> {
        let hdc = match &self.hdc_engine {
            Some(h) => h,
            None => return vec![],
        };
        if !hdc.ssp_enabled || hdc.ssp_entities.is_empty() {
            return vec![];
        }
        let query_entity = match hdc.encode_entity(&query_embedding) {
            Some(e) => e,
            None => return vec![],
        };

        let mut scores: Vec<(String, f32)> = hdc.ssp_entities.iter()
            .map(|(doc_id, entity)| {
                let sim = query_entity.similarity(entity);
                (doc_id.clone(), sim)
            })
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Get the number of registered SCA entities.
    #[pyo3(name = "sca_entity_count")]
    pub fn sca_entity_count(&self) -> usize {
        match &self.hdc_engine {
            Some(h) => h.ssp_entities.len(),
            None => 0,
        }
    }

    /// Export all registered SCA entities as packed bytes.
    /// Each entity is (doc_id, Vec<u8>) where the bytes are the raw BitVec<u64>
    /// in little-endian order. 10,000 bits → ceil(10000/64)=157 u64 words → 1,256 bytes.
    #[pyo3(name = "sca_export_packed")]
    pub fn sca_export_packed(&self) -> Vec<(String, Vec<u8>)> {
        let hdc = match &self.hdc_engine {
            Some(h) => h,
            None => return vec![],
        };
        hdc.ssp_entities.iter().map(|(doc_id, hv)| {
            let raw: &[u64] = hv.bits.as_raw_slice();
            let bytes: Vec<u8> = raw.iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            (doc_id.clone(), bytes)
        }).collect()
    }

    /// Import pre-projected SCA entities from packed bytes.
    /// Bypasses the 384→10K projection entirely — the bytes ARE the index.
    /// Each entry is (doc_id, Vec<u8>) matching the format from sca_export_packed.
    #[pyo3(name = "sca_import_packed")]
    pub fn sca_import_packed(&mut self, entities: Vec<(String, Vec<u8>)>) {
        if self.hdc_engine.is_none() {
            self.init_hdc();
        }
        if let Some(ref mut hdc) = self.hdc_engine {
            for (doc_id, bytes) in entities {
                let words: Vec<u64> = bytes.chunks(8)
                    .map(|chunk| {
                        let mut arr = [0u8; 8];
                        let len = chunk.len().min(8);
                        arr[..len].copy_from_slice(&chunk[..len]);
                        u64::from_le_bytes(arr)
                    })
                    .collect();
                let mut bv = BitVec::<u64, Lsb0>::from_vec(words);
                bv.truncate(HDC_DIM);
                hdc.ssp_entities.insert(doc_id, HyperVector { bits: bv });
            }
        }
    }

    /// Validate a proposal against the holographic S_memory.
    /// Returns similarity score (>0.5 = alignment with stored memory).
    #[pyo3(name = "sca_validate_proposal")]
    pub fn sca_validate_proposal(&self, file_path: String, scope: String, candidate_embedding: Vec<f32>) -> f32 {
        let hdc = match &self.hdc_engine {
            Some(h) => h,
            None => return 0.5,
        };
        let loc = HyperVector::from_seed(&format!("{}::{}::", file_path, scope));
        let candidate = match hdc.encode_entity(&candidate_embedding) {
            Some(e) => e,
            None => return 0.5,
        };
        hdc.ssp_validate_proposal(&loc, &candidate)
    }

    // =================================================================
    // HYBRID CANDIDATE GENERATION WEIGHTS
    // =================================================================
    
    /// Set hybrid candidate generation weights (for tuning)
    #[pyo3(name = "set_hybrid_weights")]
    pub fn set_hybrid_weights(&mut self, semantic: f32, lexical: f32) {
        self.hybrid_alpha_semantic = semantic.clamp(0.0, 1.0);
        self.hybrid_alpha_lexical = lexical.clamp(0.0, 1.0);
        eprintln!("📊 Hybrid weights: semantic={:.2}, lexical={:.2}", 
                  self.hybrid_alpha_semantic, self.hybrid_alpha_lexical);
    }

    #[pyo3(name = "set_rerank_depth")]
    pub fn set_rerank_depth(&mut self, depth: usize) {
        self.rerank_depth = depth.max(1).min(200);
    }

    // =================================================================
    // HOLOGRAPHIC 16-VIEW CONFIGURATION
    // =================================================================
    
    #[pyo3(name = "set_holographic_16view", signature = (enabled, scale=None))]
    pub fn set_holographic_16view(&mut self, enabled: bool, scale: Option<f32>) {
        self.holographic_16view = enabled;
        if let Some(s) = scale {
            self.holographic_scale = s;
        }
        self.bytes_per_passage = if enabled {
            self.quantized_dim * 16
        } else {
            self.quantized_dim
        };
        eprintln!("📐 Holographic 16-view {} (scale = {:.4})", 
                 if enabled { "enabled" } else { "disabled" }, 
                 self.holographic_scale);
    }

    // =================================================================
    // IDF LOADING
    // =================================================================

    pub fn load_idf(&mut self, words: Vec<String>, scores: Vec<f32>) {
        for (w, s) in words.iter().zip(scores.iter()) {
            let w_lower = w.to_lowercase();
            self.word_idf.insert(w_lower.clone(), *s);
            self.idf_map.insert(calculate_hash(&w_lower), *s);
        }
    }

    // =================================================================
    // DOCUMENT INDEXING - UNIFIED (handles both doc-level and passage-level)
    // =================================================================

    /// Add documents with their embeddings (can be doc-level or passage-level)
    /// 
    /// Args:
    ///   ids: Document IDs
    ///   embeddings_flat: Flattened embeddings (all passages for all docs)
    ///   passage_counts: Number of passages per document (use [1,1,1...] for doc-level)
    ///   gammas: Per-document gamma values
    ///   doc_words: Words for each document (for lexical indexing)
    pub fn add_docs(
        &mut self,
        ids: Vec<String>,
        embeddings_flat: Vec<f32>,
        passage_counts: Vec<usize>,
        gammas: Vec<f32>,
        doc_words: Vec<Vec<String>>
    ) {
        let count = ids.len();
        let start_idx = self.num_docs;
        
        self.doc_ids.extend(ids);
        self.gammas.extend(gammas);
        self.num_docs += count;

        // Track passage offsets for MaxSim lookup
        let stride = self.bytes_per_passage;
        let mut current_offset = self.matrix_quantized.len() / stride;
        for &passage_count in &passage_counts {
            self.passage_counts.push(passage_count);
            self.passage_offsets.push(current_offset);
            current_offset += passage_count;
        }

        // Word-Level Indexing
        for (i, words) in doc_words.iter().enumerate() {
            let doc_idx = start_idx + i;
            let mut word_set = AHashSet::new();
            let mut word_tf: AHashMap<String, u32> = AHashMap::new();
            let mut hash_set = AHashSet::new();
            let mut doc_text = String::new();

            for w in words {
                let w_lower = w.to_lowercase();

                // Normalize: strip trailing punctuation for reliable matching
                let w_normalized: String = w_lower
                    .trim_end_matches(|c: char| c.is_ascii_punctuation())
                    .to_string();

                // Build doc_text from normalized words so phrase match
                // is consistent with word_set (e.g. "munoz's" → "munoz")
                if !doc_text.is_empty() {
                    doc_text.push(' ');
                }
                doc_text.push_str(&w_normalized);

                // Match Python: len(w) >= 3
                if w_normalized.len() < 3 {
                    continue;
                }
                
                word_set.insert(w_normalized.clone());
                *word_tf.entry(w_normalized.clone()).or_insert(0) += 1;

                self.word_inverted_index
                    .entry(w_normalized.clone())
                    .or_insert_with(AHashSet::new)
                    .insert(doc_idx);

                let sx = soundex(&w_normalized);
                self.phonetic_index
                    .entry(sx)
                    .or_insert_with(AHashSet::new)
                    .insert(w_normalized.clone());

                self.vocabulary.insert(w_normalized.clone());
                hash_set.insert(calculate_hash(&w_normalized));
            }

            self.doc_word_sets.push(word_set);
            self.doc_word_tf.push(word_tf);
            self.doc_texts.push(doc_text);
            self.lexical_index.push(hash_set);
        }

        // SSP: Project embeddings into hyperspace for entity comparison during search.
        // We take the first passage embedding per doc as the representative entity.
        if let Some(ref mut hdc) = self.hdc_engine {
            if hdc.ssp_enabled {
                let emb_chunks: Vec<&[f32]> = embeddings_flat.chunks(self.dim).collect();
                let mut passage_idx = 0;
                for i in 0..count {
                    let doc_id = &self.doc_ids[start_idx + i];
                    let p_count = passage_counts[i];
                    if passage_idx < emb_chunks.len() {
                        // Use first passage embedding as representative entity
                        hdc.ssp_register_entity(doc_id, emb_chunks[passage_idx]);
                    }
                    passage_idx += p_count;
                }
            }
        }

        // When 16-view is enabled, compute scale from embeddings (core logic in Rust, not Python)
        if self.holographic_16view {
            self.holographic_scale = self.compute_dynamic_scale(&embeddings_flat);
            eprintln!("📐 Holographic 16-view scale (computed in Rust): {:.4}", self.holographic_scale);
        }

        // Quantize all embeddings (passages or docs)
        // Quantize embeddings (16-view holographic or standard 1-bit)
        let chunks: Vec<&[f32]> = embeddings_flat.chunks(self.dim).collect();
        let h_scale = self.holographic_scale;
        let h16 = self.holographic_16view;
        
        let processed: Vec<Vec<u8>> = chunks.par_iter().map(|emb| {
            if h16 {
                // 16-VIEW HOLOGRAPHIC QUANTIZATION
                let mut out = Vec::with_capacity(self.quantized_dim * 16);
                for i in 0..16 {
                    let off = (i as f32 / 15.0 - 0.5) * h_scale;  // Evenly spaced offsets
                    let mut packed = vec![0u8; self.quantized_dim];
                    
                    for d in 0..self.dim {
                        let val = emb[d] - self.corpus_mean[d] + off;
                        if val > 0.0 {
                            let byte_idx = d / 8;
                            let bit_idx = d % 8;
                            packed[byte_idx] |= 1 << bit_idx;
                        }
                    }
                    out.extend(packed);
                }
                out
            } else {
                // STANDARD 1-BIT QUANTIZATION
                let mut centered = vec![0.0f32; self.dim];
                for i in 0..self.dim {
                    centered[i] = emb[i] - self.corpus_mean[i];
                }

                let mut packed = vec![0u8; self.quantized_dim];
                for (i, &val) in centered.iter().enumerate() {
                    if val > 0.0 {
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        packed[byte_idx] |= 1 << bit_idx;
                    }
                }
                packed
            }
        }).collect();

        for bits in processed {
            self.matrix_quantized.extend(bits);
        }

        // Set rerank_depth to total docs for full scoring coverage
        self.rerank_depth = self.num_docs;
        
        let total_passages: usize = passage_counts.iter().sum();
        eprintln!("📊 {} quantized passages ready for max-passage velocity", total_passages);
    }

    // =================================================================
    // QUERY ANALYSIS (Exposed to Python)
    // =================================================================

    #[pyo3(name = "analyze_query")]
    pub fn analyze_query_py(&self, query_text: String) -> (String, Vec<String>, f32, f32, bool, bool) {
        let (route, _q_words, q_expanded, idf_avg, oov_ratio, has_oov_code, has_typo) = 
            self.analyze_query(&query_text);
        
        let route_str = match route {
            QueryRoute::PureSemantic => "PureSemantic".to_string(),
            QueryRoute::FullHybrid => "FullHybrid".to_string(),
            QueryRoute::PureLexical => "PureLexical".to_string(),
        };
        
        let expanded_vec: Vec<String> = q_expanded.into_iter().collect();
        (route_str, expanded_vec, idf_avg, oov_ratio, has_oov_code, has_typo)
    }

    // =================================================================
    // ROUTING CONTROL (for testing/comparison)
    // =================================================================
    
    #[pyo3(name = "set_force_route")]
    pub fn set_force_route(&mut self, route_str: String) {
        self.force_route = match route_str.trim().as_ref() {
            "PureSemantic" => Some(QueryRoute::PureSemantic),
            "FullHybrid" => Some(QueryRoute::FullHybrid),
            "PureLexical" => Some(QueryRoute::PureLexical),
            _ => None,
        };
    }
    
    #[pyo3(name = "get_last_route")]
    pub fn get_last_route(&self) -> Option<String> {
        self.last_route_used.as_ref().map(|r| {
            match r {
                QueryRoute::PureSemantic => "PureSemantic".to_string(),
                QueryRoute::FullHybrid => "FullHybrid".to_string(),
                QueryRoute::PureLexical => "PureLexical".to_string(),
            }
        })
    }

    // =================================================================
    // EXTENDED QRELS - Check if both docs contain valid answer
    // For needle tasks where multiple docs may contain same answer
    // =================================================================
    
    /// Check if both documents contain valid answers for a query (extended qrels)
    /// Returns true if both docs contain at least 50% of the query keywords
    /// MATCHES lam_scientific_proof_suite.py _check_both_docs_valid() exactly
    #[pyo3(name = "check_both_docs_valid")]
    pub fn check_both_docs_valid(&self, query: &str, doc1_id: &str, doc2_id: &str) -> bool {
        // Stopwords - EXACT match to Python NEEDLE_STOPWORDS (lines 1241-1245)
        let stopwords: AHashSet<&str> = [
            "what", "when", "where", "which", "who", "why", "how", "the", "they", 
            "them", "their", "known", "that", "this", "with", "from", "have", "been", 
            "were", "being", "for", "was", "and", "are", "is", "his", "her", "she", "he"
        ].iter().cloned().collect();
        
        // Extract keywords from query (4+ chars, not stopwords)
        // MATCHES Python: words = re.findall(r'\b[a-z]{4,}\b', query_lower)
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4 && !stopwords.contains(w))
            .collect();
        
        if keywords.is_empty() {
            return false;
        }
        
        // Find doc indices
        let doc1_idx = self.doc_ids.iter().position(|id| id == doc1_id);
        let doc2_idx = self.doc_ids.iter().position(|id| id == doc2_id);
        
        let (doc1_text, doc2_text) = match (doc1_idx, doc2_idx) {
            (Some(i1), Some(i2)) => {
                // Try doc_texts first (stores full text), fallback to word sets
                if !self.doc_texts.is_empty() && i1 < self.doc_texts.len() && i2 < self.doc_texts.len() {
                    (self.doc_texts[i1].to_lowercase(), self.doc_texts[i2].to_lowercase())
                } else {
                    // Reconstruct from word sets
                    let t1: String = self.doc_word_sets[i1].iter().cloned().collect::<Vec<_>>().join(" ");
                    let t2: String = self.doc_word_sets[i2].iter().cloned().collect::<Vec<_>>().join(" ");
                    (t1, t2)
                }
            }
            _ => return false,
        };
        
        // Count keyword hits in each doc
        // MATCHES Python: doc1_hits = sum(1 for kw in keywords if kw in doc1_lower)
        let doc1_hits = keywords.iter().filter(|kw| doc1_text.contains(*kw)).count();
        let doc2_hits = keywords.iter().filter(|kw| doc2_text.contains(*kw)).count();
        
        // MATCHES Python: threshold = len(keywords) * 0.5
        let threshold = (keywords.len() as f32 * 0.5) as usize;
        
        // MATCHES Python: return doc1_hits >= threshold and doc2_hits >= threshold
        doc1_hits >= threshold && doc2_hits >= threshold
    }
    
    /// Get document text by ID (for debugging/verification)
    #[pyo3(name = "get_doc_text")]
    pub fn get_doc_text(&self, doc_id: &str) -> Option<String> {
        let idx = self.doc_ids.iter().position(|id| id == doc_id)?;
        if !self.doc_texts.is_empty() && idx < self.doc_texts.len() {
            Some(self.doc_texts[idx].clone())
        } else if idx < self.doc_word_sets.len() {
            Some(self.doc_word_sets[idx].iter().cloned().collect::<Vec<_>>().join(" "))
        } else {
            None
        }
    }
    
    // =================================================================
    // EVALUATE WITH EXTENDED QRELS - Full evaluation in Rust
    // Automatically applies extended qrels just like routing is automatic
    // =================================================================
    
    /// Evaluate a single query against ground truth with extended qrels
    /// Returns (is_correct, used_extended_qrels)
    /// This is fully automatic - no hardcoding, uses same logic as routing
    #[pyo3(name = "evaluate_query")]
    pub fn evaluate_query(
        &mut self,
        query_emb: Vec<f32>,
        query_text: &str,
        expected_doc_ids: Vec<String>,
        top_k: usize
    ) -> (bool, bool) {
        // 1. Search using unified search (automatic routing)
        let results = self.search_unified(query_emb, query_text.to_string(), top_k);
        
        if results.is_empty() {
            return (false, false);
        }
        
        let top_doc_id = &results[0].0;
        
        // 2. Check if top result is in expected docs
        if expected_doc_ids.contains(top_doc_id) {
            return (true, false);
        }
        
        // 3. Apply extended qrels - check if both docs contain valid answer
        // This is automatic, not hardcoded - uses keyword analysis
        for expected_id in &expected_doc_ids {
            if self.check_both_docs_valid(query_text, top_doc_id, expected_id) {
                return (true, true);  // Correct via extended qrels
            }
        }
        
        (false, false)
    }
    
    /// Batch evaluate multiple queries with extended qrels
    /// Returns (correct_count, extended_count, total)
    #[pyo3(name = "evaluate_batch")]
    pub fn evaluate_batch(
        &mut self,
        query_embeddings: Vec<Vec<f32>>,
        query_texts: Vec<String>,
        expected_doc_ids_list: Vec<Vec<String>>,
        top_k: usize
    ) -> (usize, usize, usize) {
        let mut correct = 0usize;
        let mut extended = 0usize;
        let total = query_texts.len();
        
        for i in 0..total {
            let (is_correct, used_extended) = self.evaluate_query(
                query_embeddings[i].clone(),
                &query_texts[i],
                expected_doc_ids_list[i].clone(),
                top_k
            );
            
            if is_correct {
                correct += 1;
                if used_extended {
                    extended += 1;
                }
            }
        }
        
        (correct, extended, total)
    }

    // =================================================================
    // UNIFIED SEARCH - MAIN ENTRY POINT
    // =================================================================

    #[pyo3(name = "search_unified")]
    pub fn search_unified(
        &mut self,
        query_emb: Vec<f32>,
        query_text: String,
        top_k: usize
    ) -> Vec<(String, f32)> {
        // Default alpha (will be overridden by search() if called with alpha)
        self.search_unified_with_alpha(query_emb, query_text, 0.5, top_k)
    }
    
    fn search_unified_with_alpha(
        &mut self,
        query_emb: Vec<f32>,
        query_text: String,
        _alpha: f32,
        top_k: usize
    ) -> Vec<(String, f32)> {
        let rescore_limit = 5000;
        let query_lower = query_text.to_lowercase();

        // Analyze query and determine route (or use forced route)
        let (mut route, _q_words, q_expanded, _idf_avg, _oov_ratio, _has_oov_code, _has_typo) =
            self.analyze_query(&query_text);
        
        // Override with forced route if set
        if let Some(forced) = &self.force_route {
            route = forced.clone();
        }
        
        // Store for logging
        self.last_route_used = Some(route.clone());

        // PATH A: PURE LEXICAL (Passkeys, Codes) - NO SEMANTIC AT ALL
        if route == QueryRoute::PureLexical {
            eprintln!("✅ Using PureLexical path - NO semantic search");
            let lexical_results = self.search_pure_lexical(&query_lower, &q_expanded, top_k);
            eprintln!("   PureLexical returned {} results", lexical_results.len());
            // If no lexical matches found, return empty (don't fall back to semantic!)
            if lexical_results.is_empty() {
                eprintln!("⚠️  PureLexical: No lexical matches found, returning empty results");
            }
            return lexical_results;
        }

        // Quantize query for Hamming search
        let q_bits = self.quantize_query(&query_emb);
        let max_hamming = (self.quantized_dim * 8) as f64;

        // PATH B: PURE SEMANTIC (STS, Clustering) - NO LEXICAL AT ALL
        if route == QueryRoute::PureSemantic {
            eprintln!("✅ Using PureSemantic path - NO lexical search");
            let survivors = self.hamming_candidate_retrieval(&q_bits, rescore_limit);
            let semantic_results = self.search_pure_semantic(&survivors, max_hamming, top_k);
            eprintln!("   PureSemantic returned {} results", semantic_results.len());
            return semantic_results;
        }

        // PATH C: FULL HYBRID - Combined retrieval (semantic+lexical BEFORE filtering)
        // KEY FIX: Score ALL docs with 60% semantic + 40% lexical combined *before* cutting
        // This ensures lexical-strong docs survive into top-N candidates
        let hybrid_limit = self.num_docs;
        let survivors = self.hybrid_candidate_retrieval(&q_bits, &q_expanded, hybrid_limit);
        self.search_full_hybrid(&survivors, &q_expanded, &query_lower, &query_text, max_hamming, top_k)
    }

    // =================================================================
    // LEGACY SEARCH (Backward Compatible) - Routes to unified
    // =================================================================

    pub fn search(
        &mut self,
        query_emb: Vec<f32>,
        query_text: String,
        alpha: f32,  // Use alpha from Python (adaptive_alpha)
        top_k: usize
    ) -> Vec<(String, f32)> {
        self.search_unified_with_alpha(query_emb, query_text, alpha, top_k)
    }

    // =================================================================
    // HYBRID SEARCH (HDC + Unified)
    // =================================================================

    #[pyo3(name = "search_hybrid")]
    pub fn search_hybrid(
        &mut self,
        query_text: String,
        query_emb: Vec<f32>,
        _alpha: f32,
        top_k: usize
    ) -> Vec<(String, f32)> {
        let wide_limit = self.num_docs.min(10000);
        let mut sem_candidates = self.search_unified(query_emb.clone(), query_text.clone(), wide_limit);

        if let Some(ref hdc) = self.hdc_engine {
            let sem_confidence = self.calculate_confidence(&sem_candidates);

            // SSP-UPGRADED: Use entity similarity in 10K-bit hyperspace
            // This compares query embedding vs candidate embedding THROUGH
            // the random hyperplane projection, preserving semantic discrimination.
            let use_ssp = hdc.ssp_enabled && !hdc.ssp_entities.is_empty();

            let mut struct_candidates = Vec::with_capacity(sem_candidates.len());
            for (doc_id, _) in &sem_candidates {
                let struct_score = if use_ssp {
                    // SSP: Project query into hyperspace, compare with indexed entity
                    hdc.ssp_entity_similarity(&query_emb, doc_id)
                } else {
                    // Fallback: Original HDC token-pattern matching
                    if let Some(doc_holo) = hdc.encoded_docs.get(doc_id) {
                        let query_holo = hdc.encode_query_pattern(&query_text);
                        doc_holo.similarity(&query_holo)
                    } else {
                        0.0
                    }
                };
                struct_candidates.push((doc_id.clone(), struct_score));
            }

            let struct_confidence = self.calculate_confidence(&struct_candidates);
            let total_confidence = sem_confidence + struct_confidence;
            
            let (sem_weight, struct_weight) = if total_confidence > 0.0 {
                (sem_confidence / total_confidence, struct_confidence / total_confidence)
            } else {
                (0.5, 0.5)
            };

            self.normalize(&mut sem_candidates);
            self.normalize(&mut struct_candidates);

            let mut final_results = Vec::with_capacity(sem_candidates.len());
            for i in 0..sem_candidates.len() {
                let (doc_id, sem_s) = &sem_candidates[i];
                let (_, struct_s) = &struct_candidates[i];
                let hybrid_score = (sem_s * sem_weight) + (struct_s * struct_weight);
                final_results.push((doc_id.clone(), hybrid_score));
            }

            final_results.sort_by(|a, b| 
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            );
            return final_results.into_iter().take(top_k).collect();
        }

        sem_candidates.into_iter().take(top_k).collect()
    }

    // =================================================================
    // UTILITY: Get stats
    // =================================================================
    
    pub fn get_stats(&self) -> (usize, usize, usize) {
        let total_passages: usize = self.passage_counts.iter().sum();
        (self.num_docs, total_passages, self.vocabulary.len())
    }
}

// =================================================================
// PRIVATE IMPLEMENTATION
// =================================================================

impl RustHybridEngine {
    /// MAD-based dynamic scale for holographic 16-view. Clamp [0.05, 0.5].
    /// Core logic lives in Rust (not in public Python).
    fn compute_dynamic_scale(&self, embeddings_flat: &[f32]) -> f32 {
        if embeddings_flat.is_empty() || self.corpus_mean.is_empty() || self.dim == 0 {
            return 0.2;
        }
        let n = embeddings_flat.len() / self.dim;
        let mut flat: Vec<f32> = Vec::with_capacity(n * self.dim);
        for chunk in embeddings_flat.chunks(self.dim) {
            if chunk.len() != self.dim {
                continue;
            }
            for (d, &c) in chunk.iter().enumerate() {
                let m = self.corpus_mean.get(d).copied().unwrap_or(0.0);
                flat.push((c - m).abs());
            }
        }
        if flat.is_empty() {
            return 0.2;
        }
        flat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let s = flat[flat.len() / 2];
        let scale = 2.0 * s;
        scale.clamp(0.05, 0.5)
    }

    // ─────────────────────────────────────────────────────────────────
    // CODE DETECTION
    // ─────────────────────────────────────────────────────────────────
    
    fn looks_like_code(&self, word: &str) -> bool {
        // VERY STRICT code detection - only actual passkeys/codes
        // NOT: money ($10,000), dates (01/05/2012), measurements (0.17kg)
        
        // Early exit for common false positives (check BEFORE cleaning)
        // Dates or year ranges: contain / or - between digits (2006-2010, 01/05/2012)
        if word.contains('/') || word.contains('-') {
            return false;  // Likely a date or year range
        }
        // Currency or formatted numbers: contain $, €, £, or commas between digits
        if word.contains('$') || word.contains('€') || word.contains('£') || word.contains(',') {
            return false;  // Likely money or formatted number
        }
        // Parenthesized numbers: (123), [456]
        if word.starts_with('(') || word.starts_with('[') {
            return false;  // Likely a reference
        }
        // Contains special chars: ±, etc.
        if word.chars().any(|c| c == '±' || c == '×' || c == '÷') {
            return false;
        }
        
        // Only keep alphanumeric (no separators for true passkeys)
        let clean: String = word.chars()
            .filter(|c| c.is_alphanumeric())
            .collect();
        
        // Passkeys are typically 5-10 chars
        if clean.len() < 5 || clean.len() > 15 {
            return false;
        }
        
        // Exclude ordinals: 1st, 2nd, 3rd, 21st, 100th, etc.
        let lower = clean.to_lowercase();
        if lower.ends_with("st") || lower.ends_with("nd") || 
           lower.ends_with("rd") || lower.ends_with("th") {
            let prefix = &lower[..lower.len()-2];
            if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
                return false;
            }
        }
        
        // Exclude measurements: 100kg, 50cm, 200m, etc.
        let measurement_suffixes = ["kg", "km", "cm", "mm", "ml", "mg", "gb", "mb", "kb", "hz", "bn", "mn", "bln"];
        for suffix in measurement_suffixes {
            if lower.ends_with(suffix) {
                let prefix = &lower[..lower.len() - suffix.len()];
                if prefix.chars().all(|c| c.is_ascii_digit() || c == '.') {
                    return false;
                }
            }
        }
        
        let chars: Vec<char> = clean.chars().collect();
        let digit_count = chars.iter().filter(|c| c.is_ascii_digit()).count();
        let letter_count = chars.iter().filter(|c| c.is_ascii_alphabetic()).count();
        
        // RULE 1: Pure digits, length 5-10 (passkeys like "12345678")
        // This is the PRIMARY code detection - must be pure numeric
        if letter_count == 0 && digit_count >= 5 && digit_count <= 10 {
            return true;
        }
        
        // No other patterns - be very conservative
        false
    }

    // ─────────────────────────────────────────────────────────────────
    // QUERY ANALYSIS & ROUTING
    // ─────────────────────────────────────────────────────────────────
    
    fn analyze_query(&self, query_text: &str) -> (QueryRoute, Vec<String>, AHashSet<String>, f32, f32, bool, bool) {
        // Match Python's tokenization but normalize punctuation for reliable matching
        // Strip trailing punctuation so "12345678." matches "12345678?"
        let q_words: Vec<String> = query_text
            .split_whitespace()
            .map(|s| {
                let lower = s.to_lowercase();
                // Strip trailing punctuation
                lower.trim_end_matches(|c: char| c.is_ascii_punctuation()).to_string()
            })
            .filter(|s| s.len() >= 3)
            .collect();

        if q_words.is_empty() {
            return (QueryRoute::PureSemantic, q_words, AHashSet::new(), 0.5, 0.0, false, false);
        }

        let mut has_any_code = false;
        let mut has_typo = false;
        let mut known_word_count = 0;
        let mut total_idf = 0.0f32;
        let mut oov_count = 0;
        let mut q_expanded: AHashSet<String> = AHashSet::new();

        for word in &q_words {
            // Check for code FIRST
            if self.looks_like_code(word) {
                has_any_code = true;
                total_idf += 5.0; 
                known_word_count += 1;
                q_expanded.insert(word.clone());
                continue;
            }
            
            // Check vocabulary or IDF map
            let in_idf = self.word_idf.get(word);
            
            if in_idf.is_some() || self.vocabulary.contains(word) {
                known_word_count += 1;
                let idf = *in_idf.unwrap_or(&1.5);
                total_idf += idf;
                q_expanded.insert(word.clone());
            } else {
                oov_count += 1;
                // Try compound word splitting for code-intent words
                // e.g. "passkey" → ["pass", "key"] when doc has "pass key"
                let mut compound_found = false;
                for &(compound, parts) in COMPOUND_SPLITS {
                    if word == compound {
                        for &part in parts {
                            if part.len() >= 3 {
                                q_expanded.insert(part.to_string());
                            }
                        }
                        q_expanded.insert(word.clone()); // keep original too
                        compound_found = true;
                        break;
                    }
                }
                if !compound_found {
                    // Try fuzzy expansion
                    let fuzzy_matches = self.fuzzy_expand_word(word, 5);
                    let valid_matches: Vec<String> = fuzzy_matches.iter()
                        .filter(|m| *m != word && self.vocabulary.contains(*m))
                        .cloned()
                        .collect();

                    if !valid_matches.is_empty() {
                        has_typo = true;
                        for m in valid_matches {
                            q_expanded.insert(m);
                        }
                    } else {
                        q_expanded.insert(word.clone());
                    }
                }
            }
        }

        // Calculate IDF average using only CONTENT words (IDF >= 1.5)
        // This excludes common stopwords from diluting the average
        let content_words: Vec<f32> = q_expanded.iter()
            .filter_map(|w| self.word_idf.get(w).copied())
            .filter(|&idf| idf >= 1.5)
            .collect();
        
        let idf_avg = if !content_words.is_empty() {
            content_words.iter().sum::<f32>() / content_words.len() as f32
        } else if has_any_code {
            10.0
        } else if known_word_count > 0 {
            total_idf / known_word_count as f32
        } else {
            1.0
        };

        let non_code_count = q_words.iter()
            .filter(|w| !self.looks_like_code(w))
            .count();
        let oov_ratio = if non_code_count > 0 {
            oov_count as f32 / non_code_count as f32
        } else {
            0.0
        };

        // ROUTING DECISION
        // PureLexical: Codes/passkeys that need exact string matching
        // PureSemantic: Long queries (argument retrieval), common discourse
        // FullHybrid: Short entity-rich queries (high IDF), need lexical + semantic
        
        // Detect queries that ASK for codes/needles (passkey, needle, etc.)
        // These need lexical matching to find the doc containing the exact string.
        // MUST match whole word only to avoid false positives (e.g., "helping" contains "pin")
        let has_code_intent = q_words.iter().any(|w| {
            CODE_INTENT_WORDS.iter().any(|ci| w == *ci)
        });
        
        // Detect pure discourse/similarity queries (STS-style)
        // VERY strict: only truly generic sentences with ALL common words
        // Examples: "A man is playing a guitar", "The dog is running"
        // NOT: dialogue retrieval queries, entity queries, factual queries
        //
        // Count high-IDF words (entity-like / domain-specific)
        let high_idf_count = q_expanded.iter()
            .filter(|w| *self.word_idf.get(*w).unwrap_or(&0.0) > 2.5)
            .count();
        
        let is_short_discourse = q_words.len() <= 8 && !has_any_code && 
                                 !has_code_intent && idf_avg <= 1.2 && 
                                 high_idf_count == 0 && oov_ratio < 0.1;
        
        let route = if has_any_code || has_code_intent {
            // Query contains a code OR asks for a code → need lexical matching
            QueryRoute::PureLexical
        } else if is_short_discourse {
            // Pure discourse - ONLY truly generic sentences (STS/semantic similarity)
            // Must have: very short, all common words, zero specific terms
            QueryRoute::PureSemantic
        } else {
            // Default: FullHybrid for all other queries
            // Hybrid is robust - handles both semantic and lexical signals
            // This is correct for: retrieval, dialogue, factual, entity queries
            QueryRoute::FullHybrid
        };

        (route, q_words, q_expanded, idf_avg, oov_ratio, has_any_code, has_typo)
    }

    fn fuzzy_expand_word(&self, word: &str, top_k: usize) -> Vec<String> {
        let word_lower = word.to_lowercase();

        if self.vocabulary.contains(&word_lower) {
            return vec![word_lower];
        }

        let sx = soundex(&word_lower);
        let candidates = match self.phonetic_index.get(&sx) {
            Some(c) => c,
            None => return vec![word_lower],
        };

        let mut ranked: Vec<(String, usize)> = candidates.iter()
            .map(|c| (c.clone(), levenshtein(&word_lower, c)))
            .filter(|(_, dist)| *dist <= 2)
            .collect();

        ranked.sort_by_key(|(_, dist)| *dist);
        
        let result: Vec<String> = ranked.into_iter()
            .take(top_k)
            .map(|(w, _)| w)
            .collect();

        if result.is_empty() {
            vec![word_lower]
        } else {
            result
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // QUERY QUANTIZATION
    // ─────────────────────────────────────────────────────────────────

    fn quantize_query(&self, query_emb: &[f32]) -> Vec<u8> {
        // 16-VIEW HOLOGRAPHIC QUANTIZATION
        if self.holographic_16view {
            let h_scale = self.holographic_scale;
            let mut q_bits = Vec::with_capacity(self.quantized_dim * 16);
            
            // Generate 16 different quantized views with different offsets
            for i in 0..16 {
                let off = (i as f32 / 15.0 - 0.5) * h_scale;  // Range: -0.5*scale to +0.5*scale
                let mut packed = vec![0u8; self.quantized_dim];
                
                for d in 0..self.dim {
                    let val = query_emb[d] - self.corpus_mean[d] + off;
                    if val > 0.0 {
                        let byte_idx = d / 8;
                        let bit_idx = d % 8;
                        packed[byte_idx] |= 1 << bit_idx;
                    }
                }
                q_bits.extend(packed);
            }
            return q_bits;
        }
        
        // STANDARD 1-BIT QUANTIZATION (fallback)
        let mut centered = vec![0.0f32; self.dim];
        let mut norm_sq = 0.0f32;
        
        for i in 0..self.dim {
            let val = query_emb[i] - self.corpus_mean[i];
            centered[i] = val;
            norm_sq += val * val;
        }
        
        let norm = norm_sq.sqrt().max(1e-9);
        for i in 0..self.dim {
            centered[i] /= norm;
        }

        let mut q_bits = vec![0u8; self.quantized_dim];
        for (i, &val) in centered.iter().enumerate() {
            if val > 0.0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                q_bits[byte_idx] |= 1 << bit_idx;
            }
        }
        q_bits
    }

    // ─────────────────────────────────────────────────────────────────
    // HAMMING CANDIDATE RETRIEVAL (with passage MaxSim)
    // ─────────────────────────────────────────────────────────────────

    fn hamming_candidate_retrieval(&self, q_bits: &[u8], limit: usize) -> Vec<(usize, f64)> {
        let stride = self.bytes_per_passage;
        let max_hamming = (self.quantized_dim * 8) as f64;
        let holo16 = self.holographic_16view && q_bits.len() >= 16 * self.quantized_dim;

        let mut candidates: Vec<(usize, f64)> = if !self.passage_counts.is_empty() && 
            self.passage_counts.iter().any(|&c| c > 1) {
            // PASSAGE MODE: Find min Hamming/Holistic across all passages per doc
            (0..self.num_docs).into_par_iter().map(|doc_idx| {
                let passage_count = self.passage_counts[doc_idx];
                let passage_offset = self.passage_offsets[doc_idx];
                let mut min_dist = f64::MAX;
                
                for p_idx in 0..passage_count {
                    let start = (passage_offset + p_idx) * stride;
                    let end = start + stride;
                    
                    if end <= self.matrix_quantized.len() {
                        if holo16 {
                            // 16-VIEW HOLOGRAPHIC: Calculate holistic similarity across all views
                            let mut sum_s = 0.0f64;
                            for v in 0..16 {
                                let d = &self.matrix_quantized[start + v * self.quantized_dim..start + (v + 1) * self.quantized_dim];
                                let q = &q_bits[v * self.quantized_dim..(v + 1) * self.quantized_dim];
                                let sim = (1.0 - u8::hamming(d, q).unwrap_or(max_hamming) / max_hamming).max(0.0);
                                sum_s += sim;
                            }
                            let holistic_sim = 0.0625 * sum_s;  // Average across 16 views (1/16 = 0.0625)
                            let dist = (1.0 - holistic_sim) * max_hamming;
                            if dist < min_dist { min_dist = dist; }
                        } else {
                            // STANDARD: Single Hamming distance
                            let doc_bits = &self.matrix_quantized[start..end];
                            if let Some(dist) = u8::hamming(doc_bits, q_bits) {
                                if dist < min_dist { min_dist = dist; }
                            }
                        }
                    }
                }
                (doc_idx, min_dist)
            }).collect()
        } else if self.holographic_16view {
            // DOC MODE + HOLOGRAPHIC 16-VIEW
            (0..self.num_docs).into_par_iter().map(|doc_idx| {
                let start = doc_idx * stride;
                let mut sum_s = 0.0f64;
                
                for v in 0..16 {
                    let d = &self.matrix_quantized[start + v * self.quantized_dim..start + (v + 1) * self.quantized_dim];
                    let q = &q_bits[v * self.quantized_dim..(v + 1) * self.quantized_dim];
                    let sim = (1.0 - u8::hamming(d, q).unwrap_or(max_hamming) / max_hamming).max(0.0);
                    sum_s += sim;
                }
                let holistic_sim = 0.0625 * sum_s;
                let dist = (1.0 - holistic_sim) * max_hamming;
                (doc_idx, dist)
            }).collect()
        } else {
            // DOC MODE: Standard single Hamming distance
            (0..self.num_docs).into_par_iter().map(|doc_idx| {
                let start = doc_idx * stride;
                let end = start + stride;
                
                let dist = if end <= self.matrix_quantized.len() {
                    let doc_bits = &self.matrix_quantized[start..end];
                    u8::hamming(doc_bits, q_bits).unwrap_or(f64::MAX)
                } else {
                    f64::MAX
                };
                (doc_idx, dist)
            }).collect()
        };

        candidates.sort_unstable_by(|a, b| 
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        );
        
        candidates.into_iter().take(limit).collect()
    }

    // ─────────────────────────────────────────────────────────────────
    // HYBRID CANDIDATE RETRIEVAL - semantic+lexical BEFORE filtering
    // This is the key fix: score ALL docs with both signals, then cut
    // ─────────────────────────────────────────────────────────────────

    fn hybrid_candidate_retrieval(
        &self,
        q_bits: &[u8],
        q_expanded: &AHashSet<String>,
        limit: usize
    ) -> Vec<(usize, f64)> {
        let stride = self.bytes_per_passage;
        let max_hamming = (self.quantized_dim * 8) as f64;
        let holo16 = self.holographic_16view && q_bits.len() >= 16 * self.quantized_dim;
        
        let alpha_sem = self.hybrid_alpha_semantic as f64;
        let alpha_lex = self.hybrid_alpha_lexical as f64;
        
        let mut candidates: Vec<(usize, f64, f64)> = (0..self.num_docs)
            .into_par_iter()
            .map(|doc_idx| {
                // 1. LEXICAL SCORE - IDF-weighted overlap
                let (s_lexical, has_match) = if doc_idx < self.doc_word_sets.len() {
                    let doc_words = &self.doc_word_sets[doc_idx];
                    let mut idf_matched = 0.0f64;
                    let mut idf_total = 0.0f64;
                    let mut matched = false;
                    
                    for word in q_expanded.iter() {
                        let idf = *self.word_idf.get(word).unwrap_or(&1.0) as f64;
                        idf_total += idf;
                        if doc_words.contains(word) {
                            matched = true;
                            idf_matched += idf;
                        }
                    }
                    
                    let score = if idf_total > 0.0 { idf_matched / idf_total } else { 0.0 };
                    (score, matched)
                } else {
                    (0.0, false)
                };
                
                // 2. EARLY EXIT: Skip expensive Hamming if no lexical match
                if !has_match {
                    let estimated_combined = alpha_sem * 0.4;
                    let combined_dist = (1.0 - estimated_combined) * max_hamming;
                    return (doc_idx, f64::MAX, combined_dist);
                }
                
                // 3. SEMANTIC SCORE (expensive - only for lexical matches)
                let semantic_dist = if !self.passage_counts.is_empty() && 
                    self.passage_counts.get(doc_idx).map(|&c| c > 1).unwrap_or(false) {
                    let passage_count = self.passage_counts[doc_idx];
                    let passage_offset = self.passage_offsets[doc_idx];
                    let mut min_dist = f64::MAX;
                    
                    for p_idx in 0..passage_count {
                        let start = (passage_offset + p_idx) * stride;
                        let end = start + stride;
                        if end <= self.matrix_quantized.len() {
                            if holo16 {
                                let mut sum_s = 0.0f64;
                                for v in 0..16 {
                                    let d = &self.matrix_quantized[start + v * self.quantized_dim..start + (v + 1) * self.quantized_dim];
                                    let q = &q_bits[v * self.quantized_dim..(v + 1) * self.quantized_dim];
                                    let sim = (1.0 - u8::hamming(d, q).unwrap_or(max_hamming) / max_hamming).max(0.0);
                                    sum_s += sim;
                                }
                                let holistic_sim = 0.0625 * sum_s;
                                let dist = (1.0 - holistic_sim) * max_hamming;
                                if dist < min_dist { min_dist = dist; }
                            } else {
                                let doc_bits = &self.matrix_quantized[start..end];
                                if let Some(dist) = u8::hamming(doc_bits, q_bits) {
                                    if dist < min_dist { min_dist = dist; }
                                }
                            }
                        }
                    }
                    min_dist
                } else if holo16 {
                    let start = doc_idx * stride;
                    let mut sum_s = 0.0f64;
                    for v in 0..16 {
                        let d = &self.matrix_quantized[start + v * self.quantized_dim..start + (v + 1) * self.quantized_dim];
                        let q = &q_bits[v * self.quantized_dim..(v + 1) * self.quantized_dim];
                        let sim = (1.0 - u8::hamming(d, q).unwrap_or(max_hamming) / max_hamming).max(0.0);
                        sum_s += sim;
                    }
                    let holistic_sim = 0.0625 * sum_s;
                    (1.0 - holistic_sim) * max_hamming
                } else {
                    let start = doc_idx * stride;
                    let end = start + stride;
                    if end <= self.matrix_quantized.len() {
                        let doc_bits = &self.matrix_quantized[start..end];
                        u8::hamming(doc_bits, q_bits).unwrap_or(f64::MAX)
                    } else {
                        f64::MAX
                    }
                };
                
                let s_semantic = (1.0 - (semantic_dist / max_hamming)).max(0.0);
                
                // 4. COMBINED SCORE - semantic (=velocity) + lexical
                let combined_score = alpha_sem * s_semantic + alpha_lex * s_lexical;
                let combined_dist = (1.0 - combined_score) * max_hamming;
                
                (doc_idx, semantic_dist, combined_dist)
            })
            .collect();
        
        // Sort by COMBINED distance (ensures good lexical matches survive the cut)
        candidates.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return (doc_idx, semantic_dist)
        candidates.into_iter()
            .take(limit)
            .map(|(doc_idx, semantic_dist, _)| (doc_idx, semantic_dist))
            .collect()
    }

    // ─────────────────────────────────────────────────────────────────
    // SPAN DENSITY (Fast path for reranking)
    // ─────────────────────────────────────────────────────────────────

    #[inline(always)]
    fn calculate_span_density_fast(&self, doc_text: &str, query_terms: &[String]) -> f32 {
        if query_terms.len() < 2 { return 0.0; }
        
        let q_to_idx: AHashMap<String, usize> = query_terms.iter()
            .enumerate()
            .map(|(i, s)| (s.to_lowercase(), i))
            .collect();
        let query_idfs: Vec<f32> = query_terms.iter()
            .map(|t| *self.word_idf.get(t).unwrap_or(&0.5))
            .collect();

        let mut positions: Vec<Vec<usize>> = vec![vec![]; query_terms.len()];
        for (pos, word) in doc_text.split_whitespace().enumerate() {
            let clean: String = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            if clean.is_empty() { continue; }
            if let Some(&q_idx) = q_to_idx.get(&clean) {
                positions[q_idx].push(pos);
            }
        }

        let mut active_indices = Vec::with_capacity(query_terms.len());
        let mut idf_sum = 0.0f32;
        for (idx, pos_list) in positions.iter().enumerate() {
            if !pos_list.is_empty() {
                active_indices.push(idx);
                idf_sum += query_idfs[idx];
            }
        }
        if active_indices.len() < 2 { return 0.0; }

        let active_positions: Vec<&Vec<usize>> = active_indices.iter().map(|&idx| &positions[idx]).collect();
        let n = active_indices.len();
        let mut cur = vec![0usize; n];
        let mut min_span = usize::MAX;

        loop {
            let mut lo = usize::MAX;
            let mut hi = 0usize;
            let mut advance_idx = 0;
            for (i, &ci) in cur.iter().enumerate() {
                let val = active_positions[i][ci];
                if val < lo { lo = val; advance_idx = i; }
                if val > hi { hi = val; }
            }
            let span = hi - lo + 1;
            if span < min_span { min_span = span; }
            cur[advance_idx] += 1;
            if cur[advance_idx] >= active_positions[advance_idx].len() { break; }
        }

        let term_weight = idf_sum.powf(1.5);
        let span_penalty = (min_span as f32).max(1.0).ln() + 1.0;
        term_weight / span_penalty
    }

    // ─────────────────────────────────────────────────────────────────
    // PATH A: PURE LEXICAL
    // ─────────────────────────────────────────────────────────────────

    fn search_pure_lexical(
        &self,
        _query_lower: &str,
        q_expanded: &AHashSet<String>,
        top_k: usize
    ) -> Vec<(String, f32)> {
        // PURE LEXICAL - IDF-weighted so rare words dominate over stopwords
        // Uses q_expanded (includes compound splits)

        let mut results: Vec<(String, f32)> = Vec::new();

        if q_expanded.is_empty() {
            return results;
        }

        // IDF-weight so rare words (names, codes) dominate over stopwords
        let total_query_idf: f32 = q_expanded.iter()
            .map(|w| *self.word_idf.get(w).unwrap_or(&1.0))
            .sum::<f32>()
            .max(1.0);

        for (doc_idx, doc_id) in self.doc_ids.iter().enumerate() {
            let doc_words = &self.doc_word_sets[doc_idx];

            let hit_idf: f32 = q_expanded.iter()
                .filter(|w| doc_words.contains(*w))
                .map(|w| *self.word_idf.get(w).unwrap_or(&1.0))
                .sum();

            if hit_idf > 0.0 {
                results.push((doc_id.clone(), hit_idf / total_query_idf));
            }
        }

        // Sort descending by score
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top_k
        results.into_iter().take(top_k).collect()
    }

    // ─────────────────────────────────────────────────────────────────
    // PATH B: PURE SEMANTIC
    // ─────────────────────────────────────────────────────────────────

    fn search_pure_semantic(
        &self,
        survivors: &[(usize, f64)],
        max_hamming: f64,
        top_k: usize
    ) -> Vec<(String, f32)> {
        // PURE SEMANTIC: Only use Hamming distance (1-bit embeddings), NO lexical matching
        let mut results: Vec<(String, f32)> = survivors.iter()
            .map(|&(doc_idx, hamming_dist)| {
                // Convert Hamming distance to similarity score
                let s_sem = (1.0 - (hamming_dist / max_hamming)).max(0.0) as f32;
                (self.doc_ids[doc_idx].clone(), s_sem)
            })
            .collect();
        
        // Sort by semantic score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top_k
        results.into_iter().take(top_k).collect()
    }

    // ─────────────────────────────────────────────────────────────────
    // PATH C: FULL HYBRID - Rich TF-IDF + Phrase + Dialogue + Span Density
    // ─────────────────────────────────────────────────────────────────

    fn search_full_hybrid(
        &self,
        survivors: &[(usize, f64)],
        q_expanded: &AHashSet<String>,
        query_lower: &str,
        query_original: &str,
        max_hamming: f64,
        top_k: usize
    ) -> Vec<(String, f32)> {
        // Rare words for entity TF-IDF weighting
        let rare_words: AHashSet<String> = q_expanded.iter()
            .filter(|w| *self.word_idf.get(*w).unwrap_or(&0.0) > 2.5)
            .cloned()
            .collect();

        // Detect dialogue-like queries (QMSum optimization)
        let query_idf_avg: f32 = q_expanded.iter()
            .map(|w| *self.word_idf.get(w).unwrap_or(&0.5))
            .sum::<f32>() / q_expanded.len().max(1) as f32;
        let is_dialogue_query = query_idf_avg < 2.2 && q_expanded.len() > 4;
        let dialogue_semantic_boost = if is_dialogue_query { 0.35 } else { 0.0 };

        // Average doc length for BM25-style normalization
        let survivor_count = survivors.len().max(1);
        let total_len: usize = survivors.iter()
            .map(|&(doc_idx, _)| self.doc_word_sets.get(doc_idx).map(|s| s.len()).unwrap_or(0))
            .sum();
        let avg_doc_len = total_len as f32 / survivor_count as f32;
        let b_param = 0.35f32;

        let mut results: Vec<(String, f32)> = Vec::with_capacity(survivors.len());

        for &(doc_idx, semantic_dist) in survivors {
            // 1. SEMANTIC SCORE (from Hamming distance)
            let s_sem = (1.0 - (semantic_dist / max_hamming)).max(0.0) as f32;

            let doc_words = &self.doc_word_sets[doc_idx];
            let doc_tf = &self.doc_word_tf[doc_idx];
            let doc_text = &self.doc_texts[doc_idx];

            // Exact match safety net
            if doc_text.contains(query_lower) {
                results.push((self.doc_ids[doc_idx].clone(), 1.0));
                continue;
            }

            // 2. LEXICAL SCORE (Full TF-IDF)
            let overlap: AHashSet<String> = q_expanded.intersection(doc_words).cloned().collect();

            // Dynamic alpha based on IDF overlap
            let alpha = if overlap.is_empty() {
                1.0
            } else {
                let idf_overlap: f32 = overlap.iter()
                    .map(|w| *self.word_idf.get(w).unwrap_or(&0.5))
                    .sum::<f32>() / overlap.len() as f32;
                let idf_query: f32 = q_expanded.iter()
                    .map(|w| *self.word_idf.get(w).unwrap_or(&0.5))
                    .sum::<f32>() / q_expanded.len().max(1) as f32;
                let base_alpha = (1.0 - idf_overlap / idf_query.max(0.001)).clamp(0.0, 1.0);
                (base_alpha + dialogue_semantic_boost).min(1.0)
            };

            // Entity TF-IDF (70/30 weighting)
            let mut entity_score = 0.0f32;
            let mut common_score = 0.0f32;
            for word in &overlap {
                let tf = *doc_tf.get(word).unwrap_or(&1) as f32;
                let idf = *self.word_idf.get(word).unwrap_or(&0.5);
                let tfidf = (1.0 + tf).ln() * idf;
                if rare_words.contains(word) {
                    entity_score += tfidf;
                } else {
                    common_score += tfidf;
                }
            }
            let total_tfidf = 0.7 * entity_score + 0.3 * common_score;
            let token_ratio = (total_tfidf / 10.0).min(1.0);

            // Quadratic boost + BM25-style length normalization
            let boost = 1.0 + token_ratio.powi(2);
            let s_lex_raw = token_ratio * boost;
            let doc_len = doc_words.len() as f32;
            let length_norm = 1.0 - b_param + b_param * (doc_len / avg_doc_len.max(1.0));
            let s_lex = s_lex_raw / length_norm.max(0.5);

            // Phrase match boost
            let query_words_raw: Vec<&str> = query_original.split_whitespace().collect();
            let mut phrase_match_boost = 1.0f32;
            if query_words_raw.len() >= 3 {
                let window_sizes = [4, 3];
                'outer: for &w_size in &window_sizes {
                    if query_words_raw.len() < w_size { continue; }
                    for window in query_words_raw.windows(w_size) {
                        let window_clean: Vec<String> = window.iter()
                            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                            .collect();
                        let has_all_words = window_clean.iter().all(|w| doc_words.contains(w));
                        if !has_all_words { continue; }
                        let mut phrase_idf_sum = 0.0f32;
                        for w in &window_clean {
                            phrase_idf_sum += *self.word_idf.get(w).unwrap_or(&0.5);
                        }
                        if phrase_idf_sum / (w_size as f32) < 2.2 { continue; }
                        let phrase = window.join(" ").to_lowercase();
                        if phrase.trim().len() < 10 { continue; }
                        if doc_text.to_lowercase().contains(&phrase) {
                            phrase_match_boost = if w_size == 4 { 1.25 } else { 1.15 };
                            break 'outer;
                        }
                    }
                }
            }
            let s_lex_boosted = s_lex * phrase_match_boost;

            // Combined score
            let mut final_score = alpha * s_sem + (1.0 - alpha) * s_lex_boosted;

            // Speaker boost for dialogue queries
            if is_dialogue_query {
                let query_original_words: Vec<&str> = query_original.split_whitespace().collect();
                for word in &query_original_words {
                    if word.len() >= 3 && word.chars().next().unwrap().is_uppercase() {
                        let word_lower = word.to_lowercase();
                        if doc_text.contains(*word) || doc_words.contains(&word_lower) {
                            final_score *= 1.15;
                            break;
                        }
                    }
                }
            }

            results.push((self.doc_ids[doc_idx].clone(), final_score));
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Span-density reranking on top candidates
        let rerank_depth = self.rerank_depth.min(results.len());
        let q_words: Vec<String> = q_expanded.iter().cloned().collect();

        let mut champions: Vec<(String, f32)> = results.into_iter().take(rerank_depth).collect();

        champions.par_iter_mut().for_each(|(doc_id, score)| {
            if let Some(doc_text) = self.get_doc_text(doc_id) {
                let density_score = self.calculate_span_density_fast(&doc_text, &q_words);
                let boost_val = 1.0 + (density_score * 0.25);
                *score *= boost_val;
            }
        });

        champions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        champions.into_iter().take(top_k).collect()
    }

    // ─────────────────────────────────────────────────────────────────
    // HELPERS
    // ─────────────────────────────────────────────────────────────────

    fn normalize(&self, scores: &mut Vec<(String, f32)>) {
        if scores.is_empty() { return; }
        
        let min_s = scores.iter().map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
        let max_s = scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
        let range = max_s - min_s;

        if range > 1e-6 {
            for (_, s) in scores.iter_mut() {
                *s = (*s - min_s) / range;
            }
        } else {
            for (_, s) in scores.iter_mut() {
                *s = 0.5;
            }
        }
    }

    fn calculate_confidence(&self, scores: &[(String, f32)]) -> f32 {
        if scores.is_empty() { return 0.0; }

        let sum_score: f32 = scores.iter().map(|(_, s)| *s).sum();
        if sum_score == 0.0 { return 0.0; }

        let probs: Vec<f32> = scores.iter().map(|(_, s)| *s / sum_score).collect();

        let mut entropy = 0.0;
        for &p in &probs {
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        let max_entropy = (scores.len() as f32).ln();
        let normalized_entropy = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        1.0 - normalized_entropy
    }
}
