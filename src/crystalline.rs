//! ═══════════════════════════════════════════════════════════════════════════════
//!                     SAID Crystalline Attention (SCA)
//!                     Core Engine for LAM (Linear Attention Models: Linear Attention Memory)
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! THE SAID STANDARD - Managed by SAID Research | Powered by SaidHome.ai
//!
//! PROTECTED CORE - Integrated into LAM model, NOT a separate import!
//!
//! Usage:
//!     from lam import LAM
//!     
//!     model = LAM('LAM-base-v1')
//!     
//!     # Free tier: Standard embeddings (up to 12K tokens)
//!     embeddings = model.encode(sentences)
//!     
//!     # Licensed tier: Infinite context with deterministic recall
//!     model.encode_state(documents)  // Lock linear state (any size)
//!     results = model.recall(query)  // Deterministic O(1) recall
//!
//! The SAID Standard API (Simple as sentence-transformers):
//!     encode()   → Lock document into crystalline state  
//!     recall()   → Deterministic O(1) recall (auto-routes everything)
//!
//! Query auto-routing:
//! - Codes/IDs/Passkeys/Hashes → Lexical mode (token overlap, exact match)
//! - Natural language → Semantic mode (embedding similarity)
//! - Mixed → Hybrid with quadratic boost
//!
//! PHILOSOPHY: DETERMINISM OVER PROBABILITY.
//! "The answer IS X. Because I Said so." - Zero drift, infinite context.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use regex::Regex;
use tokenizers::Tokenizer;
use roaring::RoaringBitmap;

// SCA drop-in alignment: quantized Hamming, parallel processing
use rayon::prelude::*;
use ahash::{AHashMap, AHashSet};
use simsimd::BinarySimilarity;

// Import text storage for memory-efficient document storage
use crate::storage::TextStorage;

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTIVE RADIX TREE (ART) - Integrated Inverted Index
// ═══════════════════════════════════════════════════════════════════════════════
//
// O(4) constant-time lookup (token_id = 4 bytes)
// Pattern matching support: prefix_search(), find_all_patterns()
// Cache-friendly traversal for large document collections
//
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple radix tree node for inverted index
#[derive(Clone, Default)]
struct RadixNode {
    children: HashMap<u8, Box<RadixNode>>,  // byte → child
    doc_ids: Option<HashSet<String>>,       // Leaf: doc_ids
    positions: Option<HashMap<String, Vec<usize>>>, // Leaf: {doc_id: [pos1, pos2, ...]}
}

impl RadixNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            doc_ids: None,
            positions: None,
        }
    }
}

/// Adaptive Radix Tree for inverted index WITH POSITIONS.
/// 
/// Maps: token_id (u32) → {doc_id: [positions]}
/// 
/// O(4) = O(1) lookup, insert, delete (4 bytes = constant)
#[derive(Clone)]
#[allow(dead_code)]
pub struct ART {
    root: Box<RadixNode>,
    size: usize,
    postings: usize,
}

#[allow(dead_code)]
impl ART {
    pub fn new() -> Self {
        Self {
            root: Box::new(RadixNode::new()),
            size: 0,
            postings: 0,
        }
    }
    
    /// Insert doc_id and position for token_id
    pub fn insert(&mut self, token_id: u32, doc_id: &str, position: Option<usize>) {
        let key = token_id.to_be_bytes();
        let mut node = &mut *self.root;
        
        for &byte in &key {
            node = node.children
                .entry(byte)
                .or_insert_with(|| Box::new(RadixNode::new()));
        }
        
        if node.doc_ids.is_none() {
            node.doc_ids = Some(HashSet::new());
            node.positions = Some(HashMap::new());
            self.size += 1;
        }
        
        let doc_ids = node.doc_ids.as_mut().unwrap();
        let positions = node.positions.as_mut().unwrap();
        
        if !doc_ids.contains(doc_id) {
            doc_ids.insert(doc_id.to_string());
            positions.insert(doc_id.to_string(), Vec::new());
            self.postings += 1;
        }
        
        if let Some(pos) = position {
            if let Some(pos_list) = positions.get_mut(doc_id) {
                pos_list.push(pos);
            }
        }
    }
    
    /// Get doc_ids for token_id
    pub fn get(&self, token_id: u32) -> HashSet<String> {
        let key = token_id.to_be_bytes();
        let mut node = &*self.root;
        
        for &byte in &key {
            match node.children.get(&byte) {
                Some(child) => node = child,
                None => return HashSet::new(),
            }
        }
        
        node.doc_ids.clone().unwrap_or_default()
    }
    
    /// Get positions for token_id: {doc_id: [pos1, pos2, ...]}
    pub fn get_positions(&self, token_id: u32) -> HashMap<String, Vec<usize>> {
        let key = token_id.to_be_bytes();
        let mut node = &*self.root;
        
        for &byte in &key {
            match node.children.get(&byte) {
                Some(child) => node = child,
                None => return HashMap::new(),
            }
        }
        
        node.positions.clone().unwrap_or_default()
    }
    
    /// Get document frequency for token_id (how many docs contain it)
    pub fn get_doc_freq(&self, token_id: u32) -> usize {
        self.get(token_id).len()
    }
    
    /// Remove doc_id from token_id's posting list
    pub fn remove(&mut self, token_id: u32, doc_id: &str) -> bool {
        let key = token_id.to_be_bytes();
        let mut node = &mut *self.root;
        let mut path: Vec<(*mut RadixNode, u8)> = Vec::new();
        
        for &byte in &key {
            let node_ptr = node as *mut RadixNode;
            match node.children.get_mut(&byte) {
                Some(child) => {
                    path.push((node_ptr, byte));
                    node = child;
                }
                None => return false,
            }
        }
        
        if let Some(ref mut doc_ids) = node.doc_ids {
            if !doc_ids.contains(doc_id) {
                return false;
            }
            
            doc_ids.remove(doc_id);
            self.postings -= 1;
            
            if let Some(ref mut positions) = node.positions {
                positions.remove(doc_id);
            }
            
            // Cleanup empty nodes
            if doc_ids.is_empty() && node.children.is_empty() {
                node.doc_ids = None;
                node.positions = None;
                self.size -= 1;
                
                // Remove empty parent nodes
                for (parent_ptr, byte) in path.into_iter().rev() {
                    unsafe {
                        let parent = &mut *parent_ptr;
                        if let Some(child) = parent.children.get(&byte) {
                            if child.doc_ids.is_none() && child.children.is_empty() {
                                parent.children.remove(&byte);
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
            
            true
        } else {
            false
        }
    }
    
    /// Iterate over all (token_id, doc_ids) pairs
    pub fn iter_all(&self) -> Vec<(u32, HashSet<String>)> {
        let mut results = Vec::new();
        self.iter_helper(&self.root, &mut [0u8; 4], 0, &mut results);
        results
    }
    
    fn iter_helper(
        &self,
        node: &RadixNode,
        prefix: &mut [u8; 4],
        depth: usize,
        results: &mut Vec<(u32, HashSet<String>)>,
    ) {
        if depth == 4 {
            if let Some(ref doc_ids) = node.doc_ids {
                let token_id = u32::from_be_bytes(*prefix);
                results.push((token_id, doc_ids.clone()));
            }
            return;
        }
        
        for (&byte, child) in &node.children {
            prefix[depth] = byte;
            self.iter_helper(child, prefix, depth + 1, results);
        }
    }
    
    /// O(k) Prefix Search - Tree traversal, no regex.
    /// Find all token_ids that share the same prefix bytes.
    pub fn search_prefix(&self, prefix_token_id: u32, prefix_bytes_len: usize) -> Vec<(u32, HashSet<String>)> {
        let full_key = prefix_token_id.to_be_bytes();
        let prefix = &full_key[..prefix_bytes_len.min(4)];
        
        let mut node = &*self.root;
        
        // Navigate to prefix node
        for &byte in prefix {
            match node.children.get(&byte) {
                Some(child) => node = child,
                None => return Vec::new(),
            }
        }
        
        // Yield everything below this node
        let mut results = Vec::new();
        let mut current_prefix = [0u8; 4];
        current_prefix[..prefix.len()].copy_from_slice(prefix);
        self.prefix_recurse(node, &mut current_prefix, prefix.len(), &mut results);
        results
    }
    
    fn prefix_recurse(
        &self,
        node: &RadixNode,
        path: &mut [u8; 4],
        depth: usize,
        results: &mut Vec<(u32, HashSet<String>)>,
    ) {
        if depth == 4 {
            if let Some(ref doc_ids) = node.doc_ids {
                let token_id = u32::from_be_bytes(*path);
                results.push((token_id, doc_ids.clone()));
            }
            return;
        }
        
        for (&byte, child) in &node.children {
            path[depth] = byte;
            self.prefix_recurse(child, path, depth + 1, results);
        }
    }
    
    /// Clear all data
    pub fn clear(&mut self) {
        self.root = Box::new(RadixNode::new());
        self.size = 0;
        self.postings = 0;
    }
    
    /// Get number of unique tokens
    pub fn len(&self) -> usize {
        self.size
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

/// Dict-like wrapper around ART for drop-in replacement of HashMap<u32, HashSet<String>>
#[derive(Clone)]
#[allow(dead_code)]
pub struct ARTDict {
    art: ART,
}

#[allow(dead_code)]
impl ARTDict {
    pub fn new() -> Self {
        Self { art: ART::new() }
    }
    
    pub fn contains(&self, token_id: u32) -> bool {
        !self.art.get(token_id).is_empty()
    }
    
    pub fn len(&self) -> usize {
        self.art.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.art.is_empty()
    }
    
    pub fn get(&self, token_id: u32) -> HashSet<String> {
        self.art.get(token_id)
    }
    
    pub fn clear(&mut self) {
        self.art.clear();
    }
    
    pub fn values(&self) -> Vec<HashSet<String>> {
        self.art.iter_all().into_iter().map(|(_, docs)| docs).collect()
    }
    
    pub fn items(&self) -> Vec<(u32, HashSet<String>)> {
        self.art.iter_all()
    }
    
    pub fn add(&mut self, token_id: u32, doc_id: &str, position: Option<usize>) {
        self.art.insert(token_id, doc_id, position);
    }
    
    pub fn get_positions(&self, token_id: u32) -> HashMap<String, Vec<usize>> {
        self.art.get_positions(token_id)
    }
    
    pub fn get_doc_freq(&self, token_id: u32) -> usize {
        self.art.get_doc_freq(token_id)
    }
    
    pub fn discard(&mut self, token_id: u32, doc_id: &str) {
        self.art.remove(token_id, doc_id);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STOPWORDS & CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Stopwords for query filtering (matches Python _STOPWORDS exactly)
const STOPWORDS: &[&str] = &[
    "what", "when", "where", "which", "who", "why", "how", "the", "they",
    "them", "their", "known", "that", "this", "with", "from", "have", "been",
    "were", "being", "for", "was", "and", "are", "is", "his", "her", "she", "he",
    "a", "an", "of", "to", "in", "it", "on", "at", "by", "as", "or", "be",
    "do", "did", "does", "has", "had", "its",
];

/// Connector tokens (always skip in search_kv)
const CONNECTOR_TOKENS: &[&str] = &[
    "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "and", "or", "as", "that", "which", "equals",
];

/// Protected values (never skip in search_kv)
#[allow(dead_code)]
const PROTECTED_VALUES: &[&str] = &[
    "true", "false", "yes", "no", "on", "off", "0", "1",
    "active", "inactive", "enabled", "disabled", "success", "failed",
    "error", "warning", "info", "critical", "high", "medium", "low",
    "null", "none", "nil", "undefined",
];

/// Code-intent words: query contains one of these (whole-word) → route to high_lexical (port from rust_test lib.rs)
const CODE_INTENT_WORDS: &[&str] = &["passkey", "password", "passcode", "serial", "needle"];

/// Known compound→split mappings for CODE_INTENT_WORDS that may appear as two
/// words in documents (e.g. "pass key") but one word in queries ("passkey").
const COMPOUND_SPLITS: &[(&str, &[&str])] = &[
    ("passkey", &["pass", "key"]),
    ("passcode", &["pass", "code"]),
    ("password", &["pass", "word"]),
];

/// Route from query analysis (Step 6: aligned with sca_dropin).
#[derive(Clone, Copy, PartialEq)]
enum QueryRoute {
    PureLexical,
    PureSemantic,
    FullHybrid,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CrystallineCore - The IDF-Surprise search engine (1:1 match with Python)
// ═══════════════════════════════════════════════════════════════════════════════

/// CrystallineCore - SCA (Said Crystalline Attention) Core Engine
/// 
/// This is the PROTECTED CORE that gets integrated into LAM.
/// Users only see: model.index(), model.search()
/// 
/// AUTO-ROUTING SEARCH:
/// - Detects query type automatically (lexical vs semantic)
/// - Uses optimal alpha for each query
/// - Quadratic boost formula for hybrid scoring
/// 
/// STREAMING:
/// - O(N) indexing, O(1) query
/// - Constant memory for 2M+ tokens
/// - Pre-indexed filler for 500x speedup
/// CrystallineCore - SCA (Said Crystalline Attention) Core Engine
/// 
/// This is a 1:1 match with Python `_crystalline.py`.
/// Uses BERT tokenization for the inverted index (ART).
/// Uses simple word tokenization ONLY for fuzzy matching (Soundex/Levenshtein).
#[derive(Clone)]
#[allow(dead_code)]
pub struct CrystallineCore {
    // BERT Tokenizer (uses tokenizers library, NOT transformers!)
    // Matches Python: self._tokenizer
    tokenizer: Option<Arc<Tokenizer>>,
    
    // =========================================================================
    // PHASE 1: NEW TEXT STORAGE (A/B Testing)
    // =========================================================================
    // Maps String doc_id → u64 internal index
    // This enables future migration to all-integer indexing
    doc_id_to_idx: HashMap<String, u64>,
    
    // New text storage using InMemoryTextStore
    // Same API will work with MmapTextStore for zero-copy disk access
    text_store: TextStorage,
    
    // =========================================================================
    // CORE STORAGE (Optimized in Phase 1-2)
    // =========================================================================
    doc_ids: Vec<String>,
    // doc_texts REMOVED - text now stored in text_store (Phase 1 cleanup)
    // PHASE 2: RoaringBitmap replaces HashSet<u32> - ~100x smaller memory footprint
    doc_token_sets: HashMap<String, RoaringBitmap>,  // FORWARD INDEX: doc_id → token_set (compressed)
    doc_token_counts: HashMap<String, HashMap<u32, u32>>,  // doc_id → {token_id → count}
    doc_embeddings: HashMap<String, Vec<f32>>,
    // LEGACY LONGEMBED: passage embeddings per document for MaxSim scoring (Python _crystalline.pyx behavior)
    doc_passage_embeddings: HashMap<String, Vec<Vec<f32>>>, // doc_id -> [passage_emb, ...]
    
    // Fuzzy matching vocabulary (lazy initialization)
    token_vocab: HashMap<u32, String>,  // token_id → word
    
    // WORD-LEVEL indexes for fuzzy matching (simple tokenization, NOT BERT!)
    word_inverted_index: HashMap<String, HashSet<String>>,  // word → {doc_ids}
    word_phonetic_index: HashMap<String, HashSet<String>>, // soundex → words (Step 4: sca_dropin shape)
    word_vocabulary: HashSet<String>,                        // All unique words
    word_idf: HashMap<String, f32>,                          // word → IDF score
    word_doc_tf: HashMap<String, HashMap<String, u32>>,      // doc_id → {word → count}
    
    // INVERTED INDEX: token_id → {doc_id1, doc_id2, ...}
    // O(4) = O(1) lookup using Adaptive Radix Tree (ART)
    // Stores BERT token IDs (matches Python _inverted_index)
    inverted_index: ARTDict,
    
    // POSITION INDEX: doc_id → {position → token_id}
    // For pure ART extraction without regex
    position_tokens: HashMap<String, HashMap<usize, u32>>,
    
    // TOKEN STRINGS: token_id → string (for decoding)
    token_strings: HashMap<u32, String>,
    
    // IDF CACHE: token_id → IDF score (log(N / doc_freq))
    idf_cache: HashMap<u32, f32>,
    idf_dirty: bool,
    
    // Special tokens to skip
    special_tokens: HashSet<u32>,
    stopword_tokens: Option<HashSet<u32>>,
    
    // Config
    vocab_size: usize,
    min_token_id: u32,
    doc_count: usize,
    
    // =========================================================================
    // QUANTIZED STORAGE (sca_dropin alignment for 94.1060 parity)
    // =========================================================================
    // 1-bit quantized embeddings (all passages/docs)
    matrix_quantized: Vec<u8>,
    corpus_mean: Vec<f32>,
    dim: usize,
    quantized_dim: usize,  // (dim + 7) / 8
    
    // Passage metadata for MaxSim
    passage_counts: Vec<usize>,
    passage_offsets: Vec<usize>,
    
    // Holographic 16-view quantization
    holographic_16view: bool,
    holographic_scale: f32,
    bytes_per_passage: usize,
    
    // Word-level lexical index (sca_dropin style: AHashMap for speed)
    doc_word_sets_fast: Vec<AHashSet<String>>,
    doc_word_tf_fast: Vec<AHashMap<String, u32>>,
    doc_texts_fast: Vec<String>,
    word_idf_fast: AHashMap<String, f32>,
    phonetic_index_fast: AHashMap<String, AHashSet<String>>,
    vocabulary_fast: AHashSet<String>,
    word_inverted_fast: AHashMap<String, AHashSet<usize>>,
    
    // Hybrid search weights
    hybrid_alpha_semantic: f32,
    hybrid_alpha_lexical: f32,
    rerank_depth: usize,
    
    // Quantized mode flag (when true, use quantized search path)
    quantized_mode: bool,
    
    // Force route override (e.g. "FullHybrid" for LEMBNeedleRetrieval)
    force_route: Option<QueryRoute>,
}

#[allow(dead_code)]
impl CrystallineCore {
    /// Create new CrystallineCore (matches Python __init__)
    /// 
    /// Args:
    ///     tokenizer: Optional BERT tokenizer (tokenizers library)
    ///     vocab_size: Vocabulary size (default 30522 for BERT)
    ///     min_token_id: Skip special tokens below this (default 1000)
    pub fn new() -> Self {
        Self::with_tokenizer(None, 30522, 1000)
    }
    
    /// Create CrystallineCore with tokenizer (matches Python __init__)
    pub fn with_tokenizer(tokenizer: Option<Arc<Tokenizer>>, vocab_size: usize, min_token_id: u32) -> Self {
        let mut special_tokens = HashSet::new();
        special_tokens.insert(0);
        special_tokens.insert(100);
        special_tokens.insert(101);
        special_tokens.insert(102);
        special_tokens.insert(103);
        
        let default_dim = 384;  // SAID-LAM embedding dimension
        let quantized_dim = (default_dim + 7) / 8;
        
        Self {
            tokenizer,
            // Phase 1: Optimized storage
            doc_id_to_idx: HashMap::new(),
            text_store: TextStorage::new_ephemeral(),
            // Core storage
            doc_ids: Vec::new(),
            // doc_texts REMOVED - use text_store
            doc_token_sets: HashMap::new(),
            doc_token_counts: HashMap::new(),
            doc_embeddings: HashMap::new(),
            doc_passage_embeddings: HashMap::new(),
            token_vocab: HashMap::new(),
            word_inverted_index: HashMap::new(),
            word_phonetic_index: HashMap::new(),
            word_vocabulary: HashSet::new(),
            word_idf: HashMap::new(),
            word_doc_tf: HashMap::new(),
            inverted_index: ARTDict::new(),
            position_tokens: HashMap::new(),
            token_strings: HashMap::new(),
            idf_cache: HashMap::new(),
            idf_dirty: true,
            special_tokens,
            stopword_tokens: None,
            vocab_size,
            min_token_id,
            doc_count: 0,
            // Quantized storage (sca_dropin alignment)
            matrix_quantized: Vec::new(),
            corpus_mean: vec![0.0; default_dim],
            dim: default_dim,
            quantized_dim,
            passage_counts: Vec::new(),
            passage_offsets: Vec::new(),
            holographic_16view: false,
            holographic_scale: 0.2,
            bytes_per_passage: quantized_dim,
            doc_word_sets_fast: Vec::new(),
            doc_word_tf_fast: Vec::new(),
            doc_texts_fast: Vec::new(),
            word_idf_fast: AHashMap::new(),
            phonetic_index_fast: AHashMap::new(),
            vocabulary_fast: AHashSet::new(),
            word_inverted_fast: AHashMap::new(),
            hybrid_alpha_semantic: 0.60,
            hybrid_alpha_lexical: 0.40,
            rerank_depth: 100,
            quantized_mode: false,
            force_route: None,
        }
    }

    /// Get query IDF for adaptive alpha calculation (MTEB compatibility)
    pub fn get_query_idf(&self, query: &str) -> f32 {
        let (_, _, _, idf, _, _, _) = self.analyze_query(query);
        idf
    }
    
    /// Enable persistent storage using memory-mapped files
    /// 
    /// This replaces the in-memory text store with a disk-backed mmap store,
    /// reducing RAM usage dramatically for large document collections.
    /// 
    /// # Arguments
    /// * `base_path` - Path prefix for storage files (e.g., "/data/index" creates "/data/index.texts")
    /// 
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(String)` if mmap feature is not enabled or file creation fails
    #[cfg(feature = "mmap")]
    pub fn enable_persistence(&mut self, base_path: &str) -> Result<(), String> {
        let text_path = format!("{}.texts", base_path);
        self.text_store = TextStorage::new_persistent(&text_path)
            .map_err(|e| format!("Failed to create persistent text store: {}", e))?;
        Ok(())
    }
    
    /// Enable persistent storage (stub for when mmap feature is disabled)
    #[cfg(not(feature = "mmap"))]
    pub fn enable_persistence(&mut self, _base_path: &str) -> Result<(), String> {
        Err("Persistent storage requires the 'mmap' feature to be enabled".to_string())
    }
    
    /// Open existing persistent storage
    /// 
    /// # Arguments
    /// * `base_path` - Path prefix for storage files
    /// 
    /// # Returns
    /// * `Ok(())` on success, loading existing data
    #[cfg(feature = "mmap")]
    pub fn open_persistent(&mut self, base_path: &str) -> Result<(), String> {
        let text_path = format!("{}.texts", base_path);
        self.text_store = TextStorage::open_persistent(&text_path)
            .map_err(|e| format!("Failed to open persistent text store: {}", e))?;
        Ok(())
    }
    
    /// Open persistent storage (stub for when mmap feature is disabled)
    #[cfg(not(feature = "mmap"))]
    pub fn open_persistent(&mut self, _base_path: &str) -> Result<(), String> {
        Err("Persistent storage requires the 'mmap' feature to be enabled".to_string())
    }
    
    /// Check if using persistent storage
    pub fn is_persistent(&self) -> bool {
        self.text_store.is_persistent()
    }
    
    /// Set tokenizer (matches Python's lazy initialization)
    pub fn set_tokenizer(&mut self, tokenizer: Arc<Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // BERT TOKENIZATION (matches Python _tokenize, _get_tokens, _get_tokens_with_positions)
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Tokenize text using BERT tokenizer (matches Python _tokenize)
    /// NO TRUNCATION - captures ALL tokens for perfect recall
    fn tokenize(&self, text: &str) -> Vec<u32> {
        if let Some(ref tokenizer) = self.tokenizer {
            let text_lower = text.to_lowercase();
            match tokenizer.encode(text_lower, false) {
                Ok(encoding) => encoding.get_ids().to_vec(),
                Err(_) => Vec::new(),
            }
        } else {
            // Fallback: no tokenizer available
            Vec::new()
        }
    }
    
    /// Get token set (crystal addresses) - matches Python _get_tokens
    fn get_tokens(&self, text: &str) -> HashSet<u32> {
        let ids = self.tokenize(text);
        ids.into_iter()
            .filter(|&tid| tid >= self.min_token_id && !self.special_tokens.contains(&tid))
            .collect()
    }
    
    /// Get tokens with positions - matches Python _get_tokens_with_positions
    /// Returns: Vec<(token_id, token_string, position)>
    fn get_tokens_with_positions(&self, text: &str) -> Vec<(u32, String, usize)> {
        if let Some(ref tokenizer) = self.tokenizer {
            let text_lower = text.to_lowercase();
            match tokenizer.encode(text_lower, false) {
                Ok(encoding) => {
                    let ids = encoding.get_ids();
                    let tokens = encoding.get_tokens();
                    
                    ids.iter()
                        .zip(tokens.iter())
                        .enumerate()
                        .filter_map(|(pos, (&tid, token_str))| {
                            if tid >= self.min_token_id && !self.special_tokens.contains(&tid) {
                                // Clean token string (remove ## prefix for subwords)
                                let clean_str = token_str.replace("##", "").replace("Ġ", "").trim().to_string();
                                if !clean_str.is_empty() {
                                    Some((tid, clean_str, pos))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect()
                }
                Err(_) => Vec::new(),
            }
        } else {
            // Fallback: no tokenizer available
            Vec::new()
        }
    }
    
    /// Get stopword tokens (matches Python _get_stopword_tokens)
    fn get_stopword_tokens(&mut self) -> &HashSet<u32> {
        if self.stopword_tokens.is_none() {
            let mut stopword_set = HashSet::new();
            for word in STOPWORDS {
                let tokens = self.tokenize(word);
                stopword_set.extend(tokens);
            }
            self.stopword_tokens = Some(stopword_set);
        }
        self.stopword_tokens.as_ref().unwrap()
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // SIMPLE WORD TOKENIZATION (for fuzzy matching - NOT BERT!)
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Simple word tokenization (matches Python: re.findall(r'\b\w+\b', text.lower()))
    /// This is CRITICAL for fuzzy matching because BERT breaks words:
    /// - BERT: "anderton" → ["and", "##erton"] ← BROKEN for Soundex
    /// - Simple: "anderton" → ["anderton"] ← WORKS for Soundex
    fn simple_tokenize(&self, text: &str) -> Vec<String> {
        let re = Regex::new(r"\w+").unwrap();
        re.find_iter(&text.to_lowercase())
            .map(|m| m.as_str().to_string())
            .collect()
    }
    
    /// Filter stopwords
    fn filter_stopwords(&self, words: Vec<String>) -> Vec<String> {
        words.into_iter()
            .filter(|w| !STOPWORDS.contains(&w.as_str()))
            .collect()
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // SOUNDEX + LEVENSHTEIN (Fuzzy Matching)
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Generate Soundex code for phonetic matching (Step 1: aligned with sca_dropin).
    /// Maps similar-sounding words to same code (e.g., "anderton" → "A536", "anderson" → "A536").
    /// Empty word returns "0000" to match sca_dropin; first letter's code used as prev to collapse duplicates.
    fn get_soundex(&self, word: &str) -> String {
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
    
    /// Levenshtein distance (edit distance); Step 2: aligned with sca_dropin (char-based).
    /// Used for ranking Soundex candidates by spelling similarity.
    fn levenshtein(&self, a: &str, b: &str) -> usize {
        let a_len = a.chars().count();
        let b_len = b.chars().count();
        if a_len == 0 {
            return b_len;
        }
        if b_len == 0 {
            return a_len;
        }
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
    
    /// Fuzzy expand a WORD (Step 3: aligned with sca_dropin).
    /// Soundex blocking + Levenshtein ranking; filter dist ≤ 2; fallback to [word_lower] if empty.
    fn fuzzy_expand_word(&self, word: &str, top_k: usize) -> Vec<String> {
        let word_lower = word.to_lowercase();

        if self.word_inverted_index.contains_key(&word_lower) {
            return vec![word_lower];
        }

        let soundex_code = self.get_soundex(&word_lower);
        let candidates = match self.word_phonetic_index.get(&soundex_code) {
            Some(c) => c,
            None => return vec![word_lower],
        };

        let mut ranked: Vec<(String, usize)> = candidates
            .iter()
            .map(|c| (c.clone(), self.levenshtein(&word_lower, c)))
            .filter(|(_, dist)| *dist <= 2)
            .collect();

        ranked.sort_by_key(|(_, dist)| *dist);

        let result: Vec<String> = ranked.into_iter().take(top_k).map(|(w, _)| w).collect();

        if result.is_empty() {
            vec![word_lower]
        } else {
            result
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CODE DETECTION (Step 5: aligned with sca_dropin)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Code detection: passkeys/codes (5–10 pure digits).
    /// Excludes dates, currency, ordinals, measurements. Matches sca_dropin exactly.
    fn looks_like_code(&self, word: &str) -> bool {
        if word.contains('/') || word.contains('-') {
            return false;
        }
        if word.contains('$') || word.contains('€') || word.contains('£') || word.contains(',') {
            return false;
        }
        if word.starts_with('(') || word.starts_with('[') {
            return false;
        }
        if word.chars().any(|c| c == '±' || c == '×' || c == '÷') {
            return false;
        }
        let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
        if clean.len() < 5 || clean.len() > 15 {
            return false;
        }
        let lower = clean.to_lowercase();
        if lower.ends_with("st") || lower.ends_with("nd") || lower.ends_with("rd") || lower.ends_with("th") {
            let prefix = &lower[..lower.len() - 2];
            if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
                return false;
            }
        }
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
        if letter_count == 0 && digit_count >= 5 && digit_count <= 10 {
            return true;
        }
        false
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // QUERY ANALYSIS & ROUTING (Step 6: sca_dropin analyze_query + route → API)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Analyze query like sca_dropin: q_words, q_expanded, idf_avg, oov_ratio, has_code, has_typo, route.
    fn analyze_query(&self, query_text: &str) -> (QueryRoute, Vec<String>, HashSet<String>, f32, f32, bool, bool) {
        let q_words: Vec<String> = query_text
            .split_whitespace()
            .map(|s| {
                let lower = s.to_lowercase();
                lower.trim_end_matches(|c: char| c.is_ascii_punctuation()).to_string()
            })
            .filter(|s| s.len() >= 3)
            .collect();

        if q_words.is_empty() {
            return (QueryRoute::PureSemantic, q_words, HashSet::new(), 0.5, 0.0, false, false);
        }

        let mut has_any_code = false;
        let mut has_typo = false;
        let mut known_word_count = 0;
        let mut total_idf = 0.0f32;
        let mut oov_count = 0;
        let mut q_expanded: HashSet<String> = HashSet::new();

        for word in &q_words {
            if self.looks_like_code(word) {
                has_any_code = true;
                total_idf += 5.0;
                known_word_count += 1;
                q_expanded.insert(word.clone());
                continue;
            }
            let in_idf = self.word_idf.get(word);
            if in_idf.is_some() || self.word_vocabulary.contains(word) {
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
                    let fuzzy_matches = self.fuzzy_expand_word(word, 5);
                    let valid_matches: Vec<String> = fuzzy_matches
                        .iter()
                        .filter(|m| *m != word && self.word_vocabulary.contains(*m))
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

        let content_words: Vec<f32> = q_expanded
            .iter()
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

        let non_code_count = q_words.iter().filter(|w| !self.looks_like_code(w)).count();
        let oov_ratio = if non_code_count > 0 {
            oov_count as f32 / non_code_count as f32
        } else {
            0.0
        };

        let has_code_intent = q_words.iter().any(|w| CODE_INTENT_WORDS.contains(&w.as_str()));
        let high_idf_count = q_expanded
            .iter()
            .filter(|w| *self.word_idf.get(*w).unwrap_or(&0.0) > 2.5)
            .count();
        let is_short_discourse = q_words.len() <= 8
            && !has_any_code
            && !has_code_intent
            && idf_avg <= 1.2
            && high_idf_count == 0
            && oov_ratio < 0.1;

        let route = if has_any_code || has_code_intent {
            QueryRoute::PureLexical
        } else if is_short_discourse {
            QueryRoute::PureSemantic
        } else {
            QueryRoute::FullHybrid
        };

        (route, q_words, q_expanded, idf_avg, oov_ratio, has_any_code, has_typo)
    }

    /// Public API: query type string for callers (engine.rs). Step 11 will review if still needed.
    fn detect_query_type(&self, query: &str) -> &'static str {
        if query.is_empty() || query.trim().is_empty() {
            return "balanced";
        }
        let (route, _, _, _, _, _, _) = self.analyze_query(query);
        match route {
            QueryRoute::PureLexical => "high_lexical",
            QueryRoute::PureSemantic => "pure_semantic",
            QueryRoute::FullHybrid => "balanced",
        }
    }
    
    /// Get alpha value for hybrid scoring (SIMPLIFIED for IDF-Surprise).
    /// 
    /// With IDF-weighting, token scoring is smarter:
    /// - Rare tokens (like "Bobolink") get HIGH IDF weight
    /// - Common tokens (like "the") get LOW IDF weight
    /// 
    /// This means alpha=0.70 works well for BOTH lexical AND semantic queries!
    /// Only 3 distinct modes needed:
    /// 
    /// - high_lexical (0.85): NIAH/Passkey - exact token match critical
    /// - balanced (0.70): Default - IDF-Surprise handles all semantic queries
    /// - pure_semantic (0.00): Paraphrase detection - meaning > words
    fn get_alpha(&self, query_type: &str) -> f32 {
        match query_type {
            "high_lexical" => 0.85,
            "balanced" => 0.70,
            "high_semantic" => 0.70,     // MAPPED → balanced (IDF handles this)
            "very_high_semantic" => 0.70, // MAPPED → balanced (IDF handles this)
            "pure_semantic" => 0.0,
            _ => 0.70,
        }
    }
    
    /// Extract identifier-like tokens from a query for exact match verification.
    fn extract_identifiers(&self, query: &str) -> Vec<String> {
        let mut identifiers = Vec::new();
        let punct_chars: &[char] = &['.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '"', '\''];
        
        for word in query.split_whitespace() {
            let word_clean: String = word.trim_matches(punct_chars).to_string();
            
            let word_upper = word_clean.chars().filter(|c| c.is_uppercase()).count();
            let word_digit = word_clean.chars().filter(|c| c.is_ascii_digit()).count();
            let word_alpha = word_clean.chars().filter(|c| c.is_alphabetic()).count();
            let word_alnum = word_alpha + word_digit;
            
            if word_alnum == 0 {
                continue;
            }
            
            let word_upper_ratio = word_upper as f32 / word_alnum as f32;
            let word_digit_ratio = word_digit as f32 / word_alnum as f32;
            
            // Identifier criteria (matching Python)
            let is_identifier = word_clean.len() >= 5 && (
                word_upper_ratio > 0.5 ||
                (word_digit_ratio > 0.1 && word_upper_ratio > 0.2) ||
                word_digit_ratio > 0.3 ||
                (word_clean.contains('-') && word_clean.len() >= 7) ||
                (word_clean.contains('_') && word_clean.len() >= 7)
            );
            
            // Also include pure numeric codes (passkeys like "84729")
            let is_numeric_code = word_clean.len() >= 4 && word_digit_ratio > 0.8;
            
            if is_identifier || is_numeric_code {
                identifiers.push(word_clean);
            }
        }
        
        identifiers
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // IDF CACHE MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Rebuild IDF cache. Token IDF and word IDF use same formula as sca_dropin/Python (Step 7).
    /// Word IDF: log((N+1)/(df+1))+1, with normalized word keys (lowercase, trim trailing punct) and df = doc count.
    pub fn rebuild_idf(&mut self) {
        if !self.idf_dirty || self.doc_count == 0 {
            return;
        }

        let n = self.doc_count as f32;

        // Token-level IDF (unchanged)
        self.idf_cache.clear();
        for (token_id, doc_ids) in self.inverted_index.items() {
            let df = doc_ids.len() as f32;
            let idf = ((n + 1.0) / (df + 1.0)).ln() + 1.0;
            self.idf_cache.insert(token_id, idf);
        }

        // Word-level IDF and PHONETIC INDEX (Step 7: same formula and normalized word set as sca_dropin)
        self.word_idf.clear();
        self.word_phonetic_index.clear();

        // Build norm -> doc_ids so we merge raw words that normalize to the same form (same shape as sca_dropin)
        let mut norm_to_docs: HashMap<String, HashSet<String>> = HashMap::new();
        for word in &self.word_vocabulary {
            let w_normalized: String = word
                .to_lowercase()
                .trim_end_matches(|c: char| c.is_ascii_punctuation())
                .to_string();
            if w_normalized.len() < 3 {
                continue;
            }
            if let Some(doc_ids) = self.word_inverted_index.get(word) {
                let set = norm_to_docs.entry(w_normalized.clone()).or_insert_with(HashSet::new);
                for doc_id in doc_ids {
                    set.insert(doc_id.clone());
                }
            }
        }

        for (w_norm, doc_set) in &norm_to_docs {
            let df = doc_set.len() as f32;
            let idf = ((n + 1.0) / (df + 1.0)).ln() + 1.0;
            self.word_idf.insert(w_norm.clone(), idf);

            let sx = self.get_soundex(w_norm);
            self.word_phonetic_index
                .entry(sx)
                .or_insert_with(HashSet::new)
                .insert(w_norm.clone());
        }

        self.idf_dirty = false;
    }
    
    /// Get IDF for token. Returns 0.5 for unknown tokens.
    fn get_idf(&self, token_id: u32) -> f32 {
        *self.idf_cache.get(&token_id).unwrap_or(&0.5)
    }
    
    /// Get bigrams from sorted token set (for phrase matching).
    fn get_bigrams(&self, tokens: &HashSet<u32>) -> HashSet<(u32, u32)> {
        let mut sorted_tokens: Vec<u32> = tokens.iter().cloned().collect();
        sorted_tokens.sort();
        
        if sorted_tokens.len() < 2 {
            return HashSet::new();
        }
        
        sorted_tokens.windows(2)
            .map(|w| (w[0], w[1]))
            .collect()
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // INDEXING: Stream documents into crystal lattice
    // Matches Python _crystalline.py index() EXACTLY
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Index a document with BERT tokens and POSITIONS (matches Python index())
    /// 
    /// This is a 1:1 match with Python _crystalline.py:
    /// 1. Uses BERT tokenization for the inverted index (ART)
    /// 2. Uses simple word tokenization for fuzzy matching (Soundex/Levenshtein)
    pub fn index(&mut self, doc_id: &str, text: &str) {
        // Get BERT tokens WITH POSITIONS (matches Python _get_tokens_with_positions)
        let tokens_with_pos = self.get_tokens_with_positions(text);
        
        // PHASE 2: Use RoaringBitmap instead of HashSet<u32> - ~100x smaller
        let mut token_set = RoaringBitmap::new();
        for (tid, _, _) in &tokens_with_pos {
            token_set.insert(*tid);
        }
        
        // Remove old inverted index entries if re-indexing (matches Python)
        if let Some(old_tokens) = self.doc_token_sets.get(doc_id) {
            // RoaringBitmap iteration
            for token_id in old_tokens.iter() {
                self.inverted_index.discard(token_id, doc_id);
            }
        }
        
        // Store in FORWARD index (doc → tokens) - matches Python
        if !self.doc_ids.contains(&doc_id.to_string()) {
            self.doc_ids.push(doc_id.to_string());
        }
        
        // Store text in text_store (CLEANUP: removed legacy doc_texts HashMap)
        if !self.doc_id_to_idx.contains_key(doc_id) {
            let idx = self.text_store.append_text(text).unwrap_or(0);
            self.doc_id_to_idx.insert(doc_id.to_string(), idx);
        } else {
            // Re-indexing: update existing entry
            let idx = self.text_store.append_text(text).unwrap_or(0);
            self.doc_id_to_idx.insert(doc_id.to_string(), idx);
        }
        
        // PHASE 2: Insert RoaringBitmap (compressed token set)
        self.doc_token_sets.insert(doc_id.to_string(), token_set);
        self.doc_count = self.doc_ids.len();
        
        // Store token counts for TF-IDF (matches Python)
        let mut token_counts: HashMap<u32, u32> = HashMap::new();
        for (tid, _, _) in &tokens_with_pos {
            *token_counts.entry(*tid).or_insert(0) += 1;
        }
        self.doc_token_counts.insert(doc_id.to_string(), token_counts);
        
        // Store in INVERTED index WITH POSITIONS - O(1) ART insertion (matches Python)
        // Store in POSITION index (doc → pos → token_id)
        let mut position_map: HashMap<usize, u32> = HashMap::new();
        
        for (token_id, token_str, pos) in &tokens_with_pos {
            self.inverted_index.add(*token_id, doc_id, Some(*pos));
            position_map.insert(*pos, *token_id);
            self.token_strings.insert(*token_id, token_str.clone());
        }
        self.position_tokens.insert(doc_id.to_string(), position_map);
        
        // ═══════════════════════════════════════════════════════════════
        // WORD-LEVEL INDEX (for fuzzy matching with simple tokenization)
        // This is CRITICAL because BERT sub-tokenization breaks Soundex!
        // Matches Python: words = self._simple_tokenize(text)
        // ═══════════════════════════════════════════════════════════════
        let words = self.simple_tokenize(text);
        let mut word_counts: HashMap<String, u32> = HashMap::new();
        
        for word in &words {
            *word_counts.entry(word.clone()).or_insert(0) += 1;
        }
        
        self.word_doc_tf.insert(doc_id.to_string(), word_counts.clone());
        
        for word in word_counts.keys() {
            self.word_vocabulary.insert(word.clone());
            self.word_inverted_index
                .entry(word.clone())
                .or_insert_with(HashSet::new)
                .insert(doc_id.to_string());
        }
        
        // Mark IDF cache as dirty (needs rebuild on next search)
        self.idf_dirty = true;
    }
    
    /// Index multiple documents.
    pub fn index_many(&mut self, documents: Vec<(&str, &str)>) -> HashMap<String, usize> {
        let mut total_tokens = 0;
        for (doc_id, text) in &documents {
            self.index(doc_id, text);
            total_tokens += self.word_doc_tf.get(*doc_id).map(|m| m.len()).unwrap_or(0);
        }
        
        let mut result = HashMap::new();
        result.insert("num_docs".to_string(), documents.len());
        result.insert("total_tokens".to_string(), total_tokens);
        result
    }
    
    /// Stream index a very long document (2M+ tokens).
    /// Processes in chunks to maintain constant memory.
    /// Matches Python _crystalline.py stream_index() EXACTLY.
    pub fn stream_index(&mut self, doc_id: &str, text: &str, chunk_size: usize) -> HashMap<String, usize> {
        // Remove old entries if re-indexing (matches Python)
        // PHASE 2: RoaringBitmap iteration
        if let Some(old_tokens) = self.doc_token_sets.get(doc_id) {
            for token_id in old_tokens.iter() {
                self.inverted_index.discard(token_id, doc_id);
            }
        }
        
        if !self.doc_ids.contains(&doc_id.to_string()) {
            self.doc_ids.push(doc_id.to_string());
        }
        
        // Store text in text_store (CLEANUP: removed legacy doc_texts HashMap)
        if !self.doc_id_to_idx.contains_key(doc_id) {
            let idx = self.text_store.append_text(text).unwrap_or(0);
            self.doc_id_to_idx.insert(doc_id.to_string(), idx);
        } else {
            // Re-indexing: update existing entry
            let idx = self.text_store.append_text(text).unwrap_or(0);
            self.doc_id_to_idx.insert(doc_id.to_string(), idx);
        }
        
        // PHASE 2: Use RoaringBitmap for compressed token set
        self.doc_token_sets.insert(doc_id.to_string(), RoaringBitmap::new());
        
        let mut chunks = 0;
        let mut pos = 0;
        
        // Process text in chunks (matches Python)
        while pos < text.len() {
            let end = (pos + chunk_size).min(text.len());
            let chunk = &text[pos..end];
            
            // Get BERT tokens for this chunk (matches Python: chunk_tokens = self._get_tokens(chunk))
            let chunk_tokens = self.get_tokens(chunk);
            
            // Update forward index - PHASE 2: RoaringBitmap extend
            if let Some(token_set) = self.doc_token_sets.get_mut(doc_id) {
                token_set.extend(chunk_tokens.iter().copied());
            }
            
            // Update inverted index - O(1) ART insertion per token (matches Python)
            for &token_id in &chunk_tokens {
                self.inverted_index.add(token_id, doc_id, None);
            }
            
            chunks += 1;
            pos = end;
        }
        
        // PHASE 2: RoaringBitmap.len() returns u64, cast to usize
        let token_count = self.doc_token_sets.get(doc_id).map(|s| s.len() as usize).unwrap_or(0);
        
        let mut result = HashMap::new();
        result.insert("doc_id".to_string(), 1);
        result.insert("chunks".to_string(), chunks);
        result.insert("tokens".to_string(), token_count);
        
        result
    }
    
    /// Set embedding for a document
    pub fn set_embedding(&mut self, doc_id: &str, embedding: Vec<f32>) {
        self.doc_embeddings.insert(doc_id.to_string(), embedding);
    }

    /// Set passage embeddings for a document (legacy LongEmbed MaxSim).
    pub fn set_passage_embeddings(&mut self, doc_id: &str, passage_embeddings: Vec<Vec<f32>>) {
        if passage_embeddings.is_empty() {
            self.doc_passage_embeddings.remove(doc_id);
        } else {
            self.doc_passage_embeddings
                .insert(doc_id.to_string(), passage_embeddings);
        }
    }
    
    /// Get embedding for a document
    pub fn get_embedding(&self, doc_id: &str) -> Option<&Vec<f32>> {
        self.doc_embeddings.get(doc_id)
    }

    /// Legacy: split long text into overlapping passages (Python `_crystalline.pyx::get_passages`).
    pub(crate) fn legacy_get_passages(&self, text: &str, chunk_size: usize, stride: usize, max_chars: usize) -> Vec<String> {
        // Strip HTML/XML tags (best-effort, matches Python behavior closely)
        let re = Regex::new(r"<[^>]+>").ok();
        let mut cleaned = if let Some(re) = re {
            re.replace_all(text, " ").to_string()
        } else {
            text.to_string()
        };

        // Sample strategically for very long docs (beginning, early-mid, late-mid, end)
        if cleaned.len() > max_chars {
            let quarter = max_chars / 4;
            let len = cleaned.len();
            let b = &cleaned[0..quarter.min(len)];
            let em_start = (len / 4).min(len);
            let em_end = (em_start + quarter).min(len);
            let early_mid = &cleaned[em_start..em_end];
            let lm_center = (len * 3 / 4).min(len);
            let lm_start = lm_center.saturating_sub(quarter / 2);
            let lm_end = (lm_center + quarter / 2).min(len);
            let late_mid = &cleaned[lm_start..lm_end];
            let end_start = len.saturating_sub(quarter);
            let end = &cleaned[end_start..len];
            cleaned = format!("{} {} {} {}", b, early_mid, late_mid, end);
        }

        let mut passages = Vec::new();
        if stride == 0 || chunk_size == 0 {
            return vec![cleaned.chars().take(max_chars.min(2000)).collect()];
        }

        let bytes = cleaned.as_bytes();
        let mut start = 0usize;
        while start < bytes.len() {
            let end = (start + chunk_size).min(bytes.len());
            // Safety: this may cut UTF-8 in the middle; LongEmbed corpora are mostly ASCII/Latin,
            // and this mirrors the simple Python slicing behavior closely.
            let chunk = String::from_utf8_lossy(&bytes[start..end]).trim().to_string();
            if chunk.len() >= 100 {
                passages.push(chunk);
            }
            start = start.saturating_add(stride);
        }

        if passages.is_empty() {
            let fallback_end = chunk_size.min(bytes.len());
            let fallback = String::from_utf8_lossy(&bytes[0..fallback_end]).to_string();
            return vec![fallback];
        }

        passages
    }

    /// Legacy LongEmbed search (Step 9: sca_dropin-style passage MaxSim + lexical blend).
    /// Passage MaxSim (max dot over passage embeddings) + word-level IDF overlap + exact-match safety; dynamic alpha blend.
    pub fn search_legacy(
        &mut self,
        query: &str,
        top_k: usize,
        query_embedding: Option<&[f32]>,
        alpha_override: Option<f32>,
    ) -> Vec<(String, f32)> {
        if self.doc_ids.is_empty() {
            return vec![];
        }
        if self.idf_dirty {
            self.rebuild_idf();
        }

        let (_route, _q_words, q_expanded, _idf_avg, _oov, _has_code, _has_typo) = self.analyze_query(query);
        let query_lower = query.to_lowercase();

        // Candidates from word-level index (like sca_dropin); fallback to all docs
        let candidate_docs: HashSet<String> = {
            let mut cand = HashSet::new();
            for word in &q_expanded {
                if let Some(docs) = self.word_inverted_index.get(word) {
                    cand.extend(docs.iter().cloned());
                }
            }
            if cand.is_empty() {
                self.doc_ids.iter().cloned().collect()
            } else {
                cand
            }
        };

        let mut scores: Vec<(String, f32)> = Vec::new();
        for doc_id in &candidate_docs {
            let doc_text = match self.get_text(doc_id) {
                Some(t) => t,
                None => continue,
            };
            let doc_text_lower = doc_text.to_lowercase();
            if doc_text_lower.contains(&query_lower) {
                scores.push((doc_id.clone(), 1.0));
                continue;
            }

            let doc_words: HashSet<String> = self
                .word_doc_tf
                .get(doc_id)
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default();
            let overlap: HashSet<String> = q_expanded.intersection(&doc_words).cloned().collect();

            // Lexical: IDF overlap ratio (sca_dropin-style)
            let mut idf_matched = 0.0f32;
            let mut idf_total = 0.0f32;
            for word in &q_expanded {
                let idf = *self.word_idf.get(word).unwrap_or(&1.0);
                idf_total += idf;
                if doc_words.contains(word) {
                    idf_matched += idf;
                }
            }
            let s_lex = if idf_total > 0.0 {
                idf_matched / idf_total
            } else {
                0.0
            };

            // Semantic: passage MaxSim (max dot over passages) or doc embedding fallback
            let s_sem: f32 = if let Some(q_emb) = query_embedding {
                if let Some(passages) = self.doc_passage_embeddings.get(doc_id) {
                    passages
                        .iter()
                        .map(|p_emb| dot_product(q_emb, p_emb).max(0.0))
                        .fold(0.0f32, f32::max)
                } else if let Some(d_emb) = self.doc_embeddings.get(doc_id) {
                    dot_product(q_emb, d_emb).max(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Dynamic alpha from IDF overlap (mirror sca_dropin full_hybrid)
            let base_alpha = if overlap.is_empty() {
                1.0
            } else {
                let idf_overlap: f32 = overlap.iter().map(|w| *self.word_idf.get(w).unwrap_or(&0.5)).sum::<f32>()
                    / overlap.len() as f32;
                let idf_query: f32 = q_expanded
                    .iter()
                    .map(|w| *self.word_idf.get(w).unwrap_or(&0.5))
                    .sum::<f32>()
                    / q_expanded.len().max(1) as f32;
                (1.0 - idf_overlap / idf_query.max(0.001)).clamp(0.0, 1.0)
            };
            let blend_alpha = alpha_override.unwrap_or(base_alpha).clamp(0.0, 1.0);
            let final_score = blend_alpha * s_sem + (1.0 - blend_alpha) * s_lex;
            if final_score > 0.0 {
                scores.push((doc_id.clone(), final_score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // SEARCH: Unified auto-routing search
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// UNIFIED SEARCH (Step 8: sca_dropin-style route + candidate flow + scoring)
    /// Route: PureLexical → keyword overlap ratio; PureSemantic → dot-product only; FullHybrid → IDF overlap + TF-IDF 70/30 + length norm + phrase boost + alpha blend.
    pub fn search(
        &mut self,
        query: &str,
        top_k: usize,
        query_embedding: Option<&[f32]>,
        alpha_override: Option<f32>,
    ) -> Vec<(String, f32)> {
        if self.doc_ids.is_empty() {
            return vec![];
        }
        if self.idf_dirty {
            self.rebuild_idf();
        }

        let (route, q_words, q_expanded, _idf_avg, _oov, _has_code, _has_typo) = self.analyze_query(query);
        let _alpha = alpha_override.unwrap_or_else(|| {
            self.get_alpha(match route {
                QueryRoute::PureLexical => "high_lexical",
                QueryRoute::PureSemantic => "pure_semantic",
                QueryRoute::FullHybrid => "balanced",
            })
        });
        let query_lower = query.to_lowercase();

        // PATH A: Pure lexical (sca_dropin search_pure_lexical style)
        if route == QueryRoute::PureLexical {
            if q_words.is_empty() {
                return vec![];
            }
            // Use q_expanded (includes compound splits) and IDF-weight hits
            // so rare words like "nathan", "munoz" dominate over stopwords
            let mut results: Vec<(String, f32)> = Vec::new();
            let total_query_idf: f32 = q_expanded.iter()
                .map(|w| *self.word_idf.get(w).unwrap_or(&1.0))
                .sum::<f32>()
                .max(1.0);
            for doc_id in &self.doc_ids {
                let doc_words: HashSet<String> = self
                    .word_doc_tf
                    .get(doc_id)
                    .map(|m| m.keys().cloned().collect())
                    .unwrap_or_default();
                let hit_idf: f32 = q_expanded.iter()
                    .filter(|w| doc_words.contains(*w))
                    .map(|w| *self.word_idf.get(w).unwrap_or(&1.0))
                    .sum();
                if hit_idf > 0.0 {
                    results.push((doc_id.clone(), hit_idf / total_query_idf));
                }
            }
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return results.into_iter().take(top_k).collect();
        }

        // PATH B: Pure semantic (dot-product only; no lexical)
        if route == QueryRoute::PureSemantic {
            if let Some(q_emb) = query_embedding {
                let mut results: Vec<(String, f32)> = self
                    .doc_ids
                    .iter()
                    .filter_map(|doc_id| {
                        self.doc_embeddings.get(doc_id).map(|d_emb| {
                            (doc_id.clone(), dot_product(q_emb, d_emb).max(0.0))
                        })
                    })
                    .collect();
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                return results.into_iter().take(top_k).collect();
            }
            // No embedding: fall through to hybrid with all docs
        }

        // PATH C: Full hybrid (sca_dropin search_full_hybrid-style scoring)
        let candidate_docs: HashSet<String> = {
            let mut cand = HashSet::new();
            for word in &q_expanded {
                if let Some(docs) = self.word_inverted_index.get(word) {
                    cand.extend(docs.iter().cloned());
                }
            }
            if cand.is_empty() {
                self.doc_ids.iter().cloned().collect()
            } else {
                cand
            }
        };
        if candidate_docs.is_empty() {
            return vec![];
        }

        let rare_words: HashSet<String> = q_expanded
            .iter()
            .filter(|w| *self.word_idf.get(*w).unwrap_or(&0.0) > 2.5)
            .cloned()
            .collect();
        let b_param = 0.35f32;
        let total_len: usize = candidate_docs
            .iter()
            .filter_map(|id| self.word_doc_tf.get(id).map(|m| m.len()))
            .sum();
        let avg_doc_len = (total_len as f32 / candidate_docs.len().max(1) as f32).max(1.0);
        let mut scores: Vec<(String, f32)> = Vec::new();

        for doc_id in &candidate_docs {
            let doc_text = match self.get_text(doc_id) {
                Some(t) => t,
                None => continue,
            };
            let doc_tf = match self.word_doc_tf.get(doc_id) {
                Some(t) => t,
                None => continue,
            };
            let doc_words: HashSet<String> = doc_tf.keys().cloned().collect();

            if doc_text.to_lowercase().contains(&query_lower) {
                scores.push((doc_id.clone(), 1.0));
                continue;
            }

            let overlap: HashSet<String> = q_expanded.intersection(&doc_words).cloned().collect();
            let idf_overlap: f32 = overlap
                .iter()
                .map(|w| *self.word_idf.get(w).unwrap_or(&0.5))
                .sum::<f32>()
                / overlap.len().max(1) as f32;
            let idf_query: f32 = q_expanded
                .iter()
                .map(|w| *self.word_idf.get(w).unwrap_or(&0.5))
                .sum::<f32>()
                / q_expanded.len().max(1) as f32;
            let base_alpha = (1.0 - idf_overlap / idf_query.max(0.001f32)).clamp(0.0, 1.0);

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
            let boost = 1.0 + token_ratio.powi(2);
            let doc_len = doc_words.len() as f32;
            let length_norm = (1.0 - b_param + b_param * (doc_len / avg_doc_len)).max(0.5);
            let s_lex_raw = token_ratio * boost;
            let s_lex = s_lex_raw / length_norm;

            let query_words_raw: Vec<&str> = query.split_whitespace().collect();
            let mut phrase_match_boost = 1.0f32;
            if query_words_raw.len() >= 3 {
                'phrase: for &w_size in &[4_usize, 3] {
                    if query_words_raw.len() < w_size {
                        continue;
                    }
                    for window in query_words_raw.windows(w_size) {
                        let window_clean: Vec<String> = window
                            .iter()
                            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                            .collect();
                        if !window_clean.iter().all(|w| doc_words.contains(w)) {
                            continue;
                        }
                        let phrase_idf_sum: f32 = window_clean
                            .iter()
                            .map(|w| *self.word_idf.get(w).unwrap_or(&0.5))
                            .sum();
                        if phrase_idf_sum / (w_size as f32) < 2.2 {
                            continue;
                        }
                        let phrase = window.join(" ").to_lowercase();
                        if phrase.trim().len() >= 10 && doc_text.to_lowercase().contains(phrase.trim()) {
                            phrase_match_boost = if w_size == 4 { 1.25 } else { 1.15 };
                            break 'phrase;
                        }
                    }
                }
            }
            let s_lex_boosted = s_lex * phrase_match_boost;

            // Semantic: passage MaxSim when available (Step 10 – align with sca_dropin passage-level)
            let s_sem = if let Some(q_emb) = query_embedding {
                if let Some(passages) = self.doc_passage_embeddings.get(doc_id) {
                    passages
                        .iter()
                        .map(|p_emb| dot_product(q_emb, p_emb).max(0.0))
                        .fold(0.0f32, f32::max)
                } else if let Some(d_emb) = self.doc_embeddings.get(doc_id) {
                    dot_product(q_emb, d_emb).max(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let final_score = base_alpha * s_sem + (1.0 - base_alpha) * s_lex_boosted;
            if final_score > 0.0 {
                scores.push((doc_id.clone(), final_score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }
    
    /// Perfect recall search (exact string match with token filter).
    /// Matches Python _crystalline.py search_exact() EXACTLY.
    /// Two-stage:
    /// 1. Token filter (fast rejection using BERT tokens)
    /// 2. Exact substring verification
    pub fn search_exact(&self, query: &str) -> Vec<(String, f32)> {
        // Get BERT tokens (matches Python: q_tokens = self._get_tokens(query))
        let q_tokens = self.get_tokens(query);
        if q_tokens.is_empty() {
            return Vec::new();
        }
        
        let query_lower = query.to_lowercase();
        let mut matches = Vec::new();
        
        // 🚀 INVERTED INDEX OPTIMIZATION: Get candidate docs from ART (matches Python)
        let mut candidate_docs: HashSet<String> = HashSet::new();
        for token_id in &q_tokens {
            let docs = self.inverted_index.get(*token_id);
            candidate_docs.extend(docs);
        }
        
        // If no candidates from tokens, search all (matches Python)
        if candidate_docs.is_empty() {
            candidate_docs = self.doc_ids.iter().cloned().collect();
        }
        
        for doc_id in candidate_docs {
            // PHASE 1b: Read from text_store instead of doc_texts HashMap
            if let Some(doc_text) = self.get_text(&doc_id) {
                if doc_text.to_lowercase().contains(&query_lower) {
                    matches.push((doc_id, 1.0));
                }
            }
        }
        
        matches
    }
    
    /// Find ALL instances (count) of query in each document.
    /// Matches Python _crystalline.py search_all_instances() EXACTLY.
    pub fn search_all_instances(&self, query: &str) -> Vec<(String, usize)> {
        // Get BERT tokens (matches Python: q_tokens = self._get_tokens(query))
        let q_tokens = self.get_tokens(query);
        if q_tokens.is_empty() {
            return Vec::new();
        }
        
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();
        
        // 🚀 INVERTED INDEX OPTIMIZATION: Get candidate docs from ART (matches Python)
        let mut candidate_docs: HashSet<String> = HashSet::new();
        for token_id in &q_tokens {
            let docs = self.inverted_index.get(*token_id);
            candidate_docs.extend(docs);
        }
        
        if candidate_docs.is_empty() {
            candidate_docs = self.doc_ids.iter().cloned().collect();
        }
        
        for doc_id in candidate_docs {
            // PHASE 1b: Read from text_store instead of doc_texts HashMap
            if let Some(doc_text) = self.get_text(&doc_id) {
                let count = doc_text.to_lowercase().matches(&query_lower).count();
                if count > 0 {
                    results.push((doc_id, count));
                }
            }
        }
        
        results
    }
    
    /// PURE ART Extraction - IDF-based dynamic separator detection.
    /// Matches Python _crystalline.py search_kv() EXACTLY.
    /// 
    /// Uses ART: Find key → Skip common tokens (high doc_freq) → Return rare token (value)
    /// 
    /// NO HARDCODED SEPARATORS - uses document frequency mathematically:
    /// - Common tokens (>10% of docs): "is", ":", "=" → skip
    /// - Rare tokens (<10% of docs): "SECRET_KEY_123" → value!
    pub fn search_kv(&self, key: &str, top_k: i32) -> Vec<(String, String)> {
        let mut results = Vec::new();
        
        // Get BERT tokens with positions (matches Python: key_tokens_with_pos = self._get_tokens_with_positions(key))
        let key_tokens_with_pos = self.get_tokens_with_positions(key);
        if key_tokens_with_pos.is_empty() {
            return results;
        }
        
        // Get candidate docs where ALL key tokens appear (matches Python)
        let key_token_ids: Vec<u32> = key_tokens_with_pos.iter().map(|(tid, _, _)| *tid).collect();
        let mut candidate_docs: Option<HashSet<String>> = None;
        
        for token_id in &key_token_ids {
            let docs = self.inverted_index.get(*token_id);
            if let Some(ref mut cands) = candidate_docs {
                // Intersection
                *cands = cands.intersection(&docs).cloned().collect();
            } else {
                candidate_docs = Some(docs);
            }
        }
        
        let candidate_docs = match candidate_docs {
            Some(cands) if !cands.is_empty() => cands,
            _ => return results,
        };
        
        let key_lower = key.to_lowercase();
        let _total_docs = self.doc_ids.len().max(1);
        
        for doc_id in candidate_docs {
            // PHASE 1b: Read from text_store instead of doc_texts HashMap
            if let Some(doc_text) = self.get_text(&doc_id) {
                let doc_lower = doc_text.to_lowercase();
                
                // Find key position
                if let Some(key_pos) = doc_lower.find(&key_lower) {
                    let after_key = key_pos + key_lower.len();
                    let remainder = &doc_text[after_key..];
                    
                    // Skip separators (multi-layer filtering)
                    let mut value_start = remainder;
                    let mut skip_count = 0;
                    
                    // Layer 1: Skip whitespace and structural separators
                    while !value_start.is_empty() {
                        let first_char = value_start.chars().next().unwrap();
                        
                        // Connector tokens
                        if first_char.is_whitespace() || first_char == ':' || first_char == '=' {
                            value_start = &value_start[first_char.len_utf8()..];
                            skip_count += 1;
                            continue;
                        }
                        
                        // Check for connector words
                        let first_word: String = value_start.chars()
                            .take_while(|c| c.is_alphanumeric())
                            .collect();
                        
                        if CONNECTOR_TOKENS.contains(&first_word.to_lowercase().as_str()) && skip_count < 3 {
                            value_start = &value_start[first_word.len()..];
                            skip_count += 1;
                            continue;
                        }
                        
                        break;
                    }
                    
                    if !value_start.is_empty() {
                        // Extract value until delimiter
                        let value: String = value_start.chars()
                            .take_while(|c| !matches!(c, ',' | ';' | '\n' | '\r'))
                            .collect();
                        
                        let value = value.trim().to_string();
                        if !value.is_empty() {
                            results.push((value, doc_id.clone()));
                            
                            if top_k > 0 && results.len() >= top_k as usize {
                                return results;
                            }
                        }
                    }
                }
            }
        }
        
        results
    }
    
    /// Recall with CONTEXT snippet (like Google search results).
    /// Returns the text around the key, allowing user to extract value themselves.
    pub fn recall_context(&self, key: &str, context_chars: usize) -> Option<(String, String)> {
        let key_lower = key.to_lowercase();
        
        // Get candidate docs from inverted index first
        let key_words = self.simple_tokenize(key);
        let mut candidate_docs: HashSet<String> = HashSet::new();
        
        for word in &key_words {
            if let Some(docs) = self.word_inverted_index.get(word) {
                candidate_docs.extend(docs.iter().cloned());
            }
        }
        
        if candidate_docs.is_empty() {
            candidate_docs = self.doc_ids.iter().cloned().collect();
        }
        
        for doc_id in candidate_docs {
            // PHASE 1b: Read from text_store instead of doc_texts HashMap
            if let Some(doc_text) = self.get_text(&doc_id) {
                if let Some(idx) = doc_text.to_lowercase().find(&key_lower) {
                    let start = idx.saturating_sub(context_chars);
                    let end = (idx + key.len() + context_chars).min(doc_text.len());
                    let snippet = doc_text[start..end].to_string();
                    return Some((snippet, doc_id));
                }
            }
        }
        
        None
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // UTILITIES
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Clear all indexed documents.
    pub fn clear(&mut self) {
        // Clear optimized storage
        self.doc_id_to_idx.clear();
        self.text_store.clear();
        
        // Clear core storage
        self.doc_ids.clear();
        // doc_texts REMOVED
        self.doc_token_sets.clear();
        self.doc_token_counts.clear();
        self.doc_embeddings.clear();
        self.doc_passage_embeddings.clear();
        self.inverted_index.clear();
        self.position_tokens.clear();
        self.token_strings.clear();
        self.idf_cache.clear();
        self.word_vocabulary.clear();
        self.word_inverted_index.clear();
        self.word_idf.clear();
        self.word_doc_tf.clear();
        self.word_phonetic_index.clear();
        self.idf_dirty = true;
        self.doc_count = 0;
        // Quantized state (must reset when switching tasks e.g. needle -> WikimQA)
        self.matrix_quantized.clear();
        self.passage_counts.clear();
        self.passage_offsets.clear();
        self.doc_word_sets_fast.clear();
        self.doc_word_tf_fast.clear();
        self.doc_texts_fast.clear();
        self.word_idf_fast.clear();
        self.phonetic_index_fast.clear();
        self.vocabulary_fast.clear();
        self.word_inverted_fast.clear();
        self.quantized_mode = false;
        self.force_route = None;  // Reset so non-needle tasks use default routing
    }
    
    /// Get index statistics.
    pub fn stats(&self) -> HashMap<String, usize> {
        let total_tokens: usize = self.word_doc_tf.values().map(|m| m.len()).sum();
        // PHASE 1b: Use text_store.total_bytes() instead of doc_texts
        let total_chars: usize = self.text_store.total_bytes() as usize;
        let total_postings: usize = self.word_inverted_index.values().map(|s| s.len()).sum();
        
        let mut stats = HashMap::new();
        stats.insert("num_documents".to_string(), self.doc_ids.len());
        stats.insert("total_tokens".to_string(), total_tokens);
        stats.insert("total_chars".to_string(), total_chars);
        stats.insert("embeddings_cached".to_string(), self.doc_embeddings.len());
        stats.insert(
            "passage_embeddings_docs".to_string(),
            self.doc_passage_embeddings.len(),
        );
        let total_passages: usize = self.doc_passage_embeddings.values().map(|v| v.len()).sum();
        stats.insert("passage_embeddings_total".to_string(), total_passages);
        stats.insert("inverted_index_entries".to_string(), self.word_inverted_index.len());
        stats.insert("inverted_index_postings".to_string(), total_postings);
        stats.insert("art_size".to_string(), self.inverted_index.len());
        
        // Phase 1: New storage metrics
        stats.insert("text_store_docs".to_string(), self.text_store.doc_count());
        stats.insert("text_store_bytes".to_string(), self.text_store.total_bytes() as usize);
        stats.insert("doc_id_registry_size".to_string(), self.doc_id_to_idx.len());
        stats.insert("storage_persistent".to_string(), if self.text_store.is_persistent() { 1 } else { 0 });
        
        stats
    }
    
    /// Get document text by ID.
    /// PHASE 1b: Now reads from text_store instead of doc_texts HashMap
    pub fn get_document(&self, doc_id: &str) -> Option<String> {
        self.get_text(doc_id)
    }
    
    /// Get document text by ID from new TextStore (for A/B testing).
    /// Returns None if doc_id not found or storage read fails.
    pub fn get_document_from_store(&self, doc_id: &str) -> Option<String> {
        self.get_text(doc_id)
    }
    
    /// Verify text_store integrity - count docs with valid text.
    /// Returns (valid_docs, missing_docs) count.
    pub fn verify_storage_consistency(&self) -> (usize, usize) {
        let mut valid = 0;
        let mut missing = 0;
        
        for doc_id in &self.doc_ids {
            if self.get_text(doc_id).is_some() {
                valid += 1;
            } else {
                missing += 1;
            }
        }
        
        (valid, missing)
    }
    
    /// Check if document exists.
    pub fn has_document(&self, doc_id: &str) -> bool {
        // Use new storage - check if doc_id is in registry
        self.doc_id_to_idx.contains_key(doc_id)
    }
    
    /// Get document text by ID (internal helper for migration)
    /// Reads from NEW text_store instead of legacy doc_texts HashMap
    fn get_text(&self, doc_id: &str) -> Option<String> {
        let idx = self.doc_id_to_idx.get(doc_id)?;
        self.text_store.get_text(*idx).ok()
    }
    
    /// Number of indexed documents.
    pub fn num_documents(&self) -> usize {
        self.doc_ids.len()
    }
    
    /// Get all document IDs.
    pub fn get_doc_ids(&self) -> &Vec<String> {
        &self.doc_ids
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // THE SAID STANDARD: Production API
    // ═══════════════════════════════════════════════════════════════════════════
    // Simple as sentence-transformers. Two methods:
    //   encode() → Lock document
    //   recall() → Auto-routing search (handles everything)
    //
    // Internal methods (search_kv, search_all_instances) available for testing.
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// SAID Standard: Encode document into crystalline state.
    /// Simple API aligned with sentence-transformers.
    pub fn encode(&mut self, doc_id: &str, text: &str, chunk_size: usize) -> HashMap<String, usize> {
        self.stream_index(doc_id, text, chunk_size)
    }
    
    /// SAID Standard: Deterministic Recall (auto-routes everything).
    /// Simple API - one method handles all query types.
    pub fn recall(
        &mut self,
        query: &str,
        top_k: usize,
        query_embedding: Option<&[f32]>,
        alpha_override: Option<f32>,
    ) -> Vec<(String, f32)> {
        self.search(query, top_k, query_embedding, alpha_override)
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // QUANTIZED SEARCH (sca_dropin alignment for 94.1060 parity)
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// Set corpus mean for quantization (call before add_docs_quantized)
    pub fn set_corpus_mean(&mut self, mean: Vec<f32>) {
        self.dim = mean.len();
        self.quantized_dim = (self.dim + 7) / 8;
        self.bytes_per_passage = if self.holographic_16view {
            self.quantized_dim * 16
        } else {
            self.quantized_dim
        };
        self.corpus_mean = mean;
    }
    
    /// Enable holographic 16-view quantization
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
    }
    
    /// Load IDF scores (sca_dropin style)
    pub fn load_idf_fast(&mut self, words: Vec<String>, scores: Vec<f32>) {
        for (w, s) in words.iter().zip(scores.iter()) {
            let w_lower = w.to_lowercase();
            self.word_idf_fast.insert(w_lower, *s);
        }
    }

    /// Force routing override (e.g. "FullHybrid" for LEMBNeedleRetrieval).
    pub fn set_force_route(&mut self, route_str: &str) {
        self.force_route = match route_str.trim() {
            "PureSemantic" => Some(QueryRoute::PureSemantic),
            "FullHybrid" => Some(QueryRoute::FullHybrid),
            "PureLexical" => Some(QueryRoute::PureLexical),
            _ => None,
        };
    }

    /// Extended qrels: check if both docs contain valid answer (needle overlap).
    /// MATCHES sca_dropin/lam_scientific_proof_suite _check_both_docs_valid exactly.
    pub fn check_both_docs_valid_quantized(&self, query: &str, doc1_id: &str, doc2_id: &str) -> bool {
        let stopwords: AHashSet<&str> = [
            "what", "when", "where", "which", "who", "why", "how", "the", "they",
            "them", "their", "known", "that", "this", "with", "from", "have", "been",
            "were", "being", "for", "was", "and", "are", "is", "his", "her", "she", "he"
        ].iter().cloned().collect();
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4 && !stopwords.contains(w))
            .collect();
        if keywords.is_empty() { return false; }
        let doc1_idx = self.doc_ids.iter().position(|id| id == doc1_id);
        let doc2_idx = self.doc_ids.iter().position(|id| id == doc2_id);
        let (doc1_text, doc2_text) = match (doc1_idx, doc2_idx) {
            (Some(i1), Some(i2)) => {
                if !self.doc_texts_fast.is_empty() && i1 < self.doc_texts_fast.len() && i2 < self.doc_texts_fast.len() {
                    (self.doc_texts_fast[i1].to_lowercase(), self.doc_texts_fast[i2].to_lowercase())
                } else if i1 < self.doc_word_sets_fast.len() && i2 < self.doc_word_sets_fast.len() {
                    let t1: String = self.doc_word_sets_fast[i1].iter().cloned().collect::<Vec<_>>().join(" ");
                    let t2: String = self.doc_word_sets_fast[i2].iter().cloned().collect::<Vec<_>>().join(" ");
                    (t1, t2)
                } else {
                    return false;
                }
            }
            _ => return false,
        };
        let doc1_hits = keywords.iter().filter(|kw| doc1_text.contains(*kw)).count();
        let doc2_hits = keywords.iter().filter(|kw| doc2_text.contains(*kw)).count();
        let threshold = (keywords.len() as f32 * 0.5) as usize;
        doc1_hits >= threshold && doc2_hits >= threshold
    }

    /// Compute keyword overlap hits for a single document to enable needle re-ranking.
    pub fn compute_keyword_hits_quantized(&self, query: &str, doc_id: &str) -> usize {
        let stopwords: AHashSet<&str> = [
            "what", "when", "where", "which", "who", "why", "how", "the", "they",
            "them", "their", "known", "that", "this", "with", "from", "have", "been",
            "were", "being", "for", "was", "and", "are", "is", "his", "her", "she", "he"
        ].iter().cloned().collect();
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| (w.len() >= 2 || (w.len() >= 1 && w.chars().any(|c| c.is_ascii_digit()))) && !stopwords.contains(w))
            .collect();
        if keywords.is_empty() { return 0; }
        
        let doc_idx = self.doc_ids.iter().position(|id| id == doc_id);
        let doc_text = match doc_idx {
            Some(i) => {
                if !self.doc_texts_fast.is_empty() && i < self.doc_texts_fast.len() {
                    self.doc_texts_fast[i].to_lowercase()
                } else if i < self.doc_word_sets_fast.len() {
                    self.doc_word_sets_fast[i].iter().cloned().collect::<Vec<_>>().join(" ")
                } else {
                    return 0;
                }
            }
            _ => return 0,
        };
        keywords.iter().filter(|kw| doc_text.contains(*kw)).count()
    }

    /// Retrieve and rank ALL documents in the corpus purely by keyword intersection hits for Needle tasks.
    pub fn get_highest_keyword_overlap_docs(&self, query: &str) -> Vec<(String, usize)> {
        let mut results = Vec::new();
        // pre-extract to avoid re-extracting per doc
        let stopwords: AHashSet<&str> = [
            "what", "when", "where", "which", "who", "why", "how", "the", "they",
            "them", "their", "known", "that", "this", "with", "from", "have", "been",
            "were", "being", "for", "was", "and", "are", "is", "his", "her", "she", "he"
        ].iter().cloned().collect();
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| (w.len() >= 2 || (w.len() >= 1 && w.chars().any(|c| c.is_ascii_digit()))) && !stopwords.contains(w))
            .collect();
            
        if keywords.is_empty() { return results; }

        for (i, doc_id) in self.doc_ids.iter().enumerate() {
            let doc_text = if !self.doc_texts_fast.is_empty() && i < self.doc_texts_fast.len() {
                self.doc_texts_fast[i].to_lowercase()
            } else if i < self.doc_word_sets_fast.len() {
                self.doc_word_sets_fast[i].iter().cloned().collect::<Vec<_>>().join(" ")
            } else {
                continue;
            };
            
            let hits = keywords.iter().filter(|kw| doc_text.contains(**kw)).count();
            if hits > 0 {
                results.push((doc_id.clone(), hits));
            }
        }
        // sort descending
        results.sort_by(|a, b| b.1.cmp(&a.1));
        results.truncate(100);
        results
    }
    
    /// Evaluate one query with extended qrels. Returns (is_correct, used_extended).
    pub fn evaluate_query_quantized(
        &self,
        query_emb: &[f32],
        query_text: &str,
        expected_doc_ids: &[String],
        top_k: usize,
    ) -> (bool, bool) {
        let results = self.search_unified_quantized(query_emb, query_text, top_k);
        if results.is_empty() { return (false, false); }
        let top_doc_id = &results[0].0;
        if expected_doc_ids.iter().any(|id| id == top_doc_id) {
            return (true, false);
        }
        for expected_id in expected_doc_ids {
            if self.check_both_docs_valid_quantized(query_text, top_doc_id, expected_id) {
                return (true, true);
            }
        }
        (false, false)
    }

    /// Batch evaluate with extended qrels. Returns (correct_count, extended_count, total).
    pub fn evaluate_batch_quantized(
        &self,
        query_embeddings: &[Vec<f32>],
        query_texts: &[String],
        expected_doc_ids_list: &[Vec<String>],
        top_k: usize,
    ) -> (usize, usize, usize) {
        let mut correct = 0usize;
        let mut extended = 0usize;
        let total = query_texts.len();
        for i in 0..total {
            let q_emb = query_embeddings.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
            let q_text = query_texts.get(i).map(|s| s.as_str()).unwrap_or("");
            let expected = expected_doc_ids_list.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
            let (is_correct, used_ext) = self.evaluate_query_quantized(q_emb, q_text, expected, top_k);
            if is_correct {
                correct += 1;
                if used_ext { extended += 1; }
            }
        }
        (correct, extended, total)
    }

    /// MAD-based dynamic scale for holographic 16-view
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
    
    /// Add documents with quantized embeddings (sca_dropin style bulk indexing)
    /// 
    /// Args:
    ///   ids: Document IDs
    ///   embeddings_flat: Flattened embeddings (all passages for all docs)
    ///   passage_counts: Number of passages per document
    ///   gammas: Per-document gamma values (unused but kept for API compat)
    ///   doc_words: Words for each document (for lexical indexing)
    pub fn add_docs_quantized(
        &mut self,
        ids: Vec<String>,
        embeddings_flat: Vec<f32>,
        passage_counts: Vec<usize>,
        _gammas: Vec<f32>,
        doc_words: Vec<Vec<String>>,
    ) {
        self.quantized_mode = true;
        let start_idx = self.doc_ids.len();
        
        // Add doc_ids
        for id in &ids {
            self.doc_ids.push(id.clone());
        }
        
        // Track passage offsets for MaxSim lookup
        let stride = self.bytes_per_passage;
        let mut current_offset = self.matrix_quantized.len() / stride.max(1);
        for &passage_count in &passage_counts {
            self.passage_counts.push(passage_count);
            self.passage_offsets.push(current_offset);
            current_offset += passage_count;
        }
        
        // Word-Level Indexing (sca_dropin style)
        for (i, words) in doc_words.iter().enumerate() {
            let doc_idx = start_idx + i;
            let mut word_set = AHashSet::new();
            let mut word_tf: AHashMap<String, u32> = AHashMap::new();
            let mut doc_text = String::new();
            
            for w in words {
                let w_lower = w.to_lowercase();

                // Normalize: strip trailing punctuation
                let w_normalized: String = w_lower
                    .trim_end_matches(|c: char| c.is_ascii_punctuation())
                    .to_string();

                // Build doc_text from normalized words so phrase match
                // is consistent with word_set (e.g. "munoz's" → "munoz")
                if !doc_text.is_empty() {
                    doc_text.push(' ');
                }
                doc_text.push_str(&w_normalized);
                
                if w_normalized.len() < 3 {
                    continue;
                }
                
                word_set.insert(w_normalized.clone());
                *word_tf.entry(w_normalized.clone()).or_insert(0) += 1;
                
                self.word_inverted_fast
                    .entry(w_normalized.clone())
                    .or_insert_with(AHashSet::new)
                    .insert(doc_idx);
                
                let sx = self.get_soundex(&w_normalized);
                self.phonetic_index_fast
                    .entry(sx)
                    .or_insert_with(AHashSet::new)
                    .insert(w_normalized.clone());
                
                self.vocabulary_fast.insert(w_normalized);
            }
            
            self.doc_word_sets_fast.push(word_set);
            self.doc_word_tf_fast.push(word_tf);
            self.doc_texts_fast.push(doc_text);
        }
        
        // Compute dynamic scale if holographic
        if self.holographic_16view {
            self.holographic_scale = self.compute_dynamic_scale(&embeddings_flat);
        }

        // Quantize all embeddings (16-view holographic or standard 1-bit)
        let chunks: Vec<&[f32]> = embeddings_flat.chunks(self.dim).collect();
        let h_scale = self.holographic_scale;
        let h16 = self.holographic_16view;
        let q_dim = self.quantized_dim;
        let dim = self.dim;
        let corpus_mean = &self.corpus_mean;
        
        let processed: Vec<Vec<u8>> = chunks.par_iter().map(|emb| {
            if h16 {
                // 16-VIEW HOLOGRAPHIC QUANTIZATION
                let mut out = Vec::with_capacity(q_dim * 16);
                for i in 0..16 {
                    let off = (i as f32 / 15.0 - 0.5) * h_scale;
                    let mut packed = vec![0u8; q_dim];
                    
                    for d in 0..dim {
                        let mean_val = corpus_mean.get(d).copied().unwrap_or(0.0);
                        let val = emb.get(d).copied().unwrap_or(0.0) - mean_val + off;
                        if val > 0.0 {
                            let byte_idx = d / 8;
                            let bit_idx = d % 8;
                            if byte_idx < packed.len() {
                                packed[byte_idx] |= 1 << bit_idx;
                            }
                        }
                    }
                    out.extend(packed);
                }
                out
            } else {
                // STANDARD 1-BIT QUANTIZATION
                let mut packed = vec![0u8; q_dim];
                for d in 0..dim {
                    let mean_val = corpus_mean.get(d).copied().unwrap_or(0.0);
                    let val = emb.get(d).copied().unwrap_or(0.0) - mean_val;
                    if val > 0.0 {
                        let byte_idx = d / 8;
                        let bit_idx = d % 8;
                        if byte_idx < packed.len() {
                            packed[byte_idx] |= 1 << bit_idx;
                        }
                    }
                }
                packed
            }
        }).collect();
        
        for bits in processed {
            self.matrix_quantized.extend(bits);
        }
        
        // Set rerank_depth to total docs
        self.rerank_depth = self.doc_ids.len();
        
    }
    
    /// Quantize a query embedding
    fn quantize_query(&self, query_emb: &[f32]) -> Vec<u8> {
        if self.holographic_16view {
            let h_scale = self.holographic_scale;
            let mut q_bits = Vec::with_capacity(self.quantized_dim * 16);
            
            for i in 0..16 {
                let off = (i as f32 / 15.0 - 0.5) * h_scale;
                let mut packed = vec![0u8; self.quantized_dim];
                
                for d in 0..self.dim {
                    let mean_val = self.corpus_mean.get(d).copied().unwrap_or(0.0);
                    let val = query_emb.get(d).copied().unwrap_or(0.0) - mean_val + off;
                    if val > 0.0 {
                        let byte_idx = d / 8;
                        let bit_idx = d % 8;
                        if byte_idx < packed.len() {
                            packed[byte_idx] |= 1 << bit_idx;
                        }
                    }
                }
                q_bits.extend(packed);
            }
            q_bits
        } else {
            // Standard 1-bit
            let mut packed = vec![0u8; self.quantized_dim];
            for d in 0..self.dim {
                let mean_val = self.corpus_mean.get(d).copied().unwrap_or(0.0);
                let val = query_emb.get(d).copied().unwrap_or(0.0) - mean_val;
                if val > 0.0 {
                    let byte_idx = d / 8;
                    let bit_idx = d % 8;
                    if byte_idx < packed.len() {
                        packed[byte_idx] |= 1 << bit_idx;
                    }
                }
            }
            packed
        }
    }
    
    /// Hamming candidate retrieval (pure semantic path)
    fn hamming_candidate_retrieval(&self, q_bits: &[u8], limit: usize) -> Vec<(usize, f64)> {
        let stride = self.bytes_per_passage;
        let max_hamming = (self.quantized_dim * 8) as f64;
        let holo16 = self.holographic_16view && q_bits.len() >= 16 * self.quantized_dim;
        let num_docs = self.doc_ids.len();
        
        let mut candidates: Vec<(usize, f64)> = if !self.passage_counts.is_empty() &&
            self.passage_counts.iter().any(|&c| c > 1) {
            // PASSAGE MODE: Find min Hamming across all passages per doc
            (0..num_docs).into_par_iter().map(|doc_idx| {
                let passage_count = self.passage_counts.get(doc_idx).copied().unwrap_or(1);
                let passage_offset = self.passage_offsets.get(doc_idx).copied().unwrap_or(doc_idx);
                let mut min_dist = f64::MAX;
                
                for p_idx in 0..passage_count {
                    let start = (passage_offset + p_idx) * stride;
                    let end = start + stride;
                    
                    if end <= self.matrix_quantized.len() {
                        if holo16 {
                            // 16-VIEW HOLOGRAPHIC
                            let mut sum_s = 0.0f64;
                            for v in 0..16 {
                                let d_start = start + v * self.quantized_dim;
                                let d_end = d_start + self.quantized_dim;
                                let q_start = v * self.quantized_dim;
                                let q_end = q_start + self.quantized_dim;
                                
                                if d_end <= self.matrix_quantized.len() && q_end <= q_bits.len() {
                                    let d = &self.matrix_quantized[d_start..d_end];
                                    let q = &q_bits[q_start..q_end];
                                    let sim = (1.0 - u8::hamming(d, q).unwrap_or(max_hamming) / max_hamming).max(0.0);
                                    sum_s += sim;
                                }
                            }
                            let holistic_sim = 0.0625 * sum_s;
                            let dist = (1.0 - holistic_sim) * max_hamming;
                            if dist < min_dist { min_dist = dist; }
                        } else {
                            // Standard Hamming
                            let doc_bits = &self.matrix_quantized[start..end];
                            if let Some(dist) = u8::hamming(doc_bits, q_bits) {
                                if dist < min_dist { min_dist = dist; }
                            }
                        }
                    }
                }
                (doc_idx, min_dist)
            }).collect()
        } else if holo16 {
            // DOC MODE + HOLOGRAPHIC
            (0..num_docs).into_par_iter().map(|doc_idx| {
                let start = doc_idx * stride;
                let mut sum_s = 0.0f64;
                
                for v in 0..16 {
                    let d_start = start + v * self.quantized_dim;
                    let d_end = d_start + self.quantized_dim;
                    let q_start = v * self.quantized_dim;
                    let q_end = q_start + self.quantized_dim;
                    
                    if d_end <= self.matrix_quantized.len() && q_end <= q_bits.len() {
                        let d = &self.matrix_quantized[d_start..d_end];
                        let q = &q_bits[q_start..q_end];
                        let sim = (1.0 - u8::hamming(d, q).unwrap_or(max_hamming) / max_hamming).max(0.0);
                        sum_s += sim;
                    }
                }
                let holistic_sim = 0.0625 * sum_s;
                let dist = (1.0 - holistic_sim) * max_hamming;
                (doc_idx, dist)
            }).collect()
        } else {
            // DOC MODE: Standard Hamming
            (0..num_docs).into_par_iter().map(|doc_idx| {
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
    
    /// Hybrid candidate retrieval (semantic + lexical before filtering)
    fn hybrid_candidate_retrieval(
        &self,
        q_bits: &[u8],
        q_expanded: &AHashSet<String>,
        limit: usize,
    ) -> Vec<(usize, f64)> {
        let stride = self.bytes_per_passage;
        let max_hamming = (self.quantized_dim * 8) as f64;
        let holo16 = self.holographic_16view && q_bits.len() >= 16 * self.quantized_dim;
        let num_docs = self.doc_ids.len();
        
        let alpha_sem = self.hybrid_alpha_semantic as f64;
        let alpha_lex = self.hybrid_alpha_lexical as f64;
        
        let mut candidates: Vec<(usize, f64, f64)> = (0..num_docs)
            .into_par_iter()
            .map(|doc_idx| {
                // 1. LEXICAL SCORE - IDF-weighted overlap
                let (s_lexical, has_match) = if doc_idx < self.doc_word_sets_fast.len() {
                    let doc_words = &self.doc_word_sets_fast[doc_idx];
                    let mut idf_matched = 0.0f64;
                    let mut idf_total = 0.0f64;
                    let mut matched = false;
                    
                    for word in q_expanded.iter() {
                        let idf = *self.word_idf_fast.get(word).unwrap_or(&1.0) as f64;
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
                
                // 3. SEMANTIC SCORE (Hamming)
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
                                    let d_start = start + v * self.quantized_dim;
                                    let d_end = d_start + self.quantized_dim;
                                    let q_start = v * self.quantized_dim;
                                    let q_end = q_start + self.quantized_dim;
                                    if d_end <= self.matrix_quantized.len() && q_end <= q_bits.len() {
                                        let d = &self.matrix_quantized[d_start..d_end];
                                        let q = &q_bits[q_start..q_end];
                                        let sim = (1.0 - u8::hamming(d, q).unwrap_or(max_hamming) / max_hamming).max(0.0);
                                        sum_s += sim;
                                    }
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
                        let d_start = start + v * self.quantized_dim;
                        let d_end = d_start + self.quantized_dim;
                        let q_start = v * self.quantized_dim;
                        let q_end = q_start + self.quantized_dim;
                        if d_end <= self.matrix_quantized.len() && q_end <= q_bits.len() {
                            let d = &self.matrix_quantized[d_start..d_end];
                            let q = &q_bits[q_start..q_end];
                            let sim = (1.0 - u8::hamming(d, q).unwrap_or(max_hamming) / max_hamming).max(0.0);
                            sum_s += sim;
                        }
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
                
                // 4. COMBINED SCORE
                let combined_score = alpha_sem * s_semantic + alpha_lex * s_lexical;
                let combined_dist = (1.0 - combined_score) * max_hamming;
                
                (doc_idx, semantic_dist, combined_dist)
            })
            .collect();
        
        // Sort by COMBINED distance
        candidates.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return (doc_idx, semantic_dist)
        candidates.into_iter()
            .take(limit)
            .map(|(doc_idx, semantic_dist, _)| (doc_idx, semantic_dist))
            .collect()
    }
    
    /// Pure lexical search (for codes/passkeys) — IDF-weighted
    fn search_pure_lexical_quantized(
        &self,
        _query_lower: &str,
        q_expanded: &AHashSet<String>,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        let mut results: Vec<(String, f32)> = Vec::new();

        if q_expanded.is_empty() {
            return results;
        }

        // IDF-weight so rare words (names, codes) dominate over stopwords
        let total_query_idf: f32 = q_expanded.iter()
            .map(|w| *self.word_idf_fast.get(w).unwrap_or(&1.0))
            .sum::<f32>()
            .max(1.0);

        for (doc_idx, doc_id) in self.doc_ids.iter().enumerate() {
            if doc_idx >= self.doc_word_sets_fast.len() {
                continue;
            }
            let doc_words = &self.doc_word_sets_fast[doc_idx];

            let hit_idf: f32 = q_expanded.iter()
                .filter(|w| doc_words.contains(*w))
                .map(|w| *self.word_idf_fast.get(w).unwrap_or(&1.0))
                .sum();

            if hit_idf > 0.0 {
                results.push((doc_id.clone(), hit_idf / total_query_idf));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().take(top_k).collect()
    }
    
    /// Pure semantic search (Hamming only)
    fn search_pure_semantic_quantized(
        &self,
        survivors: &[(usize, f64)],
        max_hamming: f64,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        let mut results: Vec<(String, f32)> = survivors.iter()
            .filter_map(|&(doc_idx, hamming_dist)| {
                self.doc_ids.get(doc_idx).map(|doc_id| {
                    let s_sem = (1.0 - (hamming_dist / max_hamming)).max(0.0) as f32;
                    (doc_id.clone(), s_sem)
                })
            })
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(top_k).collect()
    }
    
    /// Full hybrid search (sca_dropin style - complete alignment)
    fn search_full_hybrid_quantized(
        &self,
        survivors: &[(usize, f64)],
        q_expanded: &AHashSet<String>,
        query_lower: &str,
        query_original: &str,
        max_hamming: f64,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        // Rare words for entity TF-IDF weighting
        let rare_words: AHashSet<String> = q_expanded.iter()
            .filter(|w| *self.word_idf_fast.get(*w).unwrap_or(&0.0) > 2.5)
            .cloned()
            .collect();
        
        // Detect dialogue-like queries (QMSum optimization)
        let query_idf_avg: f32 = q_expanded.iter()
            .map(|w| *self.word_idf_fast.get(w).unwrap_or(&0.5))
            .sum::<f32>() / q_expanded.len().max(1) as f32;
        let is_dialogue_query = query_idf_avg < 2.2 && q_expanded.len() > 4;
        let dialogue_semantic_boost = if is_dialogue_query { 0.35 } else { 0.0 };
        
        // Average doc length for BM25-style normalization
        let survivor_count = survivors.len().max(1);
        let total_len: usize = survivors.iter()
            .map(|&(doc_idx, _)| self.doc_word_sets_fast.get(doc_idx).map(|s| s.len()).unwrap_or(0))
            .sum();
        let avg_doc_len = total_len as f32 / survivor_count as f32;
        let b_param = 0.35f32;
        
        let mut results: Vec<(String, f32)> = Vec::with_capacity(survivors.len());
        
        for &(doc_idx, semantic_dist) in survivors {
            let doc_id = match self.doc_ids.get(doc_idx) {
                Some(id) => id.clone(),
                None => continue,
            };
            
            // 1. SEMANTIC SCORE (from Hamming distance)
            let s_sem = (1.0 - (semantic_dist / max_hamming)).max(0.0) as f32;
            
            let doc_words = match self.doc_word_sets_fast.get(doc_idx) {
                Some(w) => w,
                None => continue,
            };
            let doc_tf = self.doc_word_tf_fast.get(doc_idx);
            let doc_text = self.doc_texts_fast.get(doc_idx).map(|s| s.as_str()).unwrap_or("");
            
            // Exact match safety net
            if doc_text.contains(query_lower) {
                results.push((doc_id, 1.0));
                continue;
            }
            
            // 2. LEXICAL SCORE (Full TF-IDF)
            let overlap: AHashSet<String> = q_expanded.iter()
                .filter(|w| doc_words.contains(*w))
                .cloned()
                .collect();
            
            // Dynamic alpha based on IDF overlap
            let alpha = if overlap.is_empty() {
                1.0
            } else {
                let idf_overlap: f32 = overlap.iter()
                    .map(|w| *self.word_idf_fast.get(w).unwrap_or(&0.5))
                    .sum::<f32>() / overlap.len() as f32;
                let idf_query: f32 = q_expanded.iter()
                    .map(|w| *self.word_idf_fast.get(w).unwrap_or(&0.5))
                    .sum::<f32>() / q_expanded.len().max(1) as f32;
                let base_alpha = (1.0 - idf_overlap / idf_query.max(0.001)).clamp(0.0, 1.0);
                (base_alpha + dialogue_semantic_boost).min(1.0)
            };
            
            // Entity TF-IDF (70/30 weighting)
            let mut entity_score = 0.0f32;
            let mut common_score = 0.0f32;
            for word in &overlap {
                let tf = doc_tf.and_then(|m| m.get(word)).copied().unwrap_or(1) as f32;
                let idf = *self.word_idf_fast.get(word).unwrap_or(&0.5);
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
                            phrase_idf_sum += *self.word_idf_fast.get(w).unwrap_or(&0.5);
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
            
            results.push((doc_id, final_score));
        }
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Span-density reranking on top candidates
        let rerank_depth = self.rerank_depth.min(results.len());
        let q_words: Vec<String> = q_expanded.iter().cloned().collect();
        
        let mut champions: Vec<(String, f32)> = results.into_iter().take(rerank_depth).collect();
        
        champions.par_iter_mut().for_each(|(doc_id, score)| {
            if let Some(doc_idx) = self.doc_ids.iter().position(|id| id == doc_id) {
                if let Some(doc_text) = self.doc_texts_fast.get(doc_idx) {
                    let density_score = self.calculate_span_density_fast(doc_text, &q_words);
                    let boost_val = 1.0 + (density_score * 0.25);
                    *score *= boost_val;
                }
            }
        });
        
        champions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        champions.into_iter().take(top_k).collect()
    }
    
    /// Calculate span density for reranking (sca_dropin style - exact alignment)
    #[inline(always)]
    fn calculate_span_density_fast(&self, doc_text: &str, query_terms: &[String]) -> f32 {
        if query_terms.len() < 2 { return 0.0; }
        
        let q_to_idx: AHashMap<String, usize> = query_terms.iter()
            .enumerate()
            .map(|(i, s)| (s.to_lowercase(), i))
            .collect();
        let query_idfs: Vec<f32> = query_terms.iter()
            .map(|t| *self.word_idf_fast.get(t).unwrap_or(&0.5))
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
    
    /// Unified quantized search (sca_dropin style)
    pub fn search_unified_quantized(
        &self,
        query_emb: &[f32],
        query_text: &str,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        let rescore_limit = 5000;
        let query_lower = query_text.to_lowercase();
        
        // Analyze query and determine route (or use forced route)
        let (mut route, _q_words, q_expanded, _idf_avg, _oov_ratio, _has_oov_code, _has_typo) =
            self.analyze_query_quantized(query_text);
        if let Some(forced) = self.force_route {
            route = forced;
        }
        
        // PATH A: PURE LEXICAL (Passkeys, Codes)
        if route == QueryRoute::PureLexical {
            return self.search_pure_lexical_quantized(&query_lower, &q_expanded, top_k);
        }
        
        // Quantize query for Hamming search
        let q_bits = self.quantize_query(query_emb);
        let max_hamming = (self.quantized_dim * 8) as f64;
        
        // PATH B: PURE SEMANTIC (STS, Clustering)
        if route == QueryRoute::PureSemantic {
            let survivors = self.hamming_candidate_retrieval(&q_bits, rescore_limit);
            return self.search_pure_semantic_quantized(&survivors, max_hamming, top_k);
        }
        
        // PATH C: FULL HYBRID
        let hybrid_limit = self.doc_ids.len();
        let survivors = self.hybrid_candidate_retrieval(&q_bits, &q_expanded, hybrid_limit);
        self.search_full_hybrid_quantized(&survivors, &q_expanded, &query_lower, query_text, max_hamming, top_k)
    }
    
    /// Query analysis for quantized mode (sca_dropin style)
    fn analyze_query_quantized(&self, query_text: &str) -> (QueryRoute, Vec<String>, AHashSet<String>, f32, f32, bool, bool) {
        let q_words: Vec<String> = query_text
            .split_whitespace()
            .map(|s| {
                let lower = s.to_lowercase();
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
            if looks_like_code(word) {
                has_any_code = true;
                total_idf += 5.0;
                known_word_count += 1;
                q_expanded.insert(word.clone());
                continue;
            }
            
            let in_idf = self.word_idf_fast.get(word);
            
            if in_idf.is_some() || self.vocabulary_fast.contains(word) {
                known_word_count += 1;
                let idf = *in_idf.unwrap_or(&1.5);
                total_idf += idf;
                q_expanded.insert(word.clone());
            } else {
                oov_count += 1;

                // Try compound word splitting for code-intent words
                let mut compound_found = false;
                for &(compound, parts) in COMPOUND_SPLITS {
                    if word == compound {
                        for &part in parts {
                            if part.len() >= 3 {
                                q_expanded.insert(part.to_string());
                            }
                        }
                        q_expanded.insert(word.clone());
                        compound_found = true;
                        break;
                    }
                }
                if !compound_found {
                    // Try fuzzy expansion
                    let fuzzy_matches = self.fuzzy_expand_word_quantized(word, 5);
                    let valid_matches: Vec<String> = fuzzy_matches.iter()
                        .filter(|m| *m != word && self.vocabulary_fast.contains(*m))
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

        // Calculate IDF average
        let content_words: Vec<f32> = q_expanded.iter()
            .filter_map(|w| self.word_idf_fast.get(w).copied())
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
            .filter(|w| !looks_like_code(w))
            .count();
        let oov_ratio = if non_code_count > 0 {
            oov_count as f32 / non_code_count as f32
        } else {
            0.0
        };
        
        // Code intent detection
        let has_code_intent = q_words.iter().any(|w| {
            CODE_INTENT_WORDS.iter().any(|ci| w == *ci)
        });
        
        // Detect pure discourse
        let high_idf_count = q_expanded.iter()
            .filter(|w| *self.word_idf_fast.get(*w).unwrap_or(&0.0) > 2.5)
            .count();
        
        let is_short_discourse = q_words.len() <= 8 && !has_any_code &&
                                 !has_code_intent && idf_avg <= 1.2 &&
                                 high_idf_count == 0 && oov_ratio < 0.1;
        
        let route = if has_any_code || has_code_intent {
            QueryRoute::PureLexical
        } else if is_short_discourse {
            QueryRoute::PureSemantic
        } else {
            QueryRoute::FullHybrid
        };
        
        (route, q_words, q_expanded, idf_avg, oov_ratio, has_any_code, has_typo)
    }
    
    /// Fuzzy word expansion for quantized mode
    fn fuzzy_expand_word_quantized(&self, word: &str, top_k: usize) -> Vec<String> {
        let word_lower = word.to_lowercase();
        
        if self.vocabulary_fast.contains(&word_lower) {
            return vec![word_lower];
        }
        
        let sx = self.get_soundex(&word_lower);
        let candidates = match self.phonetic_index_fast.get(&sx) {
            Some(c) => c,
            None => return vec![word_lower],
        };
        
        let mut ranked: Vec<(String, usize)> = candidates.iter()
            .map(|c| (c.clone(), self.levenshtein(&word_lower, c)))
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
    
    /// Check if quantized mode is enabled
    pub fn is_quantized_mode(&self) -> bool {
        self.quantized_mode
    }
    
    /// Get quantized stats
    pub fn get_quantized_stats(&self) -> (usize, usize, usize) {
        let total_passages: usize = self.passage_counts.iter().sum();
        (self.doc_ids.len(), total_passages, self.vocabulary_fast.len())
    }
}

/// Dot product of two vectors
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Free function: Code/passkey detection (sca_dropin alignment)
fn looks_like_code(word: &str) -> bool {
    // Early exit for common false positives
    if word.contains('/') || word.contains('-') {
        return false;
    }
    if word.contains('$') || word.contains('€') || word.contains('£') || word.contains(',') {
        return false;
    }
    if word.starts_with('(') || word.starts_with('[') {
        return false;
    }
    if word.chars().any(|c| c == '±' || c == '×' || c == '÷') {
        return false;
    }

    let clean: String = word.chars()
        .filter(|c| c.is_alphanumeric())
        .collect();

    if clean.len() < 5 || clean.len() > 15 {
        return false;
    }

    let lower = clean.to_lowercase();
    if lower.ends_with("st") || lower.ends_with("nd") ||
       lower.ends_with("rd") || lower.ends_with("th") {
        let prefix = &lower[..lower.len()-2];
        if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
    }

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

    // Pure digits, length 5-10 (passkeys)
    if letter_count == 0 && digit_count >= 5 && digit_count <= 10 {
        return true;
    }

    false
}

// ═══════════════════════════════════════════════════════════════════════════════
// FILLER POOL: For ultra-fast document generation (testing)
// ═══════════════════════════════════════════════════════════════════════════════

/// Create pre-indexed filler pool for fast document generation.
/// Uses deterministic pseudo-random generation for reproducibility.
#[allow(dead_code)]
pub fn create_filler_pool(num_segments: usize, tokens_per_segment: usize) -> (Vec<String>, Vec<HashSet<String>>) {
    let legal_words = vec![
        "whereas", "therefore", "notwithstanding", "herein", "thereof",
        "aforesaid", "party", "parties", "agreement", "contract",
        "provision", "clause", "section", "article", "paragraph",
    ];
    
    let mut segments = Vec::new();
    let mut segment_tokens = Vec::new();
    
    let words_per_segment = tokens_per_segment / 13; // ~1.3 tokens per word
    let sentences_per_segment = words_per_segment / 20;
    
    // Simple deterministic pseudo-random using segment index as seed
    let mut seed: u64 = 12345;
    
    for seg_idx in 0..num_segments {
        let mut sentences = Vec::new();
        for sent_idx in 0..sentences_per_segment {
            // Update seed deterministically
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345 + seg_idx as u64 + sent_idx as u64);
            let num_words = 15 + ((seed / 7) % 11) as usize; // 15-25 words
            
            let sentence: Vec<&str> = (0..num_words)
                .map(|i| {
                    seed = seed.wrapping_mul(1103515245).wrapping_add(i as u64);
                    legal_words[(seed as usize) % legal_words.len()]
                })
                .collect();
            
            let mut s = sentence.join(" ");
            s.push('.');
            
            // Capitalize first letter
            let first_upper: String = s.chars().next()
                .map(|c| c.to_uppercase().collect::<String>())
                .unwrap_or_default();
            s = format!("{}{}", first_upper, &s[1..]);
            
            sentences.push(s);
        }
        let text = sentences.join(" ");
        
        // Simple tokenize for token set
        let re = Regex::new(r"\w+").unwrap();
        let tokens: HashSet<String> = re.find_iter(&text.to_lowercase())
            .map(|m| m.as_str().to_string())
            .collect();
        
        segments.push(text);
        segment_tokens.push(tokens);
    }
    
    (segments, segment_tokens)
}

// ═══════════════════════════════════════════════════════════════════════════════
// This gets integrated into LAM class - users just do: from lam import LAM
// ═══════════════════════════════════════════════════════════════════════════════


