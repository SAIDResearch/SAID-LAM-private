//! Memory-mapped file storage for embeddings
//! 
//! This module provides zero-copy, OS-managed storage for document embeddings
//! using memory-mapped files. Supports:
//! - Concurrent read/write access (Search While Indexing)
//! - Matryoshka representation learning (variable dimension reads)
//! - f16/f32 precision
//! - 64-byte aligned SIMD-friendly layout
//! - Atomic live updates without locking

// Many items are intentionally kept for future use or external APIs
#![allow(dead_code)]

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::mem;
use std::path::Path;
use std::slice;

#[cfg(feature = "mmap")]
use std::sync::atomic::{AtomicU64, Ordering};

// Conditional compilation: Use actual memmap2 if available, otherwise provide stubs
#[cfg(feature = "mmap")]
use memmap2::{Mmap, MmapMut, MmapOptions};

#[cfg(feature = "half")]
use half::f16;

// Growth strategy constants (Geometric Doubling with Cap)
// 
// Strategy: Start small (1MB) for new users, double each time until cap (32MB).
// This reduces remaps and is mobile-friendly:
// - Linear 10MB: 100 remaps to reach 1GB
// - Doubling: ~10 remaps to reach 1GB
// - Better user experience: 1.2MB for small app vs 10MB for empty app
const MIN_GROWTH: usize = 1 * 1024 * 1024;  // Start at 1MB (~1,300 docs at 384-dim)
const MAX_GROWTH: usize = 32 * 1024 * 1024;  // Cap growth at 32MB (prevents huge jumps on low-RAM phones)

/// 64-byte aligned header for .said file format
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct SaidHeader {
    pub magic: [u8; 4],        // "SAID"
    pub version: u16,         // Format version (1)
    pub flags: u16,           // Bit flags:
                              //   Bit 0: Interpolation enabled
                              //   Bit 1: MRL-64 mode
                              //   Bit 2: MRL-128 mode
                              //   Bit 3: MRL-256 mode
    pub doc_count: u64,       // Atomic counter (for live updates)
    pub embedding_dim: u16,   // 64/128/256/384
    pub reserved: [u8; 46],    // Padding to 64 bytes
}

// Compile-time assertions
const _: () = assert!(mem::size_of::<SaidHeader>() == 64);
const _: () = assert!(mem::align_of::<SaidHeader>() == 64);

impl SaidHeader {
    pub const MAGIC: [u8; 4] = *b"SAID";
    pub const VERSION: u16 = 1;
    
    pub fn new(embedding_dim: u16, flags: u16) -> Self {
        let header = Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            flags,
            doc_count: 0,
            embedding_dim,
            reserved: [0; 46],
        };
        header
    }
    
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != Self::MAGIC {
            return Err("Invalid magic number".to_string());
        }
        if self.version != Self::VERSION {
            return Err(format!("Unsupported version: {}", self.version));
        }
        Ok(())
    }
}

/// Memory-mapped index for embeddings
/// 
/// Provides zero-copy access to embeddings stored in a .said file.
/// Supports concurrent read/write access via atomic operations.
/// 
/// Note: This is a simplified interface. For production use, prefer
/// `MmapIndexWriter` and `MmapIndexReader` for proper separation of concerns.
pub struct MmapIndex {
    #[cfg(feature = "mmap")]
    file: File,
    #[cfg(feature = "mmap")]
    mmap: MmapMut,
    #[cfg(feature = "mmap")]
    embedding_dim: usize,
    #[cfg(feature = "mmap")]
    vector_size: usize, // embedding_dim * 2 (for f16)
    
    #[cfg(not(feature = "mmap"))]
    _header: SaidHeader,
    #[cfg(not(feature = "mmap"))]
    _embedding_dim: usize,
}

impl MmapIndex {
    /// Create a new .said file with pre-allocated space
    /// 
    /// # Arguments
    /// * `path` - Path to the .said file
    /// * `embedding_dim` - Dimension of embeddings (64/128/256/384)
    /// * `initial_capacity` - Initial number of vectors to pre-allocate
    /// 
    /// # Returns
    /// * `MmapIndex` on success
    pub fn create<P: AsRef<Path>>(
        path: P,
        embedding_dim: u16,
        initial_capacity: usize,
    ) -> io::Result<Self> {
        let path = path.as_ref();
        
        // Create file
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        
        // Calculate initial file size
        // Header: 64 bytes
        // Vectors: initial_capacity * embedding_dim * 2 (f16) or 4 (f32)
        let header_size = mem::size_of::<SaidHeader>();
        let vector_size_per_doc = embedding_dim as usize * 2; // f16 = 2 bytes
        let total_size = header_size + (initial_capacity * vector_size_per_doc);
        
        // Pre-allocate file
        file.set_len(total_size as u64)?;
        
        // Initialize header
        let header = SaidHeader::new(embedding_dim, 0);
        
        // Write header to file
        let header_bytes = unsafe {
            slice::from_raw_parts(
                &header as *const _ as *const u8,
                mem::size_of::<SaidHeader>(),
            )
        };
        file.write_all(header_bytes)?;
        file.sync_all()?;
        
        // In actual implementation:
        // let mmap = unsafe { MmapMut::map_mut(&file)? };
        // let header_ptr = mmap.as_ptr() as *const SaidHeader;
        
        Ok(Self {
            _header: header,
            _embedding_dim: embedding_dim as usize,
        })
    }
    
    /// Open an existing .said file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref();
        let _file = File::open(path)?;
        
        // In actual implementation:
        // let mmap = unsafe { Mmap::map(&file)? };
        // let header_ptr = mmap.as_ptr() as *const SaidHeader;
        // let header = unsafe { &*header_ptr };
        // header.validate()?;
        
        // For demonstration:
        let header = SaidHeader::new(384, 0);
        
        Ok(Self {
            _header: header,
            _embedding_dim: 384,
        })
    }
    
    /// Append a vector to the index
    /// 
    /// # Arguments
    /// * `embedding` - f32 embedding vector
    /// 
    /// # Returns
    /// * Vector index on success
    pub fn append_vector(&mut self, _embedding: &[f32]) -> io::Result<usize> {
        // In actual implementation:
        // 1. Check if file needs to grow
        // 2. Convert f32 → f16
        // 3. Write to mmap at current offset
        // 4. Atomically increment doc_count in header
        // 5. Return vector index
        
        // For demonstration:
        let _idx = self._header.doc_count as usize;
        Ok(_idx)
    }
    
    /// Get a vector by index (zero-copy)
    /// 
    /// # Arguments
    /// * `idx` - Vector index
    /// 
    /// # Returns
    /// * Slice of f16 values (zero-copy)
    pub fn get_vector(&self, _idx: usize) -> Option<&[u16]> {
        // In actual implementation:
        // let offset = mem::size_of::<SaidHeader>() + (idx * self._embedding_dim * 2);
        // let ptr = unsafe { self.mmap.as_ptr().add(offset) as *const f16 };
        // Some(unsafe { slice::from_raw_parts(ptr, self._embedding_dim) })
        
        // For demonstration:
        None
    }
    
    /// Get a vector slice for Matryoshka (variable dimensions)
    /// 
    /// # Arguments
    /// * `idx` - Vector index
    /// * `dims` - Number of dimensions to return (64/128/256/384)
    /// 
    /// # Returns
    /// * Slice of f16 values (zero-copy)
    pub fn get_vector_slice(&self, idx: usize, dims: usize) -> Option<&[u16]> {
        let full = self.get_vector(idx)?;
        if dims <= full.len() {
            Some(&full[..dims])
        } else {
            None
        }
    }
    
    /// Get the current document count (atomic read)
    /// 
    /// Uses atomic operations to safely read the count without locking.
    /// Acquire ordering ensures we see all data written before the count was incremented.
    pub fn doc_count(&self) -> u64 {
        #[cfg(feature = "mmap")]
        {
            unsafe {
                // Get pointer to the doc_count field in the mmapped header
                let header_ptr = self.mmap.as_ptr() as *const SaidHeader;
                let count_ptr = &((*header_ptr).doc_count) as *const u64 as *const AtomicU64;
                
                // Atomic Load with Acquire ordering
                // This ensures we see all data written before this count was incremented
                (*count_ptr).load(Ordering::Acquire)
            }
        }
        
        #[cfg(not(feature = "mmap"))]
        {
            self._header.doc_count
        }
    }
}

/// Writer for concurrent append operations
/// 
/// Handles file growth automatically by remapping when capacity is exceeded.
/// Uses atomic operations for thread-safe doc_count updates.
pub struct MmapIndexWriter {
    #[cfg(feature = "mmap")]
    file: File,
    #[cfg(feature = "mmap")]
    mmap: MmapMut,
    #[cfg(feature = "mmap")]
    embedding_dim: usize,
    #[cfg(feature = "mmap")]
    vector_size: usize, // embedding_dim * 2 (for f16)
    #[cfg(feature = "mmap")]
    current_count: usize,
    
    #[cfg(not(feature = "mmap"))]
    _phantom: (),
}

impl MmapIndexWriter {
    /// Calculate new file size using geometric doubling strategy
    /// 
    /// Strategy:
    /// 1. Try to double the current size
    /// 2. Clamp between MIN_GROWTH and MAX_GROWTH
    /// 3. Ensure it's enough for the immediate write
    /// 
    /// This reduces remaps and is mobile-friendly (fewer kernel calls).
    #[cfg(feature = "mmap")]
    fn calculate_new_size(&self, needed: usize) -> usize {
        let current_len = self.mmap.len();
        
        // 1. Calculate how much we need to add (try to double)
        let mut grow_by = current_len; // Doubling strategy
        
        // 2. Clamp the growth between MIN and MAX
        if grow_by < MIN_GROWTH {
            grow_by = MIN_GROWTH;
        }
        if grow_by > MAX_GROWTH {
            grow_by = MAX_GROWTH;
        }
        
        // 3. Ensure it's enough for the immediate write
        // (Rare edge case: if a single write is huge)
        let total_needed = needed.saturating_sub(current_len);
        if grow_by < total_needed {
            // Round up to next power of two
            grow_by = total_needed.next_power_of_two();
            // But still cap at MAX_GROWTH
            if grow_by > MAX_GROWTH {
                grow_by = MAX_GROWTH;
            }
        }
        
        current_len + grow_by
    }
    
    /// Create a writer for appending vectors
    /// 
    /// Opens existing file or creates new one with pre-allocated space.
    pub fn open<P: AsRef<Path>>(path: P, embedding_dim: u16, initial_capacity: usize) -> io::Result<Self> {
        #[cfg(feature = "mmap")]
        {
            let path = path.as_ref();
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(path)?;
            
            // Calculate sizes
            let header_size = mem::size_of::<SaidHeader>();
            let vector_size = embedding_dim as usize * 2; // f16 = 2 bytes
            let total_size = header_size + (initial_capacity * vector_size);
            
            // Pre-allocate file
            file.set_len(total_size as u64)?;
            
            // Map the file
            let mut mmap = unsafe { MmapMut::map_mut(&file)? };
            
            // Initialize header if new file
            if file.metadata()?.len() == total_size as u64 {
                let header = SaidHeader::new(embedding_dim, 0);
                let header_bytes = unsafe {
                    slice::from_raw_parts(
                        &header as *const _ as *const u8,
                        mem::size_of::<SaidHeader>(),
                    )
                };
                mmap[..header_size].copy_from_slice(header_bytes);
            }
            
            // Get current count from header
            let header_ptr = mmap.as_ptr() as *const SaidHeader;
            let count_ptr = unsafe { &((*header_ptr).doc_count) as *const u64 as *const AtomicU64 };
            let current_count = unsafe { (*count_ptr).load(Ordering::Acquire) } as usize;
            
            Ok(Self {
                file,
                mmap,
                embedding_dim: embedding_dim as usize,
                vector_size,
                current_count,
            })
        }
        
        #[cfg(not(feature = "mmap"))]
        {
            let _ = (path, embedding_dim, initial_capacity);
            Ok(Self { _phantom: () })
        }
    }
    
    /// Append a vector (thread-safe with automatic growth)
    /// 
    /// Handles file growth by remapping when capacity is exceeded.
    /// Uses atomic operations for thread-safe doc_count updates.
    pub fn append(&mut self, embedding: &[f32]) -> io::Result<usize> {
        #[cfg(feature = "mmap")]
        {
            if embedding.len() != self.embedding_dim {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Embedding dimension mismatch: expected {}, got {}", self.embedding_dim, embedding.len()),
                ));
            }
            
            // Calculate needed size
            let header_size = mem::size_of::<SaidHeader>();
            let needed_size = header_size + ((self.current_count + 1) * self.vector_size);
            
            // Check if we need to grow the file
            if needed_size > self.mmap.len() {
                // Use geometric doubling strategy (start small, double until cap)
                // This reduces remaps and is mobile-friendly
                let new_len = self.calculate_new_size(needed_size);
                self.file.set_len(new_len as u64)?;
                
                // CRITICAL: Remap after file growth
                // Drop old mmap, create new one
                drop(self.mmap);
                self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
            }
            
            // Calculate write offset
            let header_size = mem::size_of::<SaidHeader>();
            let offset = header_size + (self.current_count * self.vector_size);
            
            // Convert f32 → f16 and write
            #[cfg(feature = "half")]
            {
                let f16_vec: Vec<f16> = embedding.iter().map(|&x| f16::from_f32(x)).collect();
                let bytes = unsafe {
                    slice::from_raw_parts(
                        f16_vec.as_ptr() as *const u8,
                        f16_vec.len() * 2,
                    )
                };
                self.mmap[offset..offset + self.vector_size].copy_from_slice(bytes);
            }
            
            #[cfg(not(feature = "half"))]
            {
                // Fallback: truncate f32 to u16 (simple quantization)
                // This is NOT ideal but works if half crate is unavailable
                for (i, &val) in embedding.iter().enumerate() {
                    let quantized = (val * 32767.0).clamp(-32768.0, 32767.0) as i16;
                    let bytes = quantized.to_le_bytes();
                    self.mmap[offset + (i * 2)..offset + (i * 2) + 2].copy_from_slice(&bytes);
                }
            }
            
            // Atomically increment doc_count (Release ordering ensures all data is visible)
            // 
            // Safety: Casting &u64 to &AtomicU64 is safe because:
            // 1. SaidHeader uses #[repr(align(64))], ensuring 64-byte alignment
            // 2. 64 is divisible by 8, so doc_count (u64) is 8-byte aligned
            // 3. AtomicU64 requires 8-byte alignment, which is satisfied
            // 4. Verified safe on x86_64 and Aarch64 (Apple Silicon/Android)
            let header_ptr = self.mmap.as_ptr() as *const SaidHeader;
            let count_ptr = unsafe { &((*header_ptr).doc_count) as *const u64 as *mut AtomicU64 };
            let new_count = (self.current_count + 1) as u64;
            unsafe {
                (*count_ptr).store(new_count, Ordering::Release);
            }
            
            let idx = self.current_count;
            self.current_count += 1;
            
            Ok(idx)
        }
        
        #[cfg(not(feature = "mmap"))]
        {
            let _ = embedding;
            Ok(0)
        }
    }
    
    /// Sync changes to disk
    /// 
    /// Forces OS to write dirty pages to disk.
    /// Usually not needed (OS handles this), but useful for durability guarantees.
    pub fn sync(&self) -> io::Result<()> {
        #[cfg(feature = "mmap")]
        {
            self.mmap.flush()?;
            self.file.sync_all()?;
        }
        Ok(())
    }
    
    /// Get current document count (atomic read)
    pub fn doc_count(&self) -> u64 {
        #[cfg(feature = "mmap")]
        {
            unsafe {
                let header_ptr = self.mmap.as_ptr() as *const SaidHeader;
                let count_ptr = &((*header_ptr).doc_count) as *const u64 as *const AtomicU64;
                (*count_ptr).load(Ordering::Acquire)
            }
        }
        
        #[cfg(not(feature = "mmap"))]
        {
            0
        }
    }
}

/// Reader for concurrent query operations
/// 
/// Provides zero-copy read access to embeddings.
/// Supports live updates by checking file size and remapping when needed.
pub struct MmapIndexReader {
    #[cfg(feature = "mmap")]
    file: File,
    #[cfg(feature = "mmap")]
    mmap: Mmap,
    #[cfg(feature = "mmap")]
    embedding_dim: usize,
    #[cfg(feature = "mmap")]
    vector_size: usize,
    #[cfg(feature = "mmap")]
    cached_doc_count: u64,
    #[cfg(feature = "mmap")]
    file_path: std::path::PathBuf,
    
    #[cfg(not(feature = "mmap"))]
    _phantom: (),
}

impl MmapIndexReader {
    /// Open a reader for querying
    /// 
    /// Opens file in read-only mode and maps it for zero-copy access.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        #[cfg(feature = "mmap")]
        {
            let path = path.as_ref();
            let file = File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            
            // Validate header
            let header_ptr = mmap.as_ptr() as *const SaidHeader;
            let header = unsafe { &*header_ptr };
            header.validate().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            
            let embedding_dim = header.embedding_dim as usize;
            let vector_size = embedding_dim * 2; // f16 = 2 bytes
            
            // Get current doc_count atomically
            let count_ptr = unsafe { &(header.doc_count) as *const u64 as *const AtomicU64 };
            let cached_doc_count = unsafe { (*count_ptr).load(Ordering::Acquire) };
            
            Ok(Self {
                file,
                mmap,
                embedding_dim,
                vector_size,
                cached_doc_count,
                file_path: path.to_path_buf(),
            })
        }
        
        #[cfg(not(feature = "mmap"))]
        {
            let _ = path;
            Ok(Self { _phantom: () })
        }
    }
    
    /// Check if index has been updated (for live updates)
    /// 
    /// Compares current doc_count with cached value.
    /// Returns true if the file has been modified.
    pub fn check_update(&mut self) -> io::Result<bool> {
        #[cfg(feature = "mmap")]
        {
            // Check file size first (cheaper than atomic read)
            let current_size = self.file.metadata()?.len() as usize;
            if current_size > self.mmap.len() {
                // File has grown, need to remap
                return Ok(true);
            }
            
            // Atomic read of doc_count
            let header_ptr = self.mmap.as_ptr() as *const SaidHeader;
            let count_ptr = unsafe { &((*header_ptr).doc_count) as *const u64 as *const AtomicU64 };
            let current_count = unsafe { (*count_ptr).load(Ordering::Acquire) };
            
            if current_count != self.cached_doc_count {
                self.cached_doc_count = current_count;
                Ok(true)
            } else {
                Ok(false)
            }
        }
        
        #[cfg(not(feature = "mmap"))]
        {
            Ok(false)
        }
    }
    
    /// Re-map the file if it has grown (for live updates)
    /// 
    /// Drops old mapping and creates new one with updated file size.
    /// This is necessary because mmap cannot see file growth automatically.
    pub fn remap(&mut self) -> io::Result<()> {
        #[cfg(feature = "mmap")]
        {
            // Drop old mmap
            drop(self.mmap);
            
            // Re-open file to get updated size
            self.file = File::open(&self.file_path)?;
            
            // Re-map with new size
            self.mmap = unsafe { Mmap::map(&self.file)? };
            
            // Update cached doc_count
            let header_ptr = self.mmap.as_ptr() as *const SaidHeader;
            let count_ptr = unsafe { &((*header_ptr).doc_count) as *const u64 as *const AtomicU64 };
            self.cached_doc_count = unsafe { (*count_ptr).load(Ordering::Acquire) };
            
            Ok(())
        }
        
        #[cfg(not(feature = "mmap"))]
        {
            Ok(())
        }
    }
    
    /// Get a vector by index (zero-copy)
    /// 
    /// Returns a slice of f16 values without copying memory.
    /// The slice is aligned and ready for SIMD operations.
    pub fn get_vector(&self, idx: usize) -> Option<&[u16]> {
        #[cfg(feature = "mmap")]
        {
            // Check bounds
            let header_size = mem::size_of::<SaidHeader>();
            let offset = header_size + (idx * self.vector_size);
            let end_offset = offset + self.vector_size;
            
            if end_offset > self.mmap.len() {
                return None;
            }
            
            // Zero-copy pointer arithmetic
            // Since vectors start at 64-byte aligned header, they're also aligned
            unsafe {
                let ptr = self.mmap.as_ptr().add(offset) as *const u16;
                Some(slice::from_raw_parts(ptr, self.embedding_dim))
            }
        }
        
        #[cfg(not(feature = "mmap"))]
        {
            let _ = idx;
            None
        }
    }
    
    /// Get a vector slice for Matryoshka (variable dimensions)
    /// 
    /// Returns a zero-copy slice of the first `dims` dimensions.
    /// Perfectly aligned for SIMD operations (vld1.16 on ARM NEON).
    pub fn get_vector_slice(&self, idx: usize, dims: usize) -> Option<&[u16]> {
        let full = self.get_vector(idx)?;
        if dims <= full.len() {
            Some(&full[..dims])
        } else {
            None
        }
    }
    
    /// Get current document count (atomic read)
    pub fn doc_count(&self) -> u64 {
        #[cfg(feature = "mmap")]
        {
            unsafe {
                let header_ptr = self.mmap.as_ptr() as *const SaidHeader;
                let count_ptr = &((*header_ptr).doc_count) as *const u64 as *const AtomicU64;
                (*count_ptr).load(Ordering::Acquire)
            }
        }
        
        #[cfg(not(feature = "mmap"))]
        {
            0
        }
    }
    
    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        #[cfg(feature = "mmap")]
        {
            self.embedding_dim
        }
        
        #[cfg(not(feature = "mmap"))]
        {
            384
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEXT STORAGE - Replaces HashMap<String, String> for doc_texts
// ═══════════════════════════════════════════════════════════════════════════════
//
// RAM Impact: ~12 bytes per document (offset + length) instead of full text
// Strategy: Write text to disk, keep only pointers in RAM, use mmap for reads
//
// ═══════════════════════════════════════════════════════════════════════════════

use std::io::{Read, Seek, SeekFrom, BufReader};
use std::sync::RwLock;

/// Text pointer entry: (file_offset, byte_length)
/// Only 12 bytes per document in RAM
pub type TextPointer = (u64, u32);

/// Memory-efficient text storage using mmap
/// 
/// Replaces `HashMap<String, String>` with:
/// - Append-only file for text data
/// - In-memory pointer table (`Vec<(u64, u32)>`) for O(1) access
/// - Zero-copy reads via mmap
/// 
/// RAM usage: ~12 bytes per document (vs 1GB+ for HashMap with large texts)
pub struct MmapTextStore {
    /// File for writing text (append-only)
    file: RwLock<File>,
    
    /// Memory-mapped view for zero-copy reads
    #[cfg(feature = "mmap")]
    mmap: RwLock<Option<Mmap>>,
    
    /// Pointer table: doc_id (u64 index) → (offset, length)
    /// Maps String doc_id to u64 via external registry
    pointers: RwLock<Vec<TextPointer>>,
    
    /// Total bytes written to file
    bytes_written: RwLock<u64>,
    
    /// File path for re-opening
    file_path: std::path::PathBuf,
}

impl MmapTextStore {
    /// Create a new text store
    /// 
    /// # Arguments
    /// * `path` - Path to the .texts file
    /// 
    /// # Returns
    /// * `MmapTextStore` on success
    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref();
        
        // Create/truncate file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        
        Ok(Self {
            file: RwLock::new(file),
            #[cfg(feature = "mmap")]
            mmap: RwLock::new(None),
            pointers: RwLock::new(Vec::new()),
            bytes_written: RwLock::new(0),
            file_path: path.to_path_buf(),
        })
    }
    
    /// Open an existing text store
    /// 
    /// Note: Requires loading pointer table from index file or rebuilding
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref();
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .append(true)
            .open(path)?;
        
        let bytes_written = file.metadata()?.len();
        
        // Map file for reading
        #[cfg(feature = "mmap")]
        let mmap = if bytes_written > 0 {
            Some(unsafe { Mmap::map(&file)? })
        } else {
            None
        };
        
        Ok(Self {
            file: RwLock::new(file),
            #[cfg(feature = "mmap")]
            mmap: RwLock::new(mmap),
            pointers: RwLock::new(Vec::new()),
            bytes_written: RwLock::new(bytes_written),
            file_path: path.to_path_buf(),
        })
    }
    
    /// Append text to store
    /// 
    /// # Arguments
    /// * `text` - Text to store
    /// 
    /// # Returns
    /// * Internal document index (u64)
    /// 
    /// # RAM Impact
    /// * Only adds 12 bytes (offset + length) to memory
    /// * Text is written to disk immediately
    pub fn append_text(&self, text: &str) -> io::Result<u64> {
        let bytes = text.as_bytes();
        let length = bytes.len() as u32;
        
        // Get current offset and write
        let mut bytes_written = self.bytes_written.write().unwrap();
        let offset = *bytes_written;
        
        {
            let mut file = self.file.write().unwrap();
            file.write_all(bytes)?;
            file.sync_data()?; // Ensure data is on disk
        }
        
        // Update total bytes
        *bytes_written += length as u64;
        
        // Add pointer to table
        let mut pointers = self.pointers.write().unwrap();
        let doc_idx = pointers.len() as u64;
        pointers.push((offset, length));
        
        // Invalidate mmap (will be rebuilt on next read)
        #[cfg(feature = "mmap")]
        {
            let mut mmap = self.mmap.write().unwrap();
            *mmap = None;
        }
        
        Ok(doc_idx)
    }
    
    /// Get text by internal index (zero-copy when mmap available)
    /// 
    /// # Arguments
    /// * `idx` - Internal document index
    /// 
    /// # Returns
    /// * Text content as String
    pub fn get_text(&self, idx: u64) -> io::Result<String> {
        let pointers = self.pointers.read().unwrap();
        
        if idx as usize >= pointers.len() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Document index {} not found", idx),
            ));
        }
        
        let (offset, length) = pointers[idx as usize];
        
        #[cfg(feature = "mmap")]
        {
            // Ensure mmap is valid
            let mut mmap_guard = self.mmap.write().unwrap();
            if mmap_guard.is_none() {
                let file = self.file.read().unwrap();
                *mmap_guard = Some(unsafe { Mmap::map(&*file)? });
            }
            
            if let Some(ref mmap) = *mmap_guard {
                let end = offset as usize + length as usize;
                if end <= mmap.len() {
                    let bytes = &mmap[offset as usize..end];
                    return Ok(String::from_utf8_lossy(bytes).to_string());
                }
            }
        }
        
        // Fallback: Read from file directly
        let file = self.file.read().unwrap();
        let mut reader = BufReader::new(&*file);
        reader.seek(SeekFrom::Start(offset))?;
        
        let mut buffer = vec![0u8; length as usize];
        reader.read_exact(&mut buffer)?;
        
        Ok(String::from_utf8_lossy(&buffer).to_string())
    }
    
    /// Get number of stored documents
    pub fn doc_count(&self) -> usize {
        self.pointers.read().unwrap().len()
    }
    
    /// Get total bytes stored
    pub fn total_bytes(&self) -> u64 {
        *self.bytes_written.read().unwrap()
    }
    
    /// Load pointer table from serialized data
    /// 
    /// Used when opening an existing store with a separate index file
    pub fn load_pointers(&self, pointers: Vec<TextPointer>) {
        let mut ptr_guard = self.pointers.write().unwrap();
        *ptr_guard = pointers;
    }
    
    /// Export pointer table for serialization
    /// 
    /// Used when saving index to disk
    pub fn export_pointers(&self) -> Vec<TextPointer> {
        self.pointers.read().unwrap().clone()
    }
    
    /// Sync all data to disk
    pub fn sync(&self) -> io::Result<()> {
        let file = self.file.read().unwrap();
        file.sync_all()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// IN-MEMORY TEXT STORE (Fallback when mmap feature disabled)
// ═══════════════════════════════════════════════════════════════════════════════

/// In-memory text store that matches MmapTextStore API
/// 
/// Used when mmap feature is disabled or for testing.
/// Uses simple Vec<String> storage - efficient and Clone-able.
/// 
/// RAM Impact: Same as HashMap<String, String> for now, but:
/// - Provides unified API for future mmap migration
/// - Uses u64 indexes instead of String keys (faster lookups)
#[derive(Clone, Default)]
pub struct InMemoryTextStore {
    texts: Vec<String>,
}

impl InMemoryTextStore {
    pub fn new() -> Self {
        Self {
            texts: Vec::new(),
        }
    }
    
    /// Append text and return its index
    /// Takes &mut self since CrystallineCore uses &mut self for writes
    pub fn append_text(&mut self, text: &str) -> io::Result<u64> {
        let idx = self.texts.len() as u64;
        self.texts.push(text.to_string());
        Ok(idx)
    }
    
    /// Get text by index (immutable borrow)
    pub fn get_text(&self, idx: u64) -> io::Result<String> {
        self.texts.get(idx as usize)
            .cloned()
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::NotFound,
                format!("Document index {} not found", idx),
            ))
    }
    
    /// Get text reference by index (zero-copy read)
    pub fn get_text_ref(&self, idx: u64) -> Option<&str> {
        self.texts.get(idx as usize).map(|s| s.as_str())
    }
    
    /// Number of stored documents
    pub fn doc_count(&self) -> usize {
        self.texts.len()
    }
    
    /// Total bytes stored
    pub fn total_bytes(&self) -> u64 {
        self.texts.iter().map(|s| s.len() as u64).sum()
    }
    
    /// Clear all stored texts
    pub fn clear(&mut self) {
        self.texts.clear();
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED TEXT STORAGE - Abstraction for Clone-able storage
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "mmap")]
use std::sync::Arc;

/// Unified text storage that supports both in-memory and persistent modes
/// 
/// This enum provides a Clone-able interface to text storage:
/// - `InMemory`: Simple Vec<String> storage (Clone via derive)
/// - `Persistent`: Arc-wrapped MmapTextStore (Clone via Arc)
#[derive(Clone)]
pub enum TextStorage {
    /// In-memory storage for ephemeral/testing use
    InMemory(InMemoryTextStore),
    
    /// Persistent mmap-backed storage wrapped in Arc for Clone support
    #[cfg(feature = "mmap")]
    Persistent(Arc<MmapTextStore>),
}

impl TextStorage {
    /// Create ephemeral in-memory storage
    pub fn new_ephemeral() -> Self {
        TextStorage::InMemory(InMemoryTextStore::new())
    }
    
    /// Create persistent mmap-backed storage
    #[cfg(feature = "mmap")]
    pub fn new_persistent<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Ok(TextStorage::Persistent(Arc::new(MmapTextStore::create(path)?)))
    }
    
    /// Open existing persistent storage
    #[cfg(feature = "mmap")]
    pub fn open_persistent<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Ok(TextStorage::Persistent(Arc::new(MmapTextStore::open(path)?)))
    }
    
    /// Append text and return index
    pub fn append_text(&mut self, text: &str) -> io::Result<u64> {
        match self {
            TextStorage::InMemory(store) => store.append_text(text),
            #[cfg(feature = "mmap")]
            TextStorage::Persistent(store) => store.append_text(text),
        }
    }
    
    /// Get text by index
    pub fn get_text(&self, idx: u64) -> io::Result<String> {
        match self {
            TextStorage::InMemory(store) => store.get_text(idx),
            #[cfg(feature = "mmap")]
            TextStorage::Persistent(store) => store.get_text(idx),
        }
    }
    
    /// Get document count
    pub fn doc_count(&self) -> usize {
        match self {
            TextStorage::InMemory(store) => store.doc_count(),
            #[cfg(feature = "mmap")]
            TextStorage::Persistent(store) => store.doc_count(),
        }
    }
    
    /// Get total bytes stored
    pub fn total_bytes(&self) -> u64 {
        match self {
            TextStorage::InMemory(store) => store.total_bytes(),
            #[cfg(feature = "mmap")]
            TextStorage::Persistent(store) => store.total_bytes(),
        }
    }
    
    /// Clear storage (only works for in-memory)
    pub fn clear(&mut self) {
        match self {
            TextStorage::InMemory(store) => store.clear(),
            #[cfg(feature = "mmap")]
            TextStorage::Persistent(_) => {
                // Can't clear persistent storage - would need to truncate file
                eprintln!("Warning: clear() called on persistent storage - not supported");
            }
        }
    }
    
    /// Check if using persistent storage
    #[allow(dead_code)]
    pub fn is_persistent(&self) -> bool {
        match self {
            TextStorage::InMemory(_) => false,
            #[cfg(feature = "mmap")]
            TextStorage::Persistent(_) => true,
        }
    }
}

impl Default for TextStorage {
    fn default() -> Self {
        Self::new_ephemeral()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_header_alignment() {
        assert_eq!(mem::size_of::<SaidHeader>(), 64);
        assert_eq!(mem::align_of::<SaidHeader>(), 64);
    }
    
    #[test]
    fn test_header_validation() {
        let header = SaidHeader::new(384, 0);
        assert!(header.validate().is_ok());
    }
    
    #[test]
    fn test_create_index() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_index.said");
        
        // Clean up if exists
        let _ = fs::remove_file(&path);
        
        let index = MmapIndex::create(&path, 384, 1000).unwrap();
        assert_eq!(index.doc_count(), 0);
        
        // Clean up
        let _ = fs::remove_file(&path);
    }
    
    #[test]
    fn test_in_memory_text_store() {
        let mut store = InMemoryTextStore::new();
        
        // Append texts
        let idx0 = store.append_text("Hello World").unwrap();
        let idx1 = store.append_text("Rust is awesome").unwrap();
        let idx2 = store.append_text("Memory efficiency!").unwrap();
        
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
        
        // Retrieve texts
        assert_eq!(store.get_text(0).unwrap(), "Hello World");
        assert_eq!(store.get_text(1).unwrap(), "Rust is awesome");
        assert_eq!(store.get_text(2).unwrap(), "Memory efficiency!");
        
        // Test zero-copy reference
        assert_eq!(store.get_text_ref(0), Some("Hello World"));
        assert_eq!(store.get_text_ref(1), Some("Rust is awesome"));
        
        // Check count
        assert_eq!(store.doc_count(), 3);
        
        // Test clone
        let store2 = store.clone();
        assert_eq!(store2.doc_count(), 3);
        assert_eq!(store2.get_text(0).unwrap(), "Hello World");
    }
    
    #[test]
    fn test_in_memory_text_store_clear() {
        let mut store = InMemoryTextStore::new();
        
        store.append_text("Doc 1").unwrap();
        store.append_text("Doc 2").unwrap();
        assert_eq!(store.doc_count(), 2);
        
        store.clear();
        assert_eq!(store.doc_count(), 0);
        assert!(store.is_empty());
    }
    
    #[cfg(feature = "mmap")]
    #[test]
    fn test_mmap_text_store() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_texts.bin");
        
        // Clean up if exists
        let _ = fs::remove_file(&path);
        
        // Create store
        let store = MmapTextStore::create(&path).unwrap();
        
        // Append texts
        let idx0 = store.append_text("Hello mmap World").unwrap();
        let idx1 = store.append_text("Zero-copy is fast").unwrap();
        let idx2 = store.append_text("Memory efficient!").unwrap();
        
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
        
        // Retrieve texts
        assert_eq!(store.get_text(0).unwrap(), "Hello mmap World");
        assert_eq!(store.get_text(1).unwrap(), "Zero-copy is fast");
        assert_eq!(store.get_text(2).unwrap(), "Memory efficient!");
        
        // Check count
        assert_eq!(store.doc_count(), 3);
        
        // Sync to disk
        store.sync().unwrap();
        
        // Clean up
        let _ = fs::remove_file(&path);
    }
}

