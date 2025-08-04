//! # RustMat Snapshot Creator
//!
//! High-performance snapshot system for preloading the RustMat standard library.
//! Inspired by V8's snapshot architecture, this provides:
//!
//! - **Zero-copy serialization** with memory mapping
//! - **Multi-tier compression** with LZ4 and ZSTD
//! - **Integrity validation** with SHA-256 checksums  
//! - **Concurrent loading** with lock-free data structures
//! - **Progressive enhancement** with fallback mechanisms
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │  Standard Lib   │ -> │   Snapshot       │ -> │   Runtime       │
//! │  Components     │    │   Generator      │    │   Loader        │
//! │                 │    │                  │    │                 │
//! │ • Builtins      │    │ • Serialization  │    │ • Memory Map    │
//! │ • HIR Cache     │    │ • Compression    │    │ • Validation    │
//! │ • Bytecode      │    │ • Validation     │    │ • Integration   │
//! │ • GC Presets    │    │ • Optimization   │    │ • Performance   │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;


use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

pub mod builder;
pub mod compression;
pub mod format;
pub mod loader;
pub mod presets;
pub mod validation;

pub use builder::SnapshotBuilder;
pub use format::{SnapshotFormat, SnapshotHeader, SnapshotMetadata};
pub use loader::SnapshotLoader;

/// Core snapshot data containing preloaded standard library components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Snapshot metadata
    pub metadata: SnapshotMetadata,
    
    /// Preloaded builtin functions with optimized dispatch table
    pub builtins: BuiltinRegistry,
    
    /// Cached HIR representations of standard library functions
    pub hir_cache: HirCache,
    
    /// Precompiled bytecode for common operations
    pub bytecode_cache: BytecodeCache,
    
    /// GC configuration presets
    pub gc_presets: GcPresetCache,
    
    /// Runtime optimization hints
    pub optimization_hints: OptimizationHints,
}

/// Optimized builtin function registry for fast dispatch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinRegistry {
    /// Function name to index mapping for O(1) lookup
    pub name_index: HashMap<String, usize>,
    
    /// Function metadata array (aligned for cache efficiency)
    pub functions: Vec<BuiltinMetadata>,
    
    /// Function dispatch table (runtime-generated)
    #[serde(skip)]
    pub dispatch_table: Arc<RwLock<Vec<rustmat_builtins::BuiltinFn>>>,
}

/// Metadata for a builtin function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinMetadata {
    pub name: String,
    pub arity: BuiltinArity,
    pub category: BuiltinCategory,
    pub complexity: ComputationalComplexity,
    pub optimization_level: OptimizationLevel,
}

/// Function arity specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BuiltinArity {
    /// Exact number of arguments
    Exact(usize),
    /// Range of arguments (min, max)
    Range(usize, usize),
    /// Variadic (minimum arguments)
    Variadic(usize),
}

/// Builtin function categories for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BuiltinCategory {
    Math,
    LinearAlgebra,
    Statistics,
    MatrixOps,
    Trigonometric,
    Comparison,
    Utility,
}

/// Computational complexity for scheduling hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputationalComplexity {
    Constant,
    Linear,
    Quadratic,
    Cubic,
    Exponential,
}

/// Optimization level for JIT compilation hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    MaxPerformance,
}

/// Cached HIR representations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirCache {
    /// Standard library function HIR
    pub functions: HashMap<String, rustmat_hir::HirProgram>,
    
    /// Common expression patterns
    pub patterns: Vec<HirPattern>,
    
    /// Type inference cache
    pub type_cache: HashMap<String, rustmat_hir::Type>,
}

/// HIR pattern for common expressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirPattern {
    pub name: String,
    pub pattern: rustmat_hir::HirProgram,
    pub frequency: u32,
    pub optimization_priority: OptimizationLevel,
}

/// Precompiled bytecode cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BytecodeCache {
    /// Standard library bytecode
    pub stdlib_bytecode: HashMap<String, rustmat_ignition::Bytecode>,
    
    /// Common operation bytecode sequences
    pub operation_sequences: Vec<BytecodeSequence>,
    
    /// Hotspot bytecode (frequently executed)
    pub hotspots: Vec<HotspotBytecode>,
}

/// Bytecode sequence for common operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BytecodeSequence {
    pub name: String,
    pub bytecode: rustmat_ignition::Bytecode,
    pub usage_count: u64,
    pub average_execution_time: Duration,
}

/// Hotspot bytecode with JIT compilation hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotspotBytecode {
    pub name: String,
    pub bytecode: rustmat_ignition::Bytecode,
    pub execution_frequency: u64,
    pub jit_compilation_threshold: u32,
    pub optimization_hints: Vec<OptimizationHint>,
}

/// GC configuration presets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcPresetCache {
    /// Named GC configurations
    pub presets: HashMap<String, rustmat_gc::GcConfig>,
    
    /// Default preset name
    pub default_preset: String,
    
    /// Performance characteristics for each preset
    pub performance_profiles: HashMap<String, GcPerformanceProfile>,
}

/// GC performance profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcPerformanceProfile {
    pub average_allocation_rate: f64,
    pub average_collection_time: Duration,
    pub memory_overhead: f64,
    pub throughput_impact: f64,
}

/// Runtime optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHints {
    /// JIT compilation hints
    pub jit_hints: Vec<JitHint>,
    
    /// Memory layout hints
    pub memory_hints: Vec<MemoryHint>,
    
    /// Execution pattern hints
    pub execution_hints: Vec<ExecutionHint>,
}

/// JIT compilation hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitHint {
    pub pattern: String,
    pub hint_type: JitHintType,
    pub priority: OptimizationLevel,
    pub expected_performance_gain: f64,
}

/// Types of JIT hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JitHintType {
    InlineCandidate,
    LoopOptimization,
    VectorizeCandidate,
    ConstantFolding,
    DeadCodeElimination,
}

/// Memory layout optimization hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHint {
    pub data_structure: String,
    pub hint_type: MemoryHintType,
    pub alignment: usize,
    pub prefetch_pattern: PrefetchPattern,
}

/// Types of memory hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryHintType {
    CacheLocalityOptimization,
    PrefetchOptimization,
    AlignmentOptimization,
    CompressionCandidate,
}

/// Memory prefetch patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrefetchPattern {
    Sequential,
    Random,
    Strided(usize),
    Hierarchical,
}

/// Execution pattern hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionHint {
    pub pattern: String,
    pub hint_type: ExecutionHintType,
    pub frequency: u64,
    pub optimization_potential: f64,
}

/// Types of execution hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionHintType {
    HotPath,
    ColdPath,
    BranchPrediction,
    ParallelizationCandidate,
}

/// Optimization hint for hotspot bytecode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHint {
    pub hint_type: String,
    pub parameters: HashMap<String, String>,
    pub expected_speedup: f64,
}

/// Snapshot loading statistics
#[derive(Debug, Clone)]
pub struct LoadingStats {
    pub load_time: Duration,
    pub decompression_time: Duration,
    pub validation_time: Duration,
    pub initialization_time: Duration,
    pub total_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub builtin_count: usize,
    pub cache_hit_rate: f64,
}

impl LoadingStats {
    pub fn compression_efficiency(&self) -> f64 {
        1.0 - (self.compressed_size as f64 / self.total_size as f64)
    }
    
    pub fn loading_throughput(&self) -> f64 {
        self.total_size as f64 / self.load_time.as_secs_f64()
    }
}

/// Error types for snapshot operations
#[derive(thiserror::Error, Debug)]
pub enum SnapshotError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    
    #[error("Compression error: {message}")]
    Compression { message: String },
    
    #[error("Validation error: {message}")]
    Validation { message: String },
    
    #[error("Version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: String, found: String },
    
    #[error("Corrupted snapshot: {reason}")]
    Corrupted { reason: String },
    
    #[error("Configuration error: {message}")]
    Configuration { message: String },
}

/// Result type for snapshot operations
pub type SnapshotResult<T> = std::result::Result<T, SnapshotError>;

/// Snapshot configuration
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    /// Enable compression
    pub compression_enabled: bool,
    
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    
    /// Compression level (1-9)
    pub compression_level: u32,
    
    /// Enable validation
    pub validation_enabled: bool,
    
    /// Memory mapping for loading
    pub memory_mapping_enabled: bool,
    
    /// Parallel loading
    pub parallel_loading: bool,
    
    /// Progress reporting
    pub progress_reporting: bool,
    
    /// Maximum cache size
    pub max_cache_size: usize,
    
    /// Cache eviction policy
    pub cache_eviction_policy: CacheEvictionPolicy,
}

/// Compression algorithm options
#[derive(Debug, Clone)]
pub enum CompressionAlgorithm {
    None,
    Lz4,
    Zstd,
    Auto, // Choose best based on data characteristics
}

/// Cache eviction policies
#[derive(Debug, Clone)]
pub enum CacheEvictionPolicy {
    LeastRecentlyUsed,
    LeastFrequentlyUsed,
    TimeToLive(Duration),
    Adaptive,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            compression_enabled: true,
            compression_algorithm: CompressionAlgorithm::Auto,
            compression_level: 6,
            validation_enabled: true,
            memory_mapping_enabled: true,
            parallel_loading: true,
            progress_reporting: false,
            max_cache_size: 128 * 1024 * 1024, // 128MB
            cache_eviction_policy: CacheEvictionPolicy::Adaptive,
        }
    }
}

/// Main snapshot interface
pub struct SnapshotManager {
    config: SnapshotConfig,
    cache: Arc<RwLock<HashMap<PathBuf, Arc<Snapshot>>>>,
    stats: Arc<RwLock<HashMap<PathBuf, LoadingStats>>>,
}

impl SnapshotManager {
    /// Create a new snapshot manager
    pub fn new(config: SnapshotConfig) -> Self {
        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Create a snapshot from the current standard library
    pub fn create_snapshot<P: AsRef<Path>>(&self, output_path: P) -> SnapshotResult<()> {
        let builder = SnapshotBuilder::new(self.config.clone());
        builder.build_and_save(output_path)
    }
    
    /// Load a snapshot from disk
    pub fn load_snapshot<P: AsRef<Path>>(&self, snapshot_path: P) -> SnapshotResult<Arc<Snapshot>> {
        let path = snapshot_path.as_ref().to_path_buf();
        
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(snapshot) = cache.get(&path) {
                return Ok(Arc::clone(snapshot));
            }
        }
        
        // Load from disk
        let mut loader = SnapshotLoader::new(self.config.clone());
        let (snapshot, stats) = loader.load(&path)?;
        let snapshot = Arc::new(snapshot);
        
        // Update cache and stats
        {
            let mut cache = self.cache.write();
            cache.insert(path.clone(), Arc::clone(&snapshot));
        }
        {
            let mut stats_map = self.stats.write();
            stats_map.insert(path, stats);
        }
        
        Ok(snapshot)
    }
    
    /// Get loading statistics for a snapshot
    pub fn get_stats<P: AsRef<Path>>(&self, snapshot_path: P) -> Option<LoadingStats> {
        let stats = self.stats.read();
        stats.get(snapshot_path.as_ref()).cloned()
    }
    
    /// Clear snapshot cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write();
        cache.clear();
        let mut stats = self.stats.write();
        stats.clear();
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read();
        let total_size = cache
            .values()
            .map(|snapshot| {
                bincode::serialized_size(&**snapshot).unwrap_or(0) as usize
            })
            .sum();
        (cache.len(), total_size)
    }
}

impl Default for SnapshotManager {
    fn default() -> Self {
        Self::new(SnapshotConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_snapshot_config_default() {
        let config = SnapshotConfig::default();
        assert!(config.compression_enabled);
        assert!(config.validation_enabled);
        assert!(config.memory_mapping_enabled);
        assert!(config.parallel_loading);
    }
    
    #[test]
    fn test_snapshot_manager_creation() {
        let manager = SnapshotManager::default();
        let (cache_entries, cache_size) = manager.cache_stats();
        assert_eq!(cache_entries, 0);
        assert_eq!(cache_size, 0);
    }
    
    #[test]
    fn test_loading_stats_calculations() {
        let stats = LoadingStats {
            load_time: Duration::from_millis(100),
            decompression_time: Duration::from_millis(20),
            validation_time: Duration::from_millis(10),
            initialization_time: Duration::from_millis(5),
            total_size: 1000,
            compressed_size: 600,
            compression_ratio: 0.4,
            builtin_count: 50,
            cache_hit_rate: 0.8,
        };
        
        assert_eq!(stats.compression_efficiency(), 0.4);
        assert_eq!(stats.loading_throughput(), 10000.0); // bytes per second
    }
}