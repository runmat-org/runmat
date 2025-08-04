//! Snapshot file format and serialization
//!
//! High-performance binary format optimized for fast loading and validation.
//! Uses a structured layout with versioning and integrity checks.

use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

/// Snapshot file format magic number
pub const SNAPSHOT_MAGIC: &[u8; 8] = b"RUSTMAT\x01";

/// Current snapshot format version
pub const SNAPSHOT_VERSION: u32 = 1;

/// Snapshot file format structure
#[derive(Debug, Clone)]
pub struct SnapshotFormat {
    /// File header
    pub header: SnapshotHeader,

    /// Compressed snapshot data
    pub data: Vec<u8>,

    /// Optional integrity checksum
    pub checksum: Option<Vec<u8>>,
}

/// Snapshot file header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotHeader {
    /// Magic number for format identification
    pub magic: [u8; 8],

    /// Format version
    pub version: u32,

    /// Snapshot metadata
    pub metadata: SnapshotMetadata,

    /// Data section info
    pub data_info: DataSectionInfo,

    /// Checksum info (if enabled)
    pub checksum_info: Option<ChecksumInfo>,

    /// Header size (for format evolution)
    pub header_size: u32,
}

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Creation timestamp
    pub created_at: SystemTime,

    /// RustMat version used to create snapshot
    pub rustmat_version: String,

    /// Snapshot creation tool version
    pub tool_version: String,

    /// Build configuration used
    pub build_config: BuildConfig,

    /// Performance characteristics
    pub performance_metrics: PerformanceMetrics,

    /// Feature flags enabled during creation
    pub feature_flags: Vec<String>,

    /// Target platform information
    pub target_platform: PlatformInfo,
}

/// Build configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Optimization level used
    pub optimization_level: String,

    /// Debug information included
    pub debug_info: bool,

    /// Compiler used
    pub compiler: String,

    /// Compilation flags
    pub compile_flags: Vec<String>,

    /// Features enabled
    pub enabled_features: Vec<String>,
}

/// Performance metrics from snapshot creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Time to create snapshot
    pub creation_time: Duration,

    /// Number of builtins captured
    pub builtin_count: usize,

    /// HIR cache entries
    pub hir_cache_entries: usize,

    /// Bytecode cache entries
    pub bytecode_cache_entries: usize,

    /// Total uncompressed size
    pub uncompressed_size: usize,

    /// Compression ratio achieved
    pub compression_ratio: f64,

    /// Memory usage during creation
    pub peak_memory_usage: usize,
}

/// Target platform information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// Operating system
    pub os: String,

    /// Architecture
    pub arch: String,

    /// CPU features available
    pub cpu_features: Vec<String>,

    /// Memory page size
    pub page_size: usize,

    /// Cache line size
    pub cache_line_size: usize,

    /// Endianness
    pub endianness: Endianness,
}

/// Endianness information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Endianness {
    Little,
    Big,
}

/// Data section information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSectionInfo {
    /// Compression algorithm used
    pub compression: CompressionInfo,

    /// Uncompressed data size
    pub uncompressed_size: usize,

    /// Compressed data size
    pub compressed_size: usize,

    /// Data section offset in file
    pub data_offset: u64,

    /// Alignment requirements
    pub alignment: usize,
}

/// Compression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,

    /// Compression level
    pub level: u32,

    /// Algorithm-specific parameters
    pub parameters: std::collections::HashMap<String, String>,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompressionAlgorithm {
    None,
    Lz4 { fast: bool },
    Zstd { dictionary: Option<Vec<u8>> },
}

/// Checksum information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumInfo {
    /// Checksum algorithm
    pub algorithm: ChecksumAlgorithm,

    /// Checksum size in bytes
    pub size: usize,

    /// Checksum offset in file
    pub offset: u64,
}

/// Checksum algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChecksumAlgorithm {
    Sha256,
    Blake3,
    Crc32,
}

impl SnapshotHeader {
    /// Create a new snapshot header
    pub fn new(metadata: SnapshotMetadata) -> Self {
        Self {
            magic: *SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
            metadata,
            data_info: DataSectionInfo {
                compression: CompressionInfo {
                    algorithm: CompressionAlgorithm::None,
                    level: 0,
                    parameters: std::collections::HashMap::new(),
                },
                uncompressed_size: 0,
                compressed_size: 0,
                data_offset: 0,
                alignment: 8,
            },
            checksum_info: None,
            header_size: 0, // Will be calculated during serialization
        }
    }

    /// Validate header magic and version
    pub fn validate(&self) -> crate::SnapshotResult<()> {
        if self.magic != *SNAPSHOT_MAGIC {
            return Err(crate::SnapshotError::Corrupted {
                reason: "Invalid magic number".to_string(),
            });
        }

        if self.version > SNAPSHOT_VERSION {
            return Err(crate::SnapshotError::VersionMismatch {
                expected: SNAPSHOT_VERSION.to_string(),
                found: self.version.to_string(),
            });
        }

        Ok(())
    }

    /// Check if snapshot is compatible with current platform
    pub fn is_platform_compatible(&self) -> bool {
        let current_os = std::env::consts::OS;
        let current_arch = std::env::consts::ARCH;

        self.metadata.target_platform.os == current_os
            && self.metadata.target_platform.arch == current_arch
    }

    /// Get expected loading performance characteristics
    pub fn estimated_load_time(&self) -> Duration {
        // Estimate based on data size and compression
        let base_time = Duration::from_millis(10); // Base overhead
        let data_time = Duration::from_nanos(
            (self.data_info.compressed_size as u64 * 10) / 1024, // ~10ns per KB
        );

        match self.data_info.compression.algorithm {
            CompressionAlgorithm::None => base_time + data_time,
            CompressionAlgorithm::Lz4 { .. } => base_time + data_time * 2,
            CompressionAlgorithm::Zstd { .. } => base_time + data_time * 4,
        }
    }
}

impl SnapshotMetadata {
    /// Create metadata for current environment
    pub fn current() -> Self {
        Self {
            created_at: SystemTime::now(),
            rustmat_version: env!("CARGO_PKG_VERSION").to_string(),
            tool_version: env!("CARGO_PKG_VERSION").to_string(),
            build_config: BuildConfig::current(),
            performance_metrics: PerformanceMetrics::default(),
            feature_flags: Self::detect_feature_flags(),
            target_platform: PlatformInfo::current(),
        }
    }

    /// Detect active feature flags
    #[allow(clippy::vec_init_then_push)] // Conditional compilation makes vec![] problematic
    fn detect_feature_flags() -> Vec<String> {
        let mut flags = Vec::new();

        #[cfg(feature = "compression")]
        flags.push("compression".to_string());

        #[cfg(feature = "validation")]
        flags.push("validation".to_string());

        #[cfg(feature = "blas-lapack")]
        flags.push("blas-lapack".to_string());

        flags
    }

    /// Check compatibility with current environment
    pub fn is_compatible(&self) -> bool {
        // Check major version compatibility
        let current_version = env!("CARGO_PKG_VERSION");
        let current_major = current_version.split('.').next().unwrap_or("0");
        let snapshot_major = self.rustmat_version.split('.').next().unwrap_or("0");

        current_major == snapshot_major
    }

    /// Get human-readable age of snapshot
    pub fn age(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.created_at)
            .unwrap_or(Duration::ZERO)
    }
}

impl BuildConfig {
    /// Detect current build configuration
    pub fn current() -> Self {
        Self {
            optimization_level: if cfg!(debug_assertions) {
                "debug".to_string()
            } else {
                "release".to_string()
            },
            debug_info: cfg!(debug_assertions),
            compiler: format!(
                "rustc {}",
                option_env!("RUSTC_VERSION").unwrap_or("unknown")
            ),
            compile_flags: Vec::new(), // Would need to be passed from build system
            enabled_features: Vec::new(), // Would need feature detection
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            creation_time: Duration::ZERO,
            builtin_count: 0,
            hir_cache_entries: 0,
            bytecode_cache_entries: 0,
            uncompressed_size: 0,
            compression_ratio: 1.0,
            peak_memory_usage: 0,
        }
    }
}

impl PlatformInfo {
    /// Detect current platform information
    pub fn current() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_features: Self::detect_cpu_features(),
            page_size: Self::detect_page_size(),
            cache_line_size: Self::detect_cache_line_size(),
            endianness: if cfg!(target_endian = "little") {
                Endianness::Little
            } else {
                Endianness::Big
            },
        }
    }

    /// Detect available CPU features
    fn detect_cpu_features() -> Vec<String> {
        let mut features = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("sse4.2") {
                features.push("sse4.2".to_string());
            }
            if std::arch::is_x86_feature_detected!("avx") {
                features.push("avx".to_string());
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                features.push("avx2".to_string());
            }
            if std::arch::is_x86_feature_detected!("fma") {
                features.push("fma".to_string());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                features.push("neon".to_string());
            }
        }

        features
    }

    /// Detect memory page size
    fn detect_page_size() -> usize {
        // Default to common page sizes
        #[cfg(unix)]
        {
            unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
        }
        #[cfg(not(unix))]
        {
            4096 // Common default
        }
    }

    /// Detect CPU cache line size
    fn detect_cache_line_size() -> usize {
        // Use common default, could be detected more precisely
        64
    }
}

impl SnapshotFormat {
    /// Create a new snapshot format
    pub fn new(header: SnapshotHeader, data: Vec<u8>) -> Self {
        Self {
            header,
            data,
            checksum: None,
        }
    }

    /// Calculate and set checksum
    pub fn with_checksum(mut self, algorithm: ChecksumAlgorithm) -> crate::SnapshotResult<Self> {
        #[cfg(feature = "validation")]
        {
            use sha2::{Digest, Sha256};

            let checksum = match algorithm {
                ChecksumAlgorithm::Sha256 => {
                    let mut hasher = Sha256::new();
                    hasher.update(&self.data);
                    hasher.finalize().to_vec()
                }
                ChecksumAlgorithm::Blake3 => blake3::hash(&self.data).as_bytes().to_vec(),
                ChecksumAlgorithm::Crc32 => {
                    let crc = crc32fast::hash(&self.data);
                    crc.to_le_bytes().to_vec()
                }
            };

            self.checksum = Some(checksum.clone());
            self.header.checksum_info = Some(ChecksumInfo {
                algorithm,
                size: checksum.len(),
                offset: 0, // Will be set during serialization
            });
        }
        #[cfg(not(feature = "validation"))]
        {
            return Err(crate::SnapshotError::Configuration {
                message: "Validation feature not enabled".to_string(),
            });
        }

        Ok(self)
    }

    /// Validate checksum
    pub fn validate_checksum(&self) -> crate::SnapshotResult<bool> {
        #[cfg(feature = "validation")]
        {
            if let (Some(checksum_info), Some(stored_checksum)) =
                (&self.header.checksum_info, &self.checksum)
            {
                use sha2::{Digest, Sha256};

                let calculated_checksum = match checksum_info.algorithm {
                    ChecksumAlgorithm::Sha256 => {
                        let mut hasher = Sha256::new();
                        hasher.update(&self.data);
                        hasher.finalize().to_vec()
                    }
                    ChecksumAlgorithm::Blake3 => blake3::hash(&self.data).as_bytes().to_vec(),
                    ChecksumAlgorithm::Crc32 => {
                        let crc = crc32fast::hash(&self.data);
                        crc.to_le_bytes().to_vec()
                    }
                };

                Ok(calculated_checksum == *stored_checksum)
            } else {
                Ok(true) // No checksum to validate
            }
        }
        #[cfg(not(feature = "validation"))]
        {
            Ok(true) // Skip validation if feature disabled
        }
    }

    /// Get total file size
    pub fn total_size(&self) -> usize {
        let header_size = bincode::serialized_size(&self.header).unwrap_or(0) as usize;
        let data_size = self.data.len();
        let checksum_size = self.checksum.as_ref().map_or(0, |c| c.len());

        header_size + data_size + checksum_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_header_validation() {
        let metadata = SnapshotMetadata::current();
        let header = SnapshotHeader::new(metadata);

        assert!(header.validate().is_ok());
        assert_eq!(header.magic, *SNAPSHOT_MAGIC);
        assert_eq!(header.version, SNAPSHOT_VERSION);
    }

    #[test]
    fn test_platform_compatibility() {
        let metadata = SnapshotMetadata::current();
        let header = SnapshotHeader::new(metadata);

        assert!(header.is_platform_compatible());
    }

    #[test]
    fn test_metadata_compatibility() {
        let metadata = SnapshotMetadata::current();
        assert!(metadata.is_compatible());
    }

    #[test]
    fn test_platform_info() {
        let platform = PlatformInfo::current();
        assert!(!platform.os.is_empty());
        assert!(!platform.arch.is_empty());
        assert!(platform.page_size > 0);
        assert!(platform.cache_line_size > 0);
    }

    #[test]
    fn test_build_config() {
        let config = BuildConfig::current();
        assert!(!config.optimization_level.is_empty());
        assert!(!config.compiler.is_empty());
    }

    #[test]
    fn test_snapshot_format_creation() {
        let metadata = SnapshotMetadata::current();
        let header = SnapshotHeader::new(metadata);
        let data = vec![1, 2, 3, 4, 5];
        let format = SnapshotFormat::new(header, data);

        assert_eq!(format.data.len(), 5);
        assert!(format.checksum.is_none());
    }

    #[cfg(feature = "validation")]
    #[test]
    fn test_checksum_generation() {
        let metadata = SnapshotMetadata::current();
        let header = SnapshotHeader::new(metadata);
        let data = vec![1, 2, 3, 4, 5];
        let format = SnapshotFormat::new(header, data);

        let format_with_checksum = format.with_checksum(ChecksumAlgorithm::Sha256).unwrap();

        assert!(format_with_checksum.checksum.is_some());
        assert!(format_with_checksum.header.checksum_info.is_some());
        assert!(format_with_checksum.validate_checksum().unwrap());
    }

    #[test]
    fn test_estimated_load_time() {
        let metadata = SnapshotMetadata::current();
        let mut header = SnapshotHeader::new(metadata);
        header.data_info.compressed_size = 1024 * 1024; // 1MB

        let load_time = header.estimated_load_time();
        assert!(load_time > Duration::ZERO);
    }
}
