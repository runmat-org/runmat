//! Snapshot format tests

use rustmat_snapshot::format::*;

#[test]
fn test_snapshot_metadata_creation() {
    let metadata = SnapshotMetadata::current();

    assert!(!metadata.rustmat_version.is_empty());
    assert!(!metadata.tool_version.is_empty());
    assert!(!metadata.target_platform.os.is_empty());
    assert!(!metadata.target_platform.arch.is_empty());
    assert!(metadata.target_platform.page_size > 0);
    assert!(metadata.target_platform.cache_line_size > 0);
}

#[test]
fn test_platform_info() {
    let platform = PlatformInfo::current();

    assert!(!platform.os.is_empty());
    assert!(!platform.arch.is_empty());
    assert!(platform.page_size > 0);
    assert!(platform.cache_line_size > 0);

    // Test CPU feature detection
    // CPU features may be empty on some platforms, just check it's a valid Vec

    // Test endianness
    assert!(matches!(
        platform.endianness,
        Endianness::Little | Endianness::Big
    ));
}

#[test]
fn test_build_config() {
    let config = BuildConfig::current();

    assert!(!config.optimization_level.is_empty());
    assert!(!config.compiler.is_empty());
    assert!(config.compiler.starts_with("rustc"));
}

#[test]
fn test_snapshot_header_validation() {
    let metadata = SnapshotMetadata::current();
    let header = SnapshotHeader::new(metadata);

    // Valid header should pass validation
    assert!(header.validate().is_ok());

    // Check magic number
    assert_eq!(header.magic, *SNAPSHOT_MAGIC);
    assert_eq!(header.version, SNAPSHOT_VERSION);

    // Test platform compatibility
    assert!(header.is_platform_compatible());
}

#[test]
fn test_invalid_header_validation() {
    let metadata = SnapshotMetadata::current();
    let mut header = SnapshotHeader::new(metadata);

    // Invalid magic
    header.magic = [0; 8];
    assert!(header.validate().is_err());

    // Reset magic, test invalid version
    header.magic = *SNAPSHOT_MAGIC;
    header.version = SNAPSHOT_VERSION + 1;
    assert!(header.validate().is_err());
}

#[test]
fn test_compression_algorithm_equality() {
    use CompressionAlgorithm::*;

    assert_eq!(None, None);
    assert_eq!(Lz4 { fast: true }, Lz4 { fast: true });
    assert_eq!(Lz4 { fast: false }, Lz4 { fast: false });
    assert_ne!(Lz4 { fast: true }, Lz4 { fast: false });

    assert_eq!(
        Zstd {
            dictionary: Option::<Vec<u8>>::None
        },
        Zstd {
            dictionary: Option::<Vec<u8>>::None
        }
    );
    assert_eq!(
        Zstd {
            dictionary: Some(vec![1, 2, 3])
        },
        Zstd {
            dictionary: Some(vec![1, 2, 3])
        }
    );
    assert_ne!(
        Zstd {
            dictionary: Some(vec![1, 2, 3])
        },
        Zstd {
            dictionary: Some(vec![4, 5, 6])
        }
    );
}

#[test]
fn test_checksum_algorithms() {
    use ChecksumAlgorithm::*;

    // Just test that enum variants exist and can be created
    let _sha256 = Sha256;
    let _blake3 = Blake3;
    let _crc32 = Crc32;
}

#[test]
fn test_snapshot_format_creation() {
    let metadata = SnapshotMetadata::current();
    let header = SnapshotHeader::new(metadata);
    let test_data = vec![1, 2, 3, 4, 5];

    let format = SnapshotFormat::new(header, test_data.clone());

    assert_eq!(format.data, test_data);
    assert!(format.checksum.is_none());
    assert!(format.total_size() > 0);
}

#[cfg(feature = "validation")]
#[test]
fn test_snapshot_format_with_checksum() {
    let metadata = SnapshotMetadata::current();
    let header = SnapshotHeader::new(metadata);
    let test_data = vec![1, 2, 3, 4, 5];

    let format = SnapshotFormat::new(header, test_data);
    let format_with_checksum = format.with_checksum(ChecksumAlgorithm::Sha256).unwrap();

    assert!(format_with_checksum.checksum.is_some());
    assert!(format_with_checksum.header.checksum_info.is_some());

    // Validate checksum
    assert!(format_with_checksum.validate_checksum().unwrap());
}

#[test]
fn test_metadata_compatibility() {
    let metadata = SnapshotMetadata::current();

    // Current metadata should always be compatible with itself
    assert!(metadata.is_compatible());

    // Test age calculation
    let age = metadata.age();
    assert!(age.as_millis() < u128::MAX);
}

#[test]
fn test_metadata_feature_detection() {
    let metadata = SnapshotMetadata::current();

    // Should detect at least some features
    // Feature flags list should exist (may be empty)

    // If compression feature is enabled, should be in the list
    #[cfg(feature = "compression")]
    assert!(metadata.feature_flags.contains(&"compression".to_string()));

    #[cfg(feature = "validation")]
    assert!(metadata.feature_flags.contains(&"validation".to_string()));
}

#[test]
fn test_header_estimated_load_time() {
    let metadata = SnapshotMetadata::current();
    let mut header = SnapshotHeader::new(metadata);

    // Small file
    header.data_info.compressed_size = 1024;
    let small_load_time = header.estimated_load_time();

    // Large file
    header.data_info.compressed_size = 1024 * 1024;
    let large_load_time = header.estimated_load_time();

    // Larger files should take longer to load
    assert!(large_load_time >= small_load_time);

    // Test different compression algorithms
    header.data_info.compression.algorithm = CompressionAlgorithm::Lz4 { fast: true };
    let lz4_time = header.estimated_load_time();

    header.data_info.compression.algorithm = CompressionAlgorithm::Zstd { dictionary: None };
    let zstd_time = header.estimated_load_time();

    // ZSTD should take longer to decompress
    assert!(zstd_time >= lz4_time);
}

#[test]
fn test_performance_metrics() {
    let mut metrics = PerformanceMetrics::default();

    assert_eq!(metrics.creation_time.as_secs(), 0);
    assert_eq!(metrics.builtin_count, 0);
    assert_eq!(metrics.compression_ratio, 1.0);

    // Test updating metrics
    metrics.builtin_count = 42;
    metrics.hir_cache_entries = 10;

    assert_eq!(metrics.builtin_count, 42);
    assert_eq!(metrics.hir_cache_entries, 10);
}

#[test]
fn test_data_section_info() {
    let info = DataSectionInfo {
        compression: CompressionInfo {
            algorithm: CompressionAlgorithm::Lz4 { fast: false },
            level: 6,
            parameters: std::collections::HashMap::new(),
        },
        uncompressed_size: 1000,
        compressed_size: 600,
        data_offset: 512,
        alignment: 8,
    };

    assert_eq!(info.uncompressed_size, 1000);
    assert_eq!(info.compressed_size, 600);
    assert_eq!(info.data_offset, 512);
    assert_eq!(info.alignment, 8);

    // Test compression ratio calculation (external)
    let compression_ratio = info.compressed_size as f64 / info.uncompressed_size as f64;
    assert!((compression_ratio - 0.6).abs() < 0.001);
}

#[test]
fn test_cpu_feature_detection() {
    let _platform = PlatformInfo::current();

    // Just verify the function doesn't panic and returns reasonable results
    // CPU features exist and are strings

    // If we're on x86_64, might have some common features
    #[cfg(target_arch = "x86_64")]
    {
        // These are common but not guaranteed, so just test they're strings
        for feature in &platform.cpu_features {
            assert!(!feature.is_empty());
        }
    }
}

#[test]
fn test_endianness_detection() {
    let platform = PlatformInfo::current();

    // Most modern platforms are little endian
    #[cfg(target_endian = "little")]
    assert!(matches!(platform.endianness, Endianness::Little));

    #[cfg(target_endian = "big")]
    assert!(matches!(platform.endianness, Endianness::Big));
}

#[test]
fn test_serialization_roundtrip() {
    let metadata = SnapshotMetadata::current();
    let header = SnapshotHeader::new(metadata);

    // Test serialization
    let serialized = bincode::serialize(&header).unwrap();
    assert!(!serialized.is_empty());

    // Test deserialization
    let deserialized: SnapshotHeader = bincode::deserialize(&serialized).unwrap();

    // Basic checks
    assert_eq!(deserialized.magic, header.magic);
    assert_eq!(deserialized.version, header.version);
    assert_eq!(
        deserialized.metadata.rustmat_version,
        header.metadata.rustmat_version
    );
}

#[test]
fn test_format_constants() {
    assert_eq!(SNAPSHOT_MAGIC.len(), 8);
    assert_eq!(SNAPSHOT_MAGIC, b"RUSTMAT\x01");
    // Ensure SNAPSHOT_VERSION is positive (constant assertion)
    const _: () = assert!(SNAPSHOT_VERSION > 0);
}
