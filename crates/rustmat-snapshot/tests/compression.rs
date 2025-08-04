//! Compression system tests

use rustmat_snapshot::compression::{CompressionConfig, CompressionEngine};

#[test]
fn test_compression_config() {
    let config = CompressionConfig::default();
    assert_eq!(config.default_level, 6);
    assert!(config.adaptive_selection);
    assert_eq!(config.size_threshold, 1024);
}

#[test]
fn test_compression_engine_creation() {
    let config = CompressionConfig::default();
    let engine = CompressionEngine::new(config);
    assert_eq!(engine.stats().total_bytes, 0);
}

#[test]
fn test_small_data_skips_compression() {
    let mut engine = CompressionEngine::new(CompressionConfig::default());
    let small_data = vec![1, 2, 3]; // Less than threshold

    let result = engine.compress(&small_data).unwrap();
    assert_eq!(result.data, small_data);
    assert!(matches!(
        result.info.algorithm,
        rustmat_snapshot::format::CompressionAlgorithm::None
    ));
}

#[cfg(feature = "compression")]
#[test]
fn test_lz4_compression_roundtrip() {
    let mut engine = CompressionEngine::new(CompressionConfig {
        adaptive_selection: false,
        prefer_speed: true,
        size_threshold: 10, // Lower threshold for test
        ..CompressionConfig::default()
    });

    let test_data =
        b"Hello, World! This is a test string that should compress reasonably well with LZ4."
            .repeat(10);

    // Compress
    let result = engine.compress(&test_data).unwrap();
    assert!(matches!(
        result.info.algorithm,
        rustmat_snapshot::format::CompressionAlgorithm::Lz4 { .. }
    ));

    // Decompress
    let decompressed = engine.decompress(&result.data, &result.info).unwrap();
    assert_eq!(decompressed, test_data);
}

#[cfg(feature = "compression")]
#[test]
fn test_zstd_compression_roundtrip() {
    let mut engine = CompressionEngine::new(CompressionConfig {
        adaptive_selection: false,
        prefer_speed: false,
        size_threshold: 10,
        default_level: 3,
        ..CompressionConfig::default()
    });

    let test_data =
        b"This is a longer test string that should compress well with ZSTD compression algorithm. "
            .repeat(20);

    // Force ZSTD compression
    let result = engine
        .compress_with_algorithm(
            &test_data,
            rustmat_snapshot::format::CompressionAlgorithm::Zstd { dictionary: None },
        )
        .unwrap();

    assert!(matches!(
        result.info.algorithm,
        rustmat_snapshot::format::CompressionAlgorithm::Zstd { .. }
    ));

    // Should achieve some compression
    assert!(result.data.len() < test_data.len());

    // Decompress
    let decompressed = engine.decompress(&result.data, &result.info).unwrap();
    assert_eq!(decompressed, test_data);
}

#[test]
fn test_compression_statistics() {
    let mut engine = CompressionEngine::new(CompressionConfig::default());
    let test_data = vec![0u8; 2048]; // Compressible data

    // Initial stats
    assert_eq!(engine.stats().total_bytes, 0);

    // Compress some data
    let _result = engine.compress(&test_data).unwrap();

    // Check updated stats
    let stats = engine.stats();
    assert!(stats.total_bytes > 0);
    assert!(!stats.attempts.is_empty());

    // Test reset
    engine.reset_stats();
    assert_eq!(engine.stats().total_bytes, 0);
}

#[test]
fn test_data_characteristics_analysis() {
    let engine = CompressionEngine::new(CompressionConfig::default());

    // ASCII text data
    let ascii_data = b"Hello, World! This is ASCII text.";
    let characteristics = engine.analyze_data(ascii_data);
    assert!(characteristics.ascii_ratio > 0.9);

    // Binary data
    let binary_data: Vec<u8> = (0..=255).collect();
    let characteristics = engine.analyze_data(&binary_data);
    assert!(characteristics.ascii_ratio < 1.0);

    // Repetitive data
    let repetitive_data = vec![42u8; 1000];
    let characteristics = engine.analyze_data(&repetitive_data);
    assert!(characteristics.entropy < 0.1); // Very low entropy
}

#[test]
fn test_adaptive_algorithm_selection() {
    let engine = CompressionEngine::new(CompressionConfig {
        adaptive_selection: true,
        ..CompressionConfig::default()
    });

    // High entropy data should skip compression
    let random_data: Vec<u8> = (0..1000).map(|i| (i * 7 % 256) as u8).collect();
    let _algorithm = engine.select_optimal_algorithm(&random_data).unwrap();
    // Should choose a fast algorithm or no compression

    // Repetitive data should use compression
    let repetitive_data = vec![42u8; 1000];
    let algorithm = engine.select_optimal_algorithm(&repetitive_data).unwrap();
    assert!(!matches!(
        algorithm,
        rustmat_snapshot::format::CompressionAlgorithm::None
    ));
}

#[test]
fn test_compression_effectiveness_check() {
    let mut engine = CompressionEngine::new(CompressionConfig {
        size_threshold: 10,
        ..CompressionConfig::default()
    });

    // Data that doesn't compress well (already random/compressed)
    let incompressible_data: Vec<u8> = (0..1000).map(|i| (i * 17 + 42) as u8).collect();

    let result = engine.compress(&incompressible_data).unwrap();

    // If compression wasn't effective, should fall back to uncompressed
    if result.metrics.compression_ratio > 0.95 {
        assert!(matches!(
            result.info.algorithm,
            rustmat_snapshot::format::CompressionAlgorithm::None
        ));
    }
}

#[test]
fn test_compression_metrics() {
    let mut engine = CompressionEngine::new(CompressionConfig {
        size_threshold: 10,
        ..CompressionConfig::default()
    });

    let test_data = vec![65u8; 1000]; // Highly compressible
    let result = engine.compress(&test_data).unwrap();

    // Check metrics
    assert!(result.metrics.compression_time.as_nanos() < u128::MAX);
    assert!(result.metrics.compression_ratio >= 0.0);
    assert!(result.metrics.compression_ratio <= 1.0);
    assert!(result.metrics.throughput > 0.0);
    assert!(result.metrics.memory_usage < usize::MAX);
}

#[test]
fn test_compression_stats_aggregation() {
    let mut stats = rustmat_snapshot::compression::CompressionStats::default();

    // Initially empty
    assert_eq!(stats.overall_ratio(), 1.0);
    assert_eq!(stats.overall_throughput(), 0.0);
    assert!(stats.best_algorithm().is_none());

    // Add some stats manually (simulating compression operations)
    stats.ratios.insert("lz4".to_string(), 0.7);
    stats.ratios.insert("zstd".to_string(), 0.5);

    // Test aggregation
    let overall_ratio = stats.overall_ratio();
    assert!(overall_ratio > 0.0 && overall_ratio < 1.0);

    let best = stats.best_algorithm();
    assert_eq!(best, Some("zstd".to_string())); // Best compression ratio
}
