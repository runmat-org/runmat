//! Snapshot preset tests

use runmat_snapshot::presets::*;

#[test]
fn test_all_presets_exist() {
    let presets = SnapshotPreset::all_presets();
    assert_eq!(presets.len(), 6);

    let names: Vec<_> = presets.iter().map(|p| p.name()).collect();
    assert!(names.contains(&"Development"));
    assert!(names.contains(&"Production"));
    assert!(names.contains(&"High-Performance"));
    assert!(names.contains(&"Low-Memory"));
    assert!(names.contains(&"Network-Optimized"));
    assert!(names.contains(&"Debug"));
}

#[test]
fn test_preset_from_name() {
    // Test exact names
    assert!(matches!(
        SnapshotPreset::from_name("Development"),
        Some(SnapshotPreset::Development)
    ));
    assert!(matches!(
        SnapshotPreset::from_name("Production"),
        Some(SnapshotPreset::Production)
    ));

    // Test aliases
    assert!(matches!(
        SnapshotPreset::from_name("dev"),
        Some(SnapshotPreset::Development)
    ));
    assert!(matches!(
        SnapshotPreset::from_name("prod"),
        Some(SnapshotPreset::Production)
    ));
    assert!(matches!(
        SnapshotPreset::from_name("highperf"),
        Some(SnapshotPreset::HighPerformance)
    ));
    assert!(matches!(
        SnapshotPreset::from_name("lowmem"),
        Some(SnapshotPreset::LowMemory)
    ));
    assert!(matches!(
        SnapshotPreset::from_name("network"),
        Some(SnapshotPreset::NetworkOptimized)
    ));
    assert!(matches!(
        SnapshotPreset::from_name("debug"),
        Some(SnapshotPreset::Debug)
    ));

    // Test invalid names
    assert!(SnapshotPreset::from_name("invalid").is_none());
    assert!(SnapshotPreset::from_name("").is_none());
    assert!(SnapshotPreset::from_name("random").is_none());
}

#[test]
fn test_preset_configurations() {
    // Development preset
    let dev_config = SnapshotPreset::Development.config();
    assert!(dev_config.compression_enabled);
    assert_eq!(dev_config.compression_level, 1); // Fast compression
    assert!(dev_config.validation_enabled);
    assert!(dev_config.progress_reporting); // Helpful during development

    // Production preset
    let prod_config = SnapshotPreset::Production.config();
    assert!(prod_config.compression_enabled);
    assert_eq!(prod_config.compression_level, 6); // Balanced compression
    assert!(prod_config.validation_enabled);
    assert!(!prod_config.progress_reporting); // No progress in production

    // High-performance preset
    let perf_config = SnapshotPreset::HighPerformance.config();
    assert!(perf_config.compression_enabled);
    assert_eq!(perf_config.compression_level, 1); // Fastest decompression
    assert!(!perf_config.validation_enabled); // Skip validation for speed
    assert_eq!(perf_config.max_cache_size, 256 * 1024 * 1024); // Larger cache

    // Low-memory preset
    let mem_config = SnapshotPreset::LowMemory.config();
    assert!(mem_config.compression_enabled);
    assert_eq!(mem_config.compression_level, 9); // Maximum compression
    assert!(!mem_config.memory_mapping_enabled); // Avoid memory mapping
    assert!(!mem_config.parallel_loading); // Reduce memory overhead
    assert_eq!(mem_config.max_cache_size, 16 * 1024 * 1024); // Minimal cache

    // Network-optimized preset
    let net_config = SnapshotPreset::NetworkOptimized.config();
    assert!(net_config.compression_enabled);
    assert_eq!(net_config.compression_level, 9); // Maximum compression

    // Debug preset
    let debug_config = SnapshotPreset::Debug.config();
    assert!(!debug_config.compression_enabled); // No compression for easier debugging
    assert_eq!(debug_config.compression_level, 0);
    assert!(debug_config.validation_enabled);
    assert!(!debug_config.memory_mapping_enabled); // Easier to debug without mmap
    assert!(!debug_config.parallel_loading); // Sequential for easier debugging
    assert!(debug_config.progress_reporting); // Detailed progress
}

#[test]
fn test_preset_characteristics() {
    let high_perf = SnapshotPreset::HighPerformance;
    let chars = high_perf.characteristics();

    assert_eq!(chars.load_time, LoadTime::VeryFast);
    assert_eq!(chars.memory_usage, MemoryUsage::High);
    assert_eq!(chars.validation_level, ValidationLevel::Minimal);
    assert!(!chars.debugging_friendly);

    let debug = SnapshotPreset::Debug;
    let chars = debug.characteristics();

    assert_eq!(chars.file_size, FileSize::Large); // No compression
    assert_eq!(chars.validation_level, ValidationLevel::Comprehensive);
    assert!(chars.debugging_friendly);

    let low_mem = SnapshotPreset::LowMemory;
    let chars = low_mem.characteristics();

    assert_eq!(chars.memory_usage, MemoryUsage::VeryLow);
    assert_eq!(chars.file_size, FileSize::VerySmall); // Maximum compression
    assert_eq!(chars.build_time, BuildTime::Slow); // High compression takes time
}

#[test]
fn test_characteristic_ordering() {
    // Test that characteristic enums are properly ordered
    assert!(BuildTime::VeryFast < BuildTime::Fast);
    assert!(BuildTime::Fast < BuildTime::Medium);
    assert!(BuildTime::Medium < BuildTime::Slow);
    assert!(BuildTime::Slow < BuildTime::VerySlow);

    assert!(LoadTime::VeryFast < LoadTime::Fast);
    assert!(LoadTime::Fast < LoadTime::Medium);

    assert!(FileSize::VerySmall < FileSize::Small);
    assert!(FileSize::Small < FileSize::Medium);
    assert!(FileSize::Medium < FileSize::Large);

    assert!(MemoryUsage::VeryLow < MemoryUsage::Low);
    assert!(MemoryUsage::Low < MemoryUsage::Medium);
    assert!(MemoryUsage::Medium < MemoryUsage::High);

    assert!(ValidationLevel::None < ValidationLevel::Minimal);
    assert!(ValidationLevel::Minimal < ValidationLevel::Standard);
    assert!(ValidationLevel::Standard < ValidationLevel::Comprehensive);
}

#[test]
fn test_characteristic_display() {
    assert_eq!(BuildTime::VeryFast.to_string(), "Very Fast");
    assert_eq!(LoadTime::Medium.to_string(), "Medium");
    assert_eq!(FileSize::Large.to_string(), "Large");
    assert_eq!(MemoryUsage::Low.to_string(), "Low");
    assert_eq!(ValidationLevel::Comprehensive.to_string(), "Comprehensive");
}

#[test]
fn test_preset_names_and_descriptions() {
    for preset in SnapshotPreset::all_presets() {
        let name = preset.name();
        let description = preset.description();

        assert!(!name.is_empty());
        assert!(!description.is_empty());
        assert!(description.len() > 20); // Should be descriptive
    }
}

#[test]
fn test_custom_preset() {
    use runmat_snapshot::{CacheEvictionPolicy, CompressionAlgorithm, SnapshotConfig};
    use std::time::Duration;

    let custom_config = SnapshotConfig {
        compression_enabled: true,
        compression_algorithm: CompressionAlgorithm::Lz4,
        compression_level: 5,
        validation_enabled: false,
        memory_mapping_enabled: false,
        parallel_loading: true,
        progress_reporting: true,
        max_cache_size: 64 * 1024 * 1024,
        cache_eviction_policy: CacheEvictionPolicy::TimeToLive(Duration::from_secs(300)),
    };

    let custom_preset = SnapshotPreset::Custom(custom_config.clone());
    assert_eq!(custom_preset.name(), "Custom");

    let retrieved_config = custom_preset.config();
    assert_eq!(retrieved_config.compression_level, 5);
    assert!(!retrieved_config.validation_enabled);
    assert!(retrieved_config.progress_reporting);
}

#[test]
fn test_preset_logical_consistency() {
    // Test that preset configurations make logical sense

    // High-performance should prioritize speed
    let perf_config = SnapshotPreset::HighPerformance.config();
    assert!(perf_config.compression_level <= 3); // Fast compression
    assert!(!perf_config.validation_enabled); // Skip validation

    // Low-memory should minimize memory usage
    let mem_config = SnapshotPreset::LowMemory.config();
    assert!(mem_config.compression_level >= 7); // High compression
    assert!(!mem_config.memory_mapping_enabled); // No mmap
    assert!(!mem_config.parallel_loading); // Sequential loading
    assert!(mem_config.max_cache_size <= 32 * 1024 * 1024); // Small cache

    // Development should be developer-friendly
    let dev_config = SnapshotPreset::Development.config();
    assert!(dev_config.progress_reporting); // Show progress
    assert!(dev_config.validation_enabled); // Catch issues early

    // Production should be optimized but reliable
    let prod_config = SnapshotPreset::Production.config();
    assert!(!prod_config.progress_reporting); // No unnecessary output
    assert!(prod_config.validation_enabled); // Ensure integrity
    assert!(prod_config.compression_level >= 3 && prod_config.compression_level <= 7);
    // Balanced
}

#[test]
fn test_case_insensitive_preset_lookup() {
    assert!(matches!(
        SnapshotPreset::from_name("DEVELOPMENT"),
        Some(SnapshotPreset::Development)
    ));
    assert!(matches!(
        SnapshotPreset::from_name("production"),
        Some(SnapshotPreset::Production)
    ));
    assert!(matches!(
        SnapshotPreset::from_name("High-Performance"),
        Some(SnapshotPreset::HighPerformance)
    ));
    assert!(matches!(
        SnapshotPreset::from_name("DEBUG"),
        Some(SnapshotPreset::Debug)
    ));
}

#[test]
fn test_preset_uniqueness() {
    let presets = SnapshotPreset::all_presets();
    let names: Vec<_> = presets.iter().map(|p| p.name()).collect();

    // All names should be unique
    let mut unique_names = names.clone();
    unique_names.sort();
    unique_names.dedup();
    assert_eq!(names.len(), unique_names.len());
}

#[test]
fn test_compression_algorithm_choices() {
    use runmat_snapshot::CompressionAlgorithm;

    // High-performance should use fast algorithms
    let perf_config = SnapshotPreset::HighPerformance.config();
    assert!(matches!(
        perf_config.compression_algorithm,
        CompressionAlgorithm::Lz4
    ));

    // Network-optimized should use high-compression algorithms
    let net_config = SnapshotPreset::NetworkOptimized.config();
    assert!(matches!(
        net_config.compression_algorithm,
        CompressionAlgorithm::Zstd
    ));

    // Debug should not use compression
    let debug_config = SnapshotPreset::Debug.config();
    assert!(matches!(
        debug_config.compression_algorithm,
        CompressionAlgorithm::None
    ));
}
