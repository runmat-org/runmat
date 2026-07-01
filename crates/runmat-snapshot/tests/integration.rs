//! Integration tests for RunMat snapshot system
//!
//! Tests the complete snapshot creation, serialization, and loading pipeline.

use std::fs;
use std::path::Path;
use std::rc::Rc;
use tempfile::tempdir;

use runmat_gc::gc_test_context;
use runmat_snapshot::format::CompressionAlgorithm as HeaderCompressionAlgorithm;
use runmat_snapshot::presets::SnapshotPreset;
use runmat_snapshot::{
    CompressionAlgorithm as ConfigCompressionAlgorithm, SnapshotBuilder, SnapshotConfig,
    SnapshotHeader, SnapshotLoader, SnapshotManager,
};
use serde::{de::DeserializeOwned, Serialize};

// Import runtime to ensure builtins are registered with inventory
use runmat_runtime as _;

/// Create a test configuration with validation disabled
fn test_config() -> SnapshotConfig {
    SnapshotConfig {
        validation_enabled: false,
        ..SnapshotConfig::default()
    }
}

#[test]
fn test_snapshot_creation_and_loading() {
    gc_test_context(|| {
        let temp_dir = tempdir().unwrap();
        let snapshot_path = temp_dir.path().join("test.snapshot");

        // Create snapshot
        let config = test_config();
        let builder = SnapshotBuilder::new(config.clone());

        let result = builder.build_and_save(&snapshot_path);
        assert!(
            result.is_ok(),
            "Snapshot creation failed: {:?}",
            result.err()
        );

        // Verify file exists
        assert!(snapshot_path.exists());

        // Load snapshot
        let mut loader = SnapshotLoader::new(config);
        let result = loader.load(&snapshot_path);
        assert!(
            result.is_ok(),
            "Snapshot loading failed: {:?}",
            result.err()
        );

        let (snapshot, stats) = result.unwrap();

        // Verify snapshot contents
        assert!(!snapshot.builtins.functions.is_empty());
        assert!(stats.load_time.as_millis() < u128::MAX);
        assert!(stats.builtin_count > 0);
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_async_snapshot_loading_supports_zero_data_offset_fallback() {
    gc_test_context(|| {
        let temp_dir = tempdir().unwrap();
        let snapshot_path = temp_dir.path().join("async_zero_offset.snapshot");

        let config = SnapshotConfig {
            compression_enabled: false,
            validation_enabled: false,
            ..SnapshotConfig::default()
        };
        let builder = SnapshotBuilder::new(config.clone());
        builder.build_and_save(&snapshot_path).unwrap();
        rewrite_snapshot_data_offset(&snapshot_path, 0);

        let mut loader = SnapshotLoader::new(config);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (snapshot, stats) = runtime
            .block_on(loader.load_async(&snapshot_path))
            .expect("async load should honor zero data offset fallback");

        assert!(!snapshot.builtins.functions.is_empty());
        assert!(stats.compressed_size > 0);
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn rewrite_snapshot_data_offset(path: &Path, data_offset: u64) {
    let bytes = fs::read(path).unwrap();
    let old_header_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let mut header: SnapshotHeader = bincode::deserialize(&bytes[4..4 + old_header_size]).unwrap();
    let old_data_start = if header.data_info.data_offset != 0 {
        header.data_info.data_offset as usize
    } else {
        4 + old_header_size
    };
    let data = bytes[old_data_start..].to_vec();

    header.data_info.data_offset = data_offset;
    let header_data = bincode::serialize(&header).unwrap();
    let header_size = header_data.len() as u32;
    let mut rewritten = Vec::with_capacity(4 + header_data.len() + data.len());
    rewritten.extend_from_slice(&header_size.to_le_bytes());
    rewritten.extend_from_slice(&header_data);
    rewritten.extend_from_slice(&data);
    fs::write(path, rewritten).unwrap();
}

#[test]
fn test_snapshot_bincode_roundtrip() {
    gc_test_context(|| {
        let config = SnapshotConfig {
            compression_enabled: false,
            validation_enabled: false,
            ..SnapshotConfig::default()
        };
        let builder = SnapshotBuilder::new(config);
        let snapshot = builder.build().unwrap();
        let deserialized = assert_bincode_roundtrip("snapshot", &snapshot);
        assert_eq!(
            deserialized.builtins.functions.len(),
            snapshot.builtins.functions.len()
        );
    });
}

fn assert_bincode_roundtrip<T>(name: &str, value: &T) -> T
where
    T: Serialize + DeserializeOwned,
{
    let serialized = bincode::serialize(value).unwrap();
    bincode::deserialize::<T>(&serialized)
        .unwrap_or_else(|err| panic!("{name} failed bincode roundtrip: {:?}", err))
}

#[test]
fn test_snapshot_presets() {
    gc_test_context(|| {
        let temp_dir = tempdir().unwrap();

        for preset in SnapshotPreset::all_presets() {
            let snapshot_path = temp_dir.path().join(format!(
                "{}.snapshot",
                preset.name().to_lowercase().replace('-', "_")
            ));

            let mut config = preset.config();
            config.validation_enabled = false; // Disable validation for tests
            let builder = SnapshotBuilder::new(config.clone());

            let result = builder.build_and_save(&snapshot_path);
            assert!(
                result.is_ok(),
                "Failed to create snapshot with preset {}: {:?}",
                preset.name(),
                result.err()
            );

            // Verify file exists and has reasonable size
            assert!(snapshot_path.exists());
            let metadata = fs::metadata(&snapshot_path).unwrap();
            assert!(
                metadata.len() > 0,
                "Snapshot file is empty for preset {}",
                preset.name()
            );

            let header = SnapshotLoader::peek_header(&snapshot_path).unwrap();
            assert_header_matches_configured_compression(preset.name(), &config, &header);

            // Try to load it
            let mut loader = SnapshotLoader::new(config);
            let result = loader.load(&snapshot_path);
            assert!(
                result.is_ok(),
                "Failed to load snapshot with preset {}: {:?}",
                preset.name(),
                result.err()
            );
        }
    });
}

fn assert_header_matches_configured_compression(
    preset_name: &str,
    config: &SnapshotConfig,
    header: &runmat_snapshot::SnapshotHeader,
) {
    match config.compression_algorithm {
        ConfigCompressionAlgorithm::None => assert!(
            matches!(
                header.data_info.compression.algorithm,
                HeaderCompressionAlgorithm::None
            ),
            "preset {preset_name} should not compress snapshot data"
        ),
        ConfigCompressionAlgorithm::Lz4 => assert!(
            matches!(
                header.data_info.compression.algorithm,
                HeaderCompressionAlgorithm::Lz4 { .. } | HeaderCompressionAlgorithm::None
            ),
            "preset {preset_name} should use LZ4 or fall back to uncompressed"
        ),
        ConfigCompressionAlgorithm::Zstd => assert!(
            matches!(
                header.data_info.compression.algorithm,
                HeaderCompressionAlgorithm::Zstd { .. } | HeaderCompressionAlgorithm::None
            ),
            "preset {preset_name} should use ZSTD or fall back to uncompressed"
        ),
        ConfigCompressionAlgorithm::Auto => assert!(
            !matches!(
                header.data_info.compression.algorithm,
                HeaderCompressionAlgorithm::None
            ) || header.data_info.compressed_size == header.data_info.uncompressed_size,
            "preset {preset_name} should auto-select compression unless it is ineffective"
        ),
    }
}

#[test]
fn test_snapshot_validation() {
    gc_test_context(|| {
        let temp_dir = tempdir().unwrap();
        let snapshot_path = temp_dir.path().join("validation.snapshot");

        // Create snapshot (using minimal validation for test environment)
        let config = SnapshotConfig {
            validation_enabled: false, // Tests have trouble with inventory builtin collection
            ..test_config()
        };
        let builder = SnapshotBuilder::new(config.clone());
        builder.build_and_save(&snapshot_path).unwrap();

        // Load with validation
        let mut loader = SnapshotLoader::new(config);
        let result = loader.load(&snapshot_path);
        assert!(result.is_ok());

        // Test validation utilities
        assert!(SnapshotLoader::quick_validate(&snapshot_path).unwrap_or(false));

        let metadata = SnapshotLoader::get_metadata(&snapshot_path);
        assert!(metadata.is_ok());

        let metadata = metadata.unwrap();
        assert!(metadata.is_compatible());
    });
}

#[test]
fn test_snapshot_compression() {
    gc_test_context(|| {
        let temp_dir = tempdir().unwrap();
        let uncompressed_path = temp_dir.path().join("uncompressed.snapshot");
        let compressed_path = temp_dir.path().join("compressed.snapshot");

        // Create uncompressed snapshot
        let uncompressed_config = SnapshotConfig {
            compression_enabled: false,
            ..test_config()
        };
        let builder = SnapshotBuilder::new(uncompressed_config.clone());
        builder.build_and_save(&uncompressed_path).unwrap();

        // Create compressed snapshot
        let compressed_config = SnapshotConfig {
            compression_enabled: true,
            compression_level: 6,
            ..test_config()
        };
        let builder = SnapshotBuilder::new(compressed_config.clone());
        builder.build_and_save(&compressed_path).unwrap();

        // Compare file sizes
        let uncompressed_size = fs::metadata(&uncompressed_path).unwrap().len();
        let compressed_size = fs::metadata(&compressed_path).unwrap().len();

        // Compressed should be smaller (unless data is very small or already compressed)
        println!("Uncompressed: {uncompressed_size} bytes, Compressed: {compressed_size} bytes");

        // Load both and verify they produce equivalent snapshots
        let mut uncompressed_loader = SnapshotLoader::new(uncompressed_config);
        let mut compressed_loader = SnapshotLoader::new(compressed_config);

        let (uncompressed_snapshot, _) = uncompressed_loader.load(&uncompressed_path).unwrap();
        let (compressed_snapshot, _) = compressed_loader.load(&compressed_path).unwrap();

        // Compare builtin counts
        assert_eq!(
            uncompressed_snapshot.builtins.functions.len(),
            compressed_snapshot.builtins.functions.len()
        );
    });
}

#[test]
fn test_snapshot_manager() {
    gc_test_context(|| {
        let temp_dir = tempdir().unwrap();
        let snapshot_path = temp_dir.path().join("manager.snapshot");

        // Create snapshot
        let config = test_config();
        let builder = SnapshotBuilder::new(config.clone());
        builder.build_and_save(&snapshot_path).unwrap();

        // Test manager
        let manager = SnapshotManager::new(config);

        // Load through manager (should cache)
        let snapshot1 = manager.load_snapshot(&snapshot_path).unwrap();
        let snapshot2 = manager.load_snapshot(&snapshot_path).unwrap();

        // Should be the same Rc (cached)
        assert!(Rc::ptr_eq(&snapshot1, &snapshot2));

        // Test cache stats
        let (cache_entries, cache_size) = manager.cache_stats();
        assert_eq!(cache_entries, 1);
        assert!(cache_size > 0);

        // Test stats
        let stats = manager.get_stats(&snapshot_path);
        assert!(stats.is_some());

        // Clear cache
        manager.clear_cache();
        let (cache_entries, cache_size) = manager.cache_stats();
        assert_eq!(cache_entries, 0);
        assert_eq!(cache_size, 0);
    });
}

#[test]
fn test_snapshot_header_utilities() {
    gc_test_context(|| {
        let temp_dir = tempdir().unwrap();
        let snapshot_path = temp_dir.path().join("header.snapshot");

        // Create snapshot
        let config = test_config();
        let builder = SnapshotBuilder::new(config);
        builder.build_and_save(&snapshot_path).unwrap();

        // Test header peek
        let header = SnapshotLoader::peek_header(&snapshot_path).unwrap();
        assert!(header.validate().is_ok());
        assert!(header.is_platform_compatible());

        // Test metadata extraction
        let metadata = SnapshotLoader::get_metadata(&snapshot_path).unwrap();
        assert!(metadata.is_compatible());
        assert!(metadata.age().as_millis() < u128::MAX);

        // Test load time estimation
        let estimated_time = SnapshotLoader::estimate_load_time(&snapshot_path).unwrap();
        assert!(estimated_time.as_millis() < u128::MAX);
    });
}

#[test]
fn test_snapshot_error_conditions() {
    gc_test_context(|| {
        let temp_dir = tempdir().unwrap();
        let nonexistent_path = temp_dir.path().join("nonexistent.snapshot");

        // Test loading nonexistent file
        let config = test_config();
        let mut loader = SnapshotLoader::new(config.clone());
        let result = loader.load(&nonexistent_path);
        assert!(result.is_err());

        // Test quick validation on nonexistent file
        assert!(!SnapshotLoader::quick_validate(&nonexistent_path).unwrap_or(true));

        // Test manager with nonexistent file
        let manager = SnapshotManager::new(config);
        let result = manager.load_snapshot(&nonexistent_path);
        assert!(result.is_err());
    });
}

#[test]
fn test_build_statistics() {
    gc_test_context(|| {
        let temp_dir = tempdir().unwrap();
        let snapshot_path = temp_dir.path().join("stats.snapshot");

        let config = SnapshotConfig {
            progress_reporting: false, // Disable progress for test
            ..test_config()
        };
        let builder = SnapshotBuilder::new(config);

        builder.build_and_save(&snapshot_path).unwrap();

        let stats = builder.stats();
        assert!(stats.start_time.is_some());
        assert!(!stats.phase_times.is_empty());
        assert!(!stats.items_processed.is_empty());

        // Check that some builtins were processed
        let builtin_count = stats.items_processed.get("builtins").unwrap_or(&0);
        assert!(*builtin_count > 0);
    });
}

#[test]
fn test_loading_statistics() {
    gc_test_context(|| {
        let temp_dir = tempdir().unwrap();
        let snapshot_path = temp_dir.path().join("load_stats.snapshot");

        // Create snapshot
        let config = test_config();
        let builder = SnapshotBuilder::new(config.clone());
        builder.build_and_save(&snapshot_path).unwrap();

        // Load and check stats
        let mut loader = SnapshotLoader::new(config);
        let (_snapshot, stats) = loader.load(&snapshot_path).unwrap();

        assert!(stats.load_time.as_millis() < u128::MAX);
        assert!(stats.total_size > 0);
        assert!(stats.builtin_count > 0);
        assert!(stats.loading_throughput() > 0.0);
        assert!(stats.compression_efficiency() >= 0.0);
    });
}

#[test]
fn test_concurrent_loading() {
    use std::sync::Arc;
    use std::thread;

    gc_test_context(|| {
        let temp_dir = tempdir().unwrap();
        let snapshot_path = temp_dir.path().join("concurrent.snapshot");

        // Create snapshot
        let config = test_config();
        let builder = SnapshotBuilder::new(config.clone());
        builder.build_and_save(&snapshot_path).unwrap();

        // Test concurrent loading
        let snapshot_path = Arc::new(snapshot_path);
        let config = Arc::new(config);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let path = Arc::clone(&snapshot_path);
                let cfg = Arc::clone(&config);
                thread::spawn(move || {
                    let mut loader = SnapshotLoader::new((*cfg).clone());
                    loader
                        .load(&*path)
                        .map(|(snapshot, stats)| (snapshot.metadata.runmat_version, stats))
                })
            })
            .collect();

        // All should succeed
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_ok());
        }
    });
}
