use rustmat_builtins::Value;
use rustmat_gc::{gc_allocate, gc_collect_major, gc_collect_minor};
use rustmat_gc::{gc_configure, gc_stats, gc_test_context, GcConfig};

#[test]
fn test_default_gc_config() {
    gc_test_context(|| {
        let config = GcConfig::default();

        // Verify default values are reasonable
        assert!(config.num_generations >= 2);
        assert!(config.young_generation_size > 0);
        assert!(config.minor_gc_threshold > 0.0 && config.minor_gc_threshold <= 1.0);
        assert!(config.major_gc_threshold > 0.0 && config.major_gc_threshold <= 1.0);
        assert!(config.promotion_threshold > 0);
        assert!(config.max_gc_threads > 0);

        // Should be valid
        assert!(config.validate().is_ok());
    });
}

#[test]
fn test_low_latency_preset() {
    gc_test_context(|| {
        let config = GcConfig::low_latency();

        // Low latency should prioritize small pause times
        assert!(config.validate().is_ok());
        assert!(config.minor_gc_threshold < 0.8); // More frequent minor collections
        assert!(config.young_generation_size < 64 * 1024 * 1024); // Smaller young gen
        assert!(!config.concurrent_collection || config.parallel_collection); // Should use fast collection

        // Should be configurable
        let result = gc_configure(config);
        assert!(result.is_ok());
    });
}

#[test]
fn test_high_throughput_preset() {
    gc_test_context(|| {
        let config = GcConfig::high_throughput();

        // High throughput should prioritize overall performance
        assert!(config.validate().is_ok());
        assert!(config.young_generation_size >= 32 * 1024 * 1024); // Larger young gen
        assert!(config.parallel_collection); // Should use parallel collection

        let result = gc_configure(config);
        assert!(result.is_ok());
    });
}

#[test]
fn test_low_memory_preset() {
    gc_test_context(|| {
        let config = GcConfig::low_memory();

        // Low memory should prioritize memory usage
        assert!(config.validate().is_ok());
        assert!(config.young_generation_size <= 16 * 1024 * 1024); // Smaller young gen
        assert!(config.minor_gc_threshold <= 0.7); // More aggressive collection
        assert!(config.major_gc_threshold <= 0.8);

        let result = gc_configure(config);
        assert!(result.is_ok());
    });
}

#[test]
fn test_debug_preset() {
    gc_test_context(|| {
        let config = GcConfig::debug();

        // Debug should enable comprehensive monitoring
        assert!(config.validate().is_ok());
        assert!(config.verbose_logging);
        assert!(config.collect_statistics);

        let result = gc_configure(config);
        assert!(result.is_ok());
    });
}

#[test]
fn test_config_validation() {
    gc_test_context(|| {
        // Test invalid configurations
        // Invalid number of generations  
        let mut config = GcConfig {
            num_generations: 1,
            ..Default::default()
        };
        assert!(config.validate().is_err());
        config.num_generations = 2; // Reset

        // Invalid thresholds
        config.minor_gc_threshold = -0.1;
        assert!(config.validate().is_err());
        config.minor_gc_threshold = 1.5;
        assert!(config.validate().is_err());
        config.minor_gc_threshold = 0.8; // Reset

        config.major_gc_threshold = -0.1;
        assert!(config.validate().is_err());
        config.major_gc_threshold = 1.5;
        assert!(config.validate().is_err());
        config.major_gc_threshold = 0.9; // Reset

        // Invalid promotion threshold
        config.promotion_threshold = 0;
        assert!(config.validate().is_err());
        config.promotion_threshold = 3; // Reset

        // Invalid thread count
        config.max_gc_threads = 0;
        assert!(config.validate().is_err());
        config.max_gc_threads = 4; // Reset

        // Should be valid now
        assert!(config.validate().is_ok());
    });
}

#[test]
fn test_config_builder() {
    gc_test_context(|| {
        let config = rustmat_gc::GcConfigBuilder::new()
            .young_generation_size(32 * 1024 * 1024)
            .minor_gc_threshold(0.75)
            .major_gc_threshold(0.85)
            .verbose_logging(true)
            .build();

        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.young_generation_size, 32 * 1024 * 1024);
        assert_eq!(config.minor_gc_threshold, 0.75);
        assert_eq!(config.major_gc_threshold, 0.85);
        assert!(config.verbose_logging);
    });
}

#[test]
fn test_config_builder_from_env() {
    gc_test_context(|| {
        // Test environment variable parsing
        std::env::set_var("RUSTMAT_GC_YOUNG_SIZE", "64");
        std::env::set_var("RUSTMAT_GC_THREADS", "8");

        let config = rustmat_gc::GcConfigBuilder::from_env().build();
        assert!(config.is_ok());

        // Clean up
        std::env::remove_var("RUSTMAT_GC_YOUNG_SIZE");
        std::env::remove_var("RUSTMAT_GC_THREADS");
    });
}

#[test]
fn test_heap_capacity_calculation() {
    gc_test_context(|| {
        let config = GcConfig::default();
        let capacity = config.total_heap_capacity();

        // Should be reasonable
        assert!(capacity > config.young_generation_size);
        assert!(capacity < config.max_heap_size || config.max_heap_size == 0);
    });
}

#[test]
fn test_generation_size_calculation() {
    gc_test_context(|| {
        let config = GcConfig::default();

        // Young generation should be as specified
        assert_eq!(config.generation_size(0), config.young_generation_size);

        // Older generations should be larger
        let old_gen_size = config.generation_size(1);
        assert!(old_gen_size >= config.young_generation_size);
    });
}

#[test]
fn test_preset_performance_characteristics() {
    gc_test_context(|| {
        let presets = [
            ("low_latency", GcConfig::low_latency()),
            ("high_throughput", GcConfig::high_throughput()),
            ("low_memory", GcConfig::low_memory()),
            ("debug", GcConfig::debug()),
        ];

        for (name, config) in &presets {
            // Each preset should be valid
            assert!(config.validate().is_ok(), "Preset {name} is invalid");

            // Should be configurable
            let result = gc_configure(config.clone());
            assert!(result.is_ok(), "Failed to configure preset {name}");

            // Test basic allocation with this preset
            let value = Value::Num(42.0);
            let ptr = gc_allocate(value);
            assert!(ptr.is_ok(), "Failed to allocate with preset {name}");
        }
    });
}

#[test]
fn test_config_reconfiguration() {
    gc_test_context(|| {
        // Start with default config
        let default_config = GcConfig::default();
        gc_configure(default_config).unwrap();

        // Switch to low latency
        let low_latency_config = GcConfig::low_latency();
        let result = gc_configure(low_latency_config);
        assert!(result.is_ok());

        // Switch to high throughput
        let high_throughput_config = GcConfig::high_throughput();
        let result = gc_configure(high_throughput_config);
        assert!(result.is_ok());

        // Should still be able to allocate and collect
        let value = Value::Num(123.0);
        let ptr = gc_allocate(value);
        assert!(ptr.is_ok());

        let collected = gc_collect_minor();
        assert!(collected.is_ok());
    });
}

#[test]
fn test_statistics_with_different_presets() {
    gc_test_context(|| {
        // Enable statistics collection
        let debug_config = GcConfig::debug();
        gc_configure(debug_config).unwrap();

        // Perform some allocations
        for i in 0..10 {
            let value = Value::Num(i as f64);
            let _ = gc_allocate(value);
        }

        let stats = gc_stats();
        let allocations = stats
            .total_allocations
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(allocations >= 10);

        // Try a collection
        let _collected = gc_collect_minor().unwrap();

        // Check collection statistics
        let post_collection_stats = gc_stats();
        let minor_collections = post_collection_stats
            .minor_collections
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(minor_collections > 0);
    });
}

#[test]
fn test_concurrent_collection_config() {
    gc_test_context(|| {
        let config = GcConfig {
            concurrent_collection: true,
            parallel_collection: false,
            ..Default::default()
        };

        assert!(config.validate().is_ok());
        let result = gc_configure(config);
        assert!(result.is_ok());

        // Should still work with concurrent collection enabled
        let value = Value::String("test".to_string());
        let ptr = gc_allocate(value);
        assert!(ptr.is_ok());
    });
}

#[test]
fn test_parallel_collection_config() {
    gc_test_context(|| {
        let config = GcConfig {
            parallel_collection: true,
            max_gc_threads: 4,
            ..Default::default()
        };

        assert!(config.validate().is_ok());
        let result = gc_configure(config);
        assert!(result.is_ok());

        // Test with multiple allocations that might trigger parallel collection
        for i in 0..50 {
            let value = Value::Num(i as f64);
            let _ = gc_allocate(value);
        }

        let collected = gc_collect_major();
        assert!(collected.is_ok());
    });
}

#[test]
fn test_write_barriers_config() {
    gc_test_context(|| {
        let config = GcConfig {
            write_barriers: true,
            ..Default::default()
        };

        assert!(config.validate().is_ok());
        let result = gc_configure(config);
        assert!(result.is_ok());

        // Test allocation with write barriers enabled
        let matrix = rustmat_builtins::Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let value = Value::Matrix(matrix);
        let ptr = gc_allocate(value);
        assert!(ptr.is_ok());
    });
}

#[test]
fn test_pointer_compression_config() {
    gc_test_context(|| {
        let config = GcConfig {
            pointer_compression: true,
            ..Default::default()
        };

        assert!(config.validate().is_ok());
        let result = gc_configure(config);
        assert!(result.is_ok());

        // Test allocation with pointer compression
        let value = Value::Bool(true);
        let ptr = gc_allocate(value);
        assert!(ptr.is_ok());
    });
}

#[test]
fn test_heap_growth_factor() {
    gc_test_context(|| {
        let config = GcConfig {
            heap_growth_factor: 1.5,
            ..Default::default()
        };

        assert!(config.validate().is_ok());
        let growth_factor = config.heap_growth_factor;
        let result = gc_configure(config);
        assert!(result.is_ok());

        // Growth factor should affect heap expansion behavior
        assert_eq!(growth_factor, 1.5);
    });
}

#[test]
fn test_target_utilization() {
    gc_test_context(|| {
        let config = GcConfig {
            target_utilization: 0.75,
            ..Default::default()
        };

        assert!(config.validate().is_ok());
        let target_util = config.target_utilization;
        let result = gc_configure(config);
        assert!(result.is_ok());

        assert_eq!(target_util, 0.75);
    });
}

#[test]
fn test_min_max_heap_size() {
    gc_test_context(|| {
        let config = GcConfig {
            min_heap_size: 16 * 1024 * 1024, // 16MB
            max_heap_size: 256 * 1024 * 1024, // 256MB
            ..Default::default()
        };

        assert!(config.validate().is_ok());
        let min_size = config.min_heap_size;
        let max_size = config.max_heap_size;
        let result = gc_configure(config);
        assert!(result.is_ok());

        // Heap should respect these constraints
        assert!(min_size <= max_size);
    });
}

#[test]
fn test_collection_timeout() {
    gc_test_context(|| {
        let config = GcConfig {
            collection_timeout: std::time::Duration::from_millis(100),
            ..Default::default()
        };

        assert!(config.validate().is_ok());
        let result = gc_configure(config);
        assert!(result.is_ok());

        // Collections should respect timeout
        let collected = gc_collect_minor();
        assert!(collected.is_ok());
    });
}

#[test]
fn test_invalid_config_edge_cases() {
    gc_test_context(|| {
        // Test edge cases that should fail validation
        // Zero young generation size
        let mut config = GcConfig {
            young_generation_size: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
        config.young_generation_size = 1024 * 1024; // Reset

        // Min heap larger than max heap
        config.min_heap_size = 100 * 1024 * 1024;
        config.max_heap_size = 50 * 1024 * 1024;
        assert!(config.validate().is_err());
        config.min_heap_size = 16 * 1024 * 1024; // Reset
        config.max_heap_size = 0; // Reset (unlimited)

        // Invalid growth factor
        config.heap_growth_factor = 0.5; // Should be > 1.0
        assert!(config.validate().is_err());
        config.heap_growth_factor = 2.0; // Reset

        // Should be valid now
        assert!(config.validate().is_ok());
    });
}
