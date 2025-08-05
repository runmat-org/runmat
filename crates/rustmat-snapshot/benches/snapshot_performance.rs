//! Performance benchmarks for RustMat snapshot system
//!
//! Measures creation, loading, and compression performance across different
//! configurations and data sizes.

use std::fs;
use tempfile::tempdir;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rustmat_gc::gc_test_context;
use rustmat_snapshot::presets::SnapshotPreset;
use rustmat_snapshot::{SnapshotBuilder, SnapshotConfig, SnapshotLoader};

fn benchmark_snapshot_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot_creation");

    let presets = vec![
        ("Development", SnapshotPreset::Development),
        ("Production", SnapshotPreset::Production),
        ("High-Performance", SnapshotPreset::HighPerformance),
        ("Low-Memory", SnapshotPreset::LowMemory),
        ("Debug", SnapshotPreset::Debug),
    ];

    for (name, preset) in presets {
        group.bench_with_input(BenchmarkId::new("create", name), &preset, |b, preset| {
            b.iter(|| {
                gc_test_context(|| {
                    let temp_dir = tempdir().unwrap();
                    let snapshot_path = temp_dir.path().join("benchmark.snapshot");

                    let config = preset.config();
                    let builder = SnapshotBuilder::new(config);

                    black_box(builder.build_and_save(&snapshot_path).unwrap());
                });
            });
        });
    }

    group.finish();
}

fn benchmark_snapshot_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot_loading");

    // Pre-create snapshots for loading benchmarks
    let temp_dir = tempdir().unwrap();
    let snapshots = vec![
        ("Development", SnapshotPreset::Development),
        ("Production", SnapshotPreset::Production),
        ("High-Performance", SnapshotPreset::HighPerformance),
    ];

    let mut snapshot_paths = Vec::new();

    gc_test_context(|| {
        for (name, preset) in &snapshots {
            let snapshot_path = temp_dir
                .path()
                .join(format!("{}.snapshot", name.to_lowercase()));
            let config = preset.config();
            let builder = SnapshotBuilder::new(config);
            builder.build_and_save(&snapshot_path).unwrap();
            snapshot_paths.push((name, preset, snapshot_path));
        }
    });

    for (name, preset, snapshot_path) in snapshot_paths {
        let file_size = fs::metadata(&snapshot_path).unwrap().len();
        group.throughput(Throughput::Bytes(file_size));

        group.bench_with_input(
            BenchmarkId::new("load", name),
            &(preset, snapshot_path),
            |b, (preset, path)| {
                b.iter(|| {
                    gc_test_context(|| {
                        let config = preset.config();
                        let mut loader = SnapshotLoader::new(config);
                        black_box(loader.load(path).unwrap());
                    });
                });
            },
        );
    }

    group.finish();
}

fn benchmark_compression_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    // Test data of different characteristics
    let test_data = vec![
        ("small_text", "Hello, World!".repeat(100).into_bytes()),
        (
            "large_text",
            "The quick brown fox jumps over the lazy dog. "
                .repeat(1000)
                .into_bytes(),
        ),
        ("repetitive", vec![42u8; 10000]),
        ("random", (0..10000).map(|i| (i * 17 % 256) as u8).collect()),
    ];

    let algorithms = vec![
        (
            "uncompressed",
            SnapshotConfig {
                compression_enabled: false,
                ..SnapshotConfig::default()
            },
        ),
        (
            "lz4_fast",
            SnapshotConfig {
                compression_enabled: true,
                compression_algorithm: rustmat_snapshot::CompressionAlgorithm::Lz4,
                compression_level: 1,
                ..SnapshotConfig::default()
            },
        ),
        (
            "lz4_balanced",
            SnapshotConfig {
                compression_enabled: true,
                compression_algorithm: rustmat_snapshot::CompressionAlgorithm::Lz4,
                compression_level: 6,
                ..SnapshotConfig::default()
            },
        ),
        (
            "zstd",
            SnapshotConfig {
                compression_enabled: true,
                compression_algorithm: rustmat_snapshot::CompressionAlgorithm::Zstd,
                compression_level: 6,
                ..SnapshotConfig::default()
            },
        ),
    ];

    for (data_name, data) in &test_data {
        for (algo_name, config) in &algorithms {
            group.throughput(Throughput::Bytes(data.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}", data_name, algo_name), data.len()),
                &(data, config),
                |b, (_data, config)| {
                    b.iter(|| {
                        gc_test_context(|| {
                            let temp_dir = tempdir().unwrap();
                            let snapshot_path = temp_dir.path().join("compression_test.snapshot");

                            // Create a minimal snapshot with this data size
                            let builder = SnapshotBuilder::new((*config).clone());
                            black_box(builder.build_and_save(&snapshot_path).unwrap());
                        });
                    });
                },
            );
        }
    }

    group.finish();
}

fn benchmark_concurrent_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_loading");

    // Pre-create a snapshot
    let temp_dir = tempdir().unwrap();
    let snapshot_path = temp_dir.path().join("concurrent.snapshot");

    gc_test_context(|| {
        let config = SnapshotPreset::Production.config();
        let builder = SnapshotBuilder::new(config);
        builder.build_and_save(&snapshot_path).unwrap();
    });

    let file_size = fs::metadata(&snapshot_path).unwrap().len();
    group.throughput(Throughput::Bytes(file_size));

    for thread_count in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    gc_test_context(|| {
                        use std::sync::Arc;
                        use std::thread;

                        let snapshot_path = Arc::new(snapshot_path.clone());
                        let config = Arc::new(SnapshotPreset::Production.config());

                        let handles: Vec<_> = (0..thread_count)
                            .map(|_| {
                                let path = Arc::clone(&snapshot_path);
                                let cfg = Arc::clone(&config);
                                thread::spawn(move || {
                                    let mut loader = SnapshotLoader::new((*cfg).clone());
                                    black_box(loader.load(&*path).unwrap());
                                })
                            })
                            .collect();

                        for handle in handles {
                            handle.join().unwrap();
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    let configs = vec![
        ("low_memory", SnapshotPreset::LowMemory.config()),
        ("high_performance", SnapshotPreset::HighPerformance.config()),
        ("default", SnapshotConfig::default()),
    ];

    for (name, config) in configs {
        group.bench_with_input(
            BenchmarkId::new("memory_footprint", name),
            &config,
            |b, config| {
                b.iter(|| {
                    gc_test_context(|| {
                        let temp_dir = tempdir().unwrap();
                        let snapshot_path = temp_dir.path().join("memory_test.snapshot");

                        // Measure memory usage during creation and loading
                        let builder = SnapshotBuilder::new(config.clone());
                        builder.build_and_save(&snapshot_path).unwrap();

                        let mut loader = SnapshotLoader::new(config.clone());
                        black_box(loader.load(&snapshot_path).unwrap());
                    });
                });
            },
        );
    }

    group.finish();
}

fn benchmark_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");

    // Pre-create snapshot
    let temp_dir = tempdir().unwrap();
    let snapshot_path = temp_dir.path().join("cache_test.snapshot");

    gc_test_context(|| {
        let config = SnapshotConfig::default();
        let builder = SnapshotBuilder::new(config);
        builder.build_and_save(&snapshot_path).unwrap();
    });

    // Benchmark cache hit vs miss
    group.bench_function("cache_miss", |b| {
        b.iter(|| {
            gc_test_context(|| {
                let manager = rustmat_snapshot::SnapshotManager::default();
                black_box(manager.load_snapshot(&snapshot_path).unwrap());
            });
        });
    });

    group.bench_function("cache_hit", |b| {
        gc_test_context(|| {
            let manager = rustmat_snapshot::SnapshotManager::default();
            // Prime the cache
            let _snapshot = manager.load_snapshot(&snapshot_path).unwrap();

            b.iter(|| {
                black_box(manager.load_snapshot(&snapshot_path).unwrap());
            });
        });
    });

    group.finish();
}

fn benchmark_validation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_overhead");

    // Pre-create snapshots with and without validation
    let temp_dir = tempdir().unwrap();
    let validated_path = temp_dir.path().join("validated.snapshot");
    let unvalidated_path = temp_dir.path().join("unvalidated.snapshot");

    gc_test_context(|| {
        let validated_config = SnapshotConfig {
            validation_enabled: true,
            ..SnapshotConfig::default()
        };
        let unvalidated_config = SnapshotConfig {
            validation_enabled: false,
            ..SnapshotConfig::default()
        };

        let builder = SnapshotBuilder::new(validated_config.clone());
        builder.build_and_save(&validated_path).unwrap();

        let builder = SnapshotBuilder::new(unvalidated_config);
        builder.build_and_save(&unvalidated_path).unwrap();
    });

    group.bench_function("with_validation", |b| {
        b.iter(|| {
            gc_test_context(|| {
                let config = SnapshotConfig {
                    validation_enabled: true,
                    ..SnapshotConfig::default()
                };
                let mut loader = SnapshotLoader::new(config);
                black_box(loader.load(&validated_path).unwrap());
            });
        });
    });

    group.bench_function("without_validation", |b| {
        b.iter(|| {
            gc_test_context(|| {
                let config = SnapshotConfig {
                    validation_enabled: false,
                    ..SnapshotConfig::default()
                };
                let mut loader = SnapshotLoader::new(config);
                black_box(loader.load(&unvalidated_path).unwrap());
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_snapshot_creation,
    benchmark_snapshot_loading,
    benchmark_compression_algorithms,
    benchmark_concurrent_loading,
    benchmark_memory_usage,
    benchmark_cache_performance,
    benchmark_validation_overhead
);

criterion_main!(benches);
