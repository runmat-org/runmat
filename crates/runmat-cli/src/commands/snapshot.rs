use anyhow::{Context, Result};
use log::{error, info};
use runmat_snapshot::presets::SnapshotPreset;
use runmat_snapshot::{SnapshotBuilder, SnapshotConfig, SnapshotLoader};

use crate::cli::{CompressionAlg, OptLevel, SnapshotCommand};

pub async fn execute_snapshot_command(snapshot_command: SnapshotCommand) -> Result<()> {
    match snapshot_command {
        SnapshotCommand::Create {
            output,
            optimization,
            compression,
        } => {
            info!("Creating snapshot: {output:?}");

            let mut config = SnapshotConfig::default();
            if let Some(comp) = compression {
                config.compression_enabled = !matches!(comp, CompressionAlg::None);
                config.compression_algorithm = comp.into();
            }

            let _optimization_level = match optimization {
                OptLevel::None => runmat_snapshot::OptimizationLevel::None,
                OptLevel::Size => runmat_snapshot::OptimizationLevel::Basic,
                OptLevel::Speed => runmat_snapshot::OptimizationLevel::Aggressive,
                OptLevel::Aggressive => runmat_snapshot::OptimizationLevel::MaxPerformance,
            };

            let builder = SnapshotBuilder::new(config);
            builder
                .build_and_save(&output)
                .with_context(|| format!("Failed to build and save snapshot to {output:?}"))?;

            println!("Snapshot created successfully: {output:?}");
        }
        SnapshotCommand::Info { snapshot } => {
            info!("Loading snapshot info: {snapshot:?}");

            let mut loader = SnapshotLoader::new(SnapshotConfig::default());
            let (loaded, stats) = loader
                .load(&snapshot)
                .with_context(|| format!("Failed to load snapshot from {snapshot:?}"))?;

            println!("Snapshot Information:");
            println!("  File: {snapshot:?}");
            println!("  Version: {}", loaded.metadata.runmat_version);
            println!("  Created: {:?}", loaded.metadata.created_at);
            println!("  Tool Version: {}", loaded.metadata.tool_version);
            println!("  Build Config: {:?}", loaded.metadata.build_config);
            println!("  Builtin Functions: {}", loaded.builtins.functions.len());
            println!(
                "  HIR Cache Functions: {}",
                loaded.hir_cache.functions.len()
            );
            println!("  HIR Cache Patterns: {}", loaded.hir_cache.patterns.len());
            println!(
                "  Bytecode Cache (stdlib): {}",
                loaded.bytecode_cache.stdlib_bytecode.len()
            );
            println!(
                "  Bytecode Cache (sequences): {}",
                loaded.bytecode_cache.operation_sequences.len()
            );
            println!(
                "  Bytecode Cache (hotspots): {}",
                loaded.bytecode_cache.hotspots.len()
            );
            println!("  GC Presets: {}", loaded.gc_presets.presets.len());
            println!("  Load Time: {:?}", stats.load_time);
            println!("  Total Size: {} bytes", stats.total_size);
            println!("  Compressed Size: {} bytes", stats.compressed_size);
            println!("  Compression Ratio: {:.2}x", stats.compression_ratio);
        }
        SnapshotCommand::Presets => {
            println!("Available Snapshot Presets:");
            println!();

            let presets = vec![
                (
                    "development",
                    SnapshotPreset::Development,
                    "Fast development iteration",
                ),
                (
                    "production",
                    SnapshotPreset::Production,
                    "Production deployment",
                ),
                (
                    "high-performance",
                    SnapshotPreset::HighPerformance,
                    "High-performance computing",
                ),
                (
                    "low-memory",
                    SnapshotPreset::LowMemory,
                    "Memory-constrained environments",
                ),
                (
                    "network-optimized",
                    SnapshotPreset::NetworkOptimized,
                    "Network-optimized (minimal size)",
                ),
                (
                    "debug",
                    SnapshotPreset::Debug,
                    "Debug-friendly (maximum validation)",
                ),
            ];

            for (name, preset, description) in presets {
                let config = preset.config();
                println!("  {name}");
                println!("    Description: {description}");
                println!("    Compression: {:?}", config.compression_algorithm);
                println!(
                    "    Validation: {}",
                    if config.validation_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!(
                    "    Memory Mapping: {}",
                    if config.memory_mapping_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!(
                    "    Parallel Loading: {}",
                    if config.parallel_loading {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!();
            }
        }
        SnapshotCommand::Validate { snapshot } => {
            info!("Validating snapshot: {snapshot:?}");

            let mut loader = SnapshotLoader::new(SnapshotConfig::default());
            match loader.load(&snapshot) {
                Ok((_, stats)) => {
                    println!("Snapshot validation passed: {snapshot:?}");
                    println!("  Load time: {:?}", stats.load_time);
                    println!("  File size: {} bytes", stats.total_size);
                    if stats.compressed_size > 0 {
                        println!("  Compressed size: {} bytes", stats.compressed_size);
                        println!("  Compression ratio: {:.2}x", stats.compression_ratio);
                    }
                }
                Err(e) => {
                    error!("Snapshot validation failed: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
    Ok(())
}
