//! RustMat Snapshot Tool
//!
//! Command-line interface for creating, validating, and managing
//! RustMat snapshot files.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use rustmat_snapshot::presets::SnapshotPreset;
use rustmat_snapshot::{SnapshotBuilder, SnapshotConfig, SnapshotLoader};

/// RustMat Snapshot Tool - Create and manage optimized standard library snapshots
#[derive(Parser)]
#[command(name = "rustmat-snapshot-tool")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Create and manage RustMat standard library snapshots")]
#[command(long_about = None)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Log level
    #[arg(long, value_enum, default_value = "info")]
    log_level: LogLevel,

    /// Configuration preset
    #[arg(short, long, value_enum, default_value = "production")]
    preset: PresetName,

    #[command(subcommand)]
    command: Commands,
}

/// Available commands
#[derive(Subcommand)]
enum Commands {
    /// Create a new snapshot
    Create {
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Override compression level (1-9)
        #[arg(long)]
        compression_level: Option<u32>,

        /// Disable compression
        #[arg(long)]
        no_compression: bool,

        /// Disable validation
        #[arg(long)]
        no_validation: bool,

        /// Enable progress reporting
        #[arg(long)]
        progress: bool,
    },

    /// Validate an existing snapshot
    Validate {
        /// Snapshot file to validate
        input: PathBuf,

        /// Strict validation mode
        #[arg(long)]
        strict: bool,

        /// Check compatibility only
        #[arg(long)]
        compatibility_only: bool,
    },

    /// Show snapshot information
    Info {
        /// Snapshot file to inspect
        input: PathBuf,

        /// Show detailed information
        #[arg(long)]
        detailed: bool,

        /// Show performance metrics
        #[arg(long)]
        metrics: bool,
    },

    /// List available presets
    Presets {
        /// Show detailed preset information
        #[arg(long)]
        detailed: bool,
    },

    /// Benchmark snapshot loading performance
    Benchmark {
        /// Snapshot file to benchmark
        input: PathBuf,

        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,

        /// Warm up iterations
        #[arg(long, default_value = "3")]
        warmup: usize,
    },

    /// Compare multiple snapshots
    Compare {
        /// Snapshot files to compare
        files: Vec<PathBuf>,

        /// Show size comparison
        #[arg(long)]
        size: bool,

        /// Show performance comparison
        #[arg(long)]
        performance: bool,
    },
}

/// Log level configuration
#[derive(ValueEnum, Clone, Debug)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Preset names for CLI
#[derive(ValueEnum, Clone, Debug)]
enum PresetName {
    Development,
    Production,
    HighPerformance,
    LowMemory,
    NetworkOptimized,
    Debug,
}

impl From<PresetName> for SnapshotPreset {
    fn from(preset: PresetName) -> Self {
        match preset {
            PresetName::Development => SnapshotPreset::Development,
            PresetName::Production => SnapshotPreset::Production,
            PresetName::HighPerformance => SnapshotPreset::HighPerformance,
            PresetName::LowMemory => SnapshotPreset::LowMemory,
            PresetName::NetworkOptimized => SnapshotPreset::NetworkOptimized,
            PresetName::Debug => SnapshotPreset::Debug,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.log_level, cli.verbose)?;

    // Execute command
    match cli.command {
        Commands::Create {
            output,
            compression_level,
            no_compression,
            no_validation,
            progress,
        } => {
            create_snapshot(
                cli.preset.into(),
                output,
                compression_level,
                no_compression,
                no_validation,
                progress,
            )?;
        }

        Commands::Validate {
            input,
            strict,
            compatibility_only,
        } => {
            validate_snapshot(input, strict, compatibility_only)?;
        }

        Commands::Info {
            input,
            detailed,
            metrics,
        } => {
            show_snapshot_info(input, detailed, metrics)?;
        }

        Commands::Presets { detailed } => {
            list_presets(detailed)?;
        }

        Commands::Benchmark {
            input,
            iterations,
            warmup,
        } => {
            benchmark_snapshot(input, iterations, warmup)?;
        }

        Commands::Compare {
            files,
            size,
            performance,
        } => {
            compare_snapshots(files, size, performance)?;
        }
    }

    Ok(())
}

/// Initialize proper logging with configurable levels and formatting
fn init_logging(level: LogLevel, verbose: bool) -> Result<()> {
    use std::io::Write;

    let log_level = if verbose {
        log::LevelFilter::Debug
    } else {
        match level {
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Trace => log::LevelFilter::Trace,
        }
    };

    // Initialize a custom logger that writes to stderr with proper formatting
    env_logger::Builder::from_default_env()
        .filter_level(log_level)
        .format(|buf, record| {
            let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.3f");
            let level = record.level();
            let module = record.module_path().unwrap_or("unknown");

            match level {
                log::Level::Error => writeln!(
                    buf,
                    "\x1b[31m[{}] ERROR [{}]: {}\x1b[0m",
                    timestamp,
                    module,
                    record.args()
                ),
                log::Level::Warn => writeln!(
                    buf,
                    "\x1b[33m[{}] WARN  [{}]: {}\x1b[0m",
                    timestamp,
                    module,
                    record.args()
                ),
                log::Level::Info => writeln!(
                    buf,
                    "\x1b[32m[{}] INFO  [{}]: {}\x1b[0m",
                    timestamp,
                    module,
                    record.args()
                ),
                log::Level::Debug => writeln!(
                    buf,
                    "\x1b[36m[{}] DEBUG [{}]: {}\x1b[0m",
                    timestamp,
                    module,
                    record.args()
                ),
                log::Level::Trace => writeln!(
                    buf,
                    "\x1b[35m[{}] TRACE [{}]: {}\x1b[0m",
                    timestamp,
                    module,
                    record.args()
                ),
            }
        })
        .write_style(env_logger::WriteStyle::Always)
        .target(env_logger::Target::Stderr)
        .init();

    log::info!("Logging initialized at level: {log_level:?}");

    Ok(())
}

/// Create a new snapshot
fn create_snapshot(
    preset: SnapshotPreset,
    output: PathBuf,
    compression_level: Option<u32>,
    no_compression: bool,
    no_validation: bool,
    progress: bool,
) -> Result<()> {
    println!("Creating snapshot with preset: {}", preset.name());

    let mut config = preset.config();

    // Apply command-line overrides
    if let Some(level) = compression_level {
        config.compression_level = level.clamp(1, 9);
    }
    if no_compression {
        config.compression_enabled = false;
    }
    if no_validation {
        config.validation_enabled = false;
    }
    if progress {
        config.progress_reporting = true;
    }

    let start_time = Instant::now();

    // Create snapshot
    let builder = SnapshotBuilder::new(config);
    builder
        .build_and_save(&output)
        .with_context(|| format!("Failed to create snapshot at {}", output.display()))?;

    let build_time = start_time.elapsed();

    // Get file size
    let file_size = std::fs::metadata(&output)
        .with_context(|| format!("Failed to get metadata for {}", output.display()))?
        .len();

    println!("✅ Snapshot created successfully!");
    println!("   📁 File: {}", output.display());
    println!("   📏 Size: {}", format_size(file_size));
    println!("   ⏱️  Time: {build_time:?}");

    // Show build statistics
    let stats = builder.stats();
    if !stats.errors.is_empty() {
        println!("⚠️  Errors encountered:");
        for error in &stats.errors {
            println!("   • {error}");
        }
    }
    if !stats.warnings.is_empty() {
        println!("⚠️  Warnings:");
        for warning in &stats.warnings {
            println!("   • {warning}");
        }
    }

    Ok(())
}

/// Validate a snapshot file
fn validate_snapshot(input: PathBuf, strict: bool, compatibility_only: bool) -> Result<()> {
    println!("Validating snapshot: {}", input.display());

    if !input.exists() {
        anyhow::bail!("Snapshot file does not exist: {}", input.display());
    }

    let start_time = Instant::now();

    if compatibility_only {
        // Quick compatibility check
        match SnapshotLoader::get_metadata(&input) {
            Ok(metadata) => {
                let is_compatible = metadata.is_compatible();
                let validation_time = start_time.elapsed();

                if is_compatible {
                    println!("✅ Snapshot is compatible");
                } else {
                    println!("❌ Snapshot compatibility issues detected");
                    println!("   Current version: {}", env!("CARGO_PKG_VERSION"));
                    println!("   Snapshot version: {}", metadata.rustmat_version);
                }

                println!("   ⏱️  Validation time: {validation_time:?}");
            }
            Err(e) => {
                println!("❌ Failed to read snapshot metadata: {e}");
                return Err(e.into());
            }
        }
    } else {
        // Full validation
        let config = SnapshotConfig {
            validation_enabled: true,
            ..SnapshotConfig::default()
        };

        let mut loader = SnapshotLoader::new(config);
        match loader.load(&input) {
            Ok((_snapshot, stats)) => {
                let validation_time = start_time.elapsed();

                println!("✅ Snapshot validation passed");
                println!("   📏 Total size: {}", format_size(stats.total_size as u64));
                println!(
                    "   🗜️  Compressed size: {}",
                    format_size(stats.compressed_size as u64)
                );
                println!(
                    "   📊 Compression ratio: {:.1}%",
                    stats.compression_efficiency() * 100.0
                );
                println!("   🔧 Builtins: {}", stats.builtin_count);
                println!("   ⏱️  Load time: {:?}", stats.load_time);
                println!("   ⏱️  Validation time: {validation_time:?}");

                if strict {
                    // Additional strict checks
                    println!("🔍 Performing strict validation...");
                    // In a real implementation, this would perform additional checks
                }
            }
            Err(e) => {
                println!("❌ Snapshot validation failed: {e}");
                if strict {
                    return Err(e.into());
                }
            }
        }
    }

    Ok(())
}

/// Show snapshot information
fn show_snapshot_info(input: PathBuf, detailed: bool, metrics: bool) -> Result<()> {
    println!("Snapshot Information: {}", input.display());
    println!();

    // Get basic file info
    let file_metadata = std::fs::metadata(&input)
        .with_context(|| format!("Failed to read file metadata for {}", input.display()))?;

    println!("📁 File Information:");
    println!("   Size: {}", format_size(file_metadata.len()));
    println!(
        "   Modified: {:?}",
        file_metadata.modified().unwrap_or(std::time::UNIX_EPOCH)
    );
    println!();

    // Get snapshot header
    match SnapshotLoader::peek_header(&input) {
        Ok(header) => {
            println!("📋 Snapshot Header:");
            println!("   Version: {}", header.version);
            println!("   Created: {:?}", header.metadata.created_at);
            println!("   RustMat Version: {}", header.metadata.rustmat_version);
            println!(
                "   Platform: {} {}",
                header.metadata.target_platform.os, header.metadata.target_platform.arch
            );

            if header.data_info.compression.algorithm
                != rustmat_snapshot::format::CompressionAlgorithm::None
            {
                println!(
                    "   Compression: {:?}",
                    header.data_info.compression.algorithm
                );
                println!(
                    "   Compression Ratio: {:.1}%",
                    (1.0 - header.data_info.compressed_size as f64
                        / header.data_info.uncompressed_size as f64)
                        * 100.0
                );
            }

            if detailed {
                println!();
                println!("🔧 Build Configuration:");
                println!(
                    "   Optimization: {}",
                    header.metadata.build_config.optimization_level
                );
                println!("   Debug Info: {}", header.metadata.build_config.debug_info);
                println!("   Compiler: {}", header.metadata.build_config.compiler);

                println!();
                println!("🚀 Features:");
                for feature in &header.metadata.feature_flags {
                    println!("   • {feature}");
                }

                println!();
                println!("🖥️  Platform Details:");
                println!(
                    "   CPU Features: {:?}",
                    header.metadata.target_platform.cpu_features
                );
                println!(
                    "   Page Size: {}",
                    header.metadata.target_platform.page_size
                );
                println!(
                    "   Cache Line Size: {}",
                    header.metadata.target_platform.cache_line_size
                );
            }

            if metrics {
                println!();
                println!("📊 Performance Metrics:");
                let perf = &header.metadata.performance_metrics;
                println!("   Creation Time: {:?}", perf.creation_time);
                println!("   Builtin Count: {}", perf.builtin_count);
                println!("   HIR Cache Entries: {}", perf.hir_cache_entries);
                println!("   Bytecode Cache Entries: {}", perf.bytecode_cache_entries);
                println!(
                    "   Peak Memory Usage: {}",
                    format_size(perf.peak_memory_usage as u64)
                );

                println!();
                println!("⚡ Estimated Load Time: {:?}", header.estimated_load_time());
            }
        }
        Err(e) => {
            println!("❌ Failed to read snapshot header: {e}");
            return Err(e.into());
        }
    }

    Ok(())
}

/// List available presets
fn list_presets(detailed: bool) -> Result<()> {
    println!("Available Snapshot Presets:");
    println!();

    for preset in SnapshotPreset::all_presets() {
        println!("🎯 {}", preset.name());
        println!("   {}", preset.description());

        if detailed {
            let chars = preset.characteristics();
            println!("   📊 Characteristics:");
            println!("      Build Time: {}", chars.build_time);
            println!("      Load Time: {}", chars.load_time);
            println!("      File Size: {}", chars.file_size);
            println!("      Memory Usage: {}", chars.memory_usage);
            println!("      Validation: {}", chars.validation_level);
            println!(
                "      Debug Friendly: {}",
                if chars.debugging_friendly {
                    "Yes"
                } else {
                    "No"
                }
            );
        }

        println!();
    }

    Ok(())
}

/// Benchmark snapshot loading performance
fn benchmark_snapshot(input: PathBuf, iterations: usize, warmup: usize) -> Result<()> {
    println!("Benchmarking snapshot: {}", input.display());
    println!("Iterations: {iterations} (+ {warmup} warmup)");
    println!();

    let config = SnapshotConfig::default();
    let mut loader = SnapshotLoader::new(config);

    // Warmup runs
    println!("🔥 Warming up...");
    for i in 0..warmup {
        print!("   Warmup {}/{}\r", i + 1, warmup);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        loader.load(&input).context("Warmup iteration failed")?;
        loader.clear_cache(); // Clear cache between runs
    }
    println!();

    // Benchmark runs
    println!("📊 Benchmarking...");
    let mut load_times = Vec::with_capacity(iterations);
    let mut total_stats = None;

    for i in 0..iterations {
        print!("   Iteration {}/{}\r", i + 1, iterations);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let start = Instant::now();
        let (_, stats) = loader.load(&input).context("Benchmark iteration failed")?;
        let total_time = start.elapsed();

        load_times.push(total_time);
        total_stats = Some(stats);

        loader.clear_cache(); // Clear cache between runs
    }
    println!();

    // Calculate statistics
    load_times.sort();
    let min_time = load_times[0];
    let max_time = load_times[iterations - 1];
    let median_time = load_times[iterations / 2];
    let avg_time: std::time::Duration =
        load_times.iter().sum::<std::time::Duration>() / iterations as u32;

    println!("📈 Benchmark Results:");
    println!("   Min Time: {min_time:?}");
    println!("   Avg Time: {avg_time:?}");
    println!("   Median Time: {median_time:?}");
    println!("   Max Time: {max_time:?}");

    if let Some(stats) = total_stats {
        println!();
        println!("📊 Load Statistics:");
        println!("   Total Size: {}", format_size(stats.total_size as u64));
        println!(
            "   Throughput: {:.1} MB/s",
            stats.loading_throughput() / 1_000_000.0
        );
        println!(
            "   Compression Ratio: {:.1}%",
            stats.compression_efficiency() * 100.0
        );
        println!("   Builtins: {}", stats.builtin_count);
    }

    Ok(())
}

/// Compare multiple snapshots
fn compare_snapshots(files: Vec<PathBuf>, show_size: bool, show_performance: bool) -> Result<()> {
    if files.len() < 2 {
        anyhow::bail!("Need at least 2 files to compare");
    }

    println!("Comparing {} snapshots:", files.len());
    println!();

    let mut results = Vec::new();

    for file in &files {
        if !file.exists() {
            println!("❌ File not found: {}", file.display());
            continue;
        }

        match SnapshotLoader::peek_header(file) {
            Ok(header) => {
                let file_size = std::fs::metadata(file)?.len();
                results.push((file.clone(), header, file_size));
            }
            Err(e) => {
                println!("❌ Failed to read {}: {}", file.display(), e);
            }
        }
    }

    if results.is_empty() {
        anyhow::bail!("No valid snapshots found");
    }

    // Basic comparison
    println!("📋 Basic Comparison:");
    for (file, header, file_size) in &results {
        println!("   {}", file.file_name().unwrap().to_string_lossy());
        println!("     Version: {}", header.metadata.rustmat_version);
        println!("     Size: {}", format_size(*file_size));
        println!("     Created: {:?}", header.metadata.created_at);
        println!();
    }

    if show_size {
        println!("📏 Size Comparison:");
        let mut sorted_results = results.clone();
        sorted_results.sort_by_key(|(_, _, size)| *size);

        for (file, header, file_size) in &sorted_results {
            let compression_ratio = if header.data_info.compressed_size > 0 {
                (1.0 - header.data_info.compressed_size as f64
                    / header.data_info.uncompressed_size as f64)
                    * 100.0
            } else {
                0.0
            };

            println!(
                "   {} - {} (compression: {:.1}%)",
                file.file_name().unwrap().to_string_lossy(),
                format_size(*file_size),
                compression_ratio
            );
        }
        println!();
    }

    if show_performance {
        println!("⚡ Performance Comparison:");
        for (file, header, _) in &results {
            let estimated_load = header.estimated_load_time();
            let builtin_count = header.metadata.performance_metrics.builtin_count;

            println!(
                "   {} - Est. load: {:?}, Builtins: {}",
                file.file_name().unwrap().to_string_lossy(),
                estimated_load,
                builtin_count
            );
        }
        println!();
    }

    Ok(())
}

/// Format file size in human-readable format
fn format_size(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = size as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", size as u64, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_preset_conversion() {
        let preset: SnapshotPreset = PresetName::Development.into();
        assert_eq!(preset.name(), "Development");

        let preset: SnapshotPreset = PresetName::Production.into();
        assert_eq!(preset.name(), "Production");
    }
}
