use anyhow::Result;
use runmat_config::RunMatConfig;
use runmat_gc::gc_stats;

pub fn show_version(detailed: bool) {
    println!("RunMat v{}", env!("CARGO_PKG_VERSION"));

    if detailed {
        println!(
            "Built with Rust {}",
            std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string())
        );
        println!(
            "Target: {}",
            std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string())
        );
        println!(
            "Profile: {}",
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        );
    }
}

pub async fn show_system_info(config: &RunMatConfig) -> Result<()> {
    println!("RunMat System Information");
    println!("==========================");
    println!();

    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!(
        "Rust Version: {}",
        std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string())
    );
    println!(
        "Target: {}",
        std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string())
    );
    println!();

    println!("Runtime Configuration:");
    println!(
        "  JIT Compiler: {}",
        if config.jit.enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!("  JIT Threshold: {}", config.jit.threshold);
    println!("  JIT Optimization: {:?}", config.jit.optimization_level);
    println!(
        "  GC Preset: {:?}",
        config
            .gc
            .preset
            .as_ref()
            .map(|p| format!("{p:?}"))
            .unwrap_or_else(|| "default".to_string())
    );
    if let Some(young_size) = config.gc.young_size_mb {
        println!("  GC Young Generation: {young_size}MB");
    }
    if let Some(threads) = config.gc.threads {
        println!("  GC Threads: {threads}");
    }
    println!("  GC Statistics: {}", config.gc.collect_stats);
    println!();

    println!("Environment:");
    println!("  RUNMAT_DEBUG: {:?}", std::env::var("RUNMAT_DEBUG").ok());
    println!(
        "  RUNMAT_LOG_LEVEL: {:?}",
        std::env::var("RUNMAT_LOG_LEVEL").ok()
    );
    println!(
        "  RUNMAT_TIMEOUT: {:?}",
        std::env::var("RUNMAT_TIMEOUT").ok()
    );
    println!(
        "  RUNMAT_JIT_ENABLE: {:?}",
        std::env::var("RUNMAT_JIT_ENABLE").ok()
    );
    println!(
        "  RUNMAT_GC_PRESET: {:?}",
        std::env::var("RUNMAT_GC_PRESET").ok()
    );
    println!();

    let gc_stats = gc_stats();
    println!("Garbage Collector Status:");
    println!("{}", gc_stats.summary_report());
    println!();

    println!("Help:");
    println!("  See 'runmat --help' for commands.");
    println!("  See 'runmat <command> --help' for subcommand details.");

    Ok(())
}
