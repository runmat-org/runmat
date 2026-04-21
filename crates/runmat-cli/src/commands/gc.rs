use anyhow::Result;
use log::{error, info};
use runmat_builtins::Value;
use runmat_gc::{gc_allocate, gc_collect_major, gc_collect_minor, gc_get_config, gc_stats};
use runmat_time::Instant;

use crate::cli::GcCommand;

pub async fn execute_gc_command(gc_command: GcCommand) -> Result<()> {
    match gc_command {
        GcCommand::Stats => {
            let stats = gc_stats();
            println!("{}", stats.summary_report());
        }
        GcCommand::Minor => {
            let start = Instant::now();
            match gc_collect_minor() {
                Ok(collected) => {
                    let duration = start.elapsed();
                    println!("Minor GC collected {collected} objects in {duration:?}");
                }
                Err(e) => {
                    error!("Minor GC failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        GcCommand::Major => {
            let start = Instant::now();
            match gc_collect_major() {
                Ok(collected) => {
                    let duration = start.elapsed();
                    println!("Major GC collected {collected} objects in {duration:?}");
                }
                Err(e) => {
                    error!("Major GC failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        GcCommand::Config => {
            println!("Current GC Configuration:");
            let config = gc_get_config();
            println!(
                "  Young Generation Size: {} MB",
                config.young_generation_size / 1024 / 1024
            );
            println!(
                "  Minor GC Threshold: {} objects",
                config.minor_gc_threshold
            );
            println!(
                "  Major GC Threshold: {} objects",
                config.major_gc_threshold
            );
            println!("  Max GC Threads: {}", config.max_gc_threads);
            println!(
                "  Collection Statistics: {}",
                if config.collect_statistics {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            println!(
                "  Verbose Logging: {}",
                if config.verbose_logging {
                    "enabled"
                } else {
                    "disabled"
                }
            );
        }
        GcCommand::Stress { allocations } => {
            info!("Starting GC stress test with {allocations} allocations");

            let start_time = Instant::now();
            let initial_stats = gc_stats();

            println!("Running GC stress test with {allocations} allocations...");

            let mut objects = Vec::new();
            for i in 0..allocations {
                let value = Value::Num(i as f64);
                match gc_allocate(value) {
                    Ok(ptr) => {
                        objects.push(ptr);

                        if i % 1000 == 0 && i > 0 {
                            let _ = gc_collect_minor();
                        }
                        if i % 5000 == 0 && i > 0 {
                            let _ = gc_collect_major();
                        }
                    }
                    Err(e) => {
                        error!("Allocation failed at iteration {i}: {e}");
                        break;
                    }
                }

                if i % (allocations / 10).max(1) == 0 {
                    println!("  Progress: {i}/{allocations} allocations");
                }
            }

            let duration = start_time.elapsed();
            let final_stats = gc_stats();

            println!("GC Stress Test Results:");
            println!("  Duration: {duration:?}");
            println!("  Allocations completed: {}", objects.len());
            println!(
                "  Allocation rate: {:.2} allocs/sec",
                objects.len() as f64 / duration.as_secs_f64()
            );
            println!(
                "  Total collections: {}",
                final_stats
                    .minor_collections
                    .load(std::sync::atomic::Ordering::Relaxed)
                    - initial_stats
                        .minor_collections
                        .load(std::sync::atomic::Ordering::Relaxed)
                    + final_stats
                        .major_collections
                        .load(std::sync::atomic::Ordering::Relaxed)
                    - initial_stats
                        .major_collections
                        .load(std::sync::atomic::Ordering::Relaxed)
            );
            println!(
                "  Final memory: {} bytes",
                final_stats
                    .current_memory_usage
                    .load(std::sync::atomic::Ordering::Relaxed)
            );

            match gc_collect_major() {
                Ok(collected) => println!("  Final collection freed {collected} objects"),
                Err(e) => error!("Final collection failed: {e}"),
            }
        }
    }
    Ok(())
}
