use anyhow::{Context, Result};
use log::{error, info};
use runmat_config::RunMatConfig;
use runmat_core::{
    runtime_error_telemetry_failure_info, TelemetryHost, TelemetryRunConfig, TelemetryRunFinish,
};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use crate::cli::Cli;
use crate::commands::session::create_session;
use crate::telemetry::{capture_provider_snapshot, RuntimeExecutionCounters, TelemetryRunKind};

pub async fn execute_benchmark(
    file: PathBuf,
    iterations: u32,
    jit: bool,
    cli: &Cli,
    config: &RunMatConfig,
) -> Result<()> {
    info!("Benchmarking script: {file:?} ({iterations} iterations, JIT: {jit})");

    let content = fs::read_to_string(&file)
        .with_context(|| format!("Failed to read script file: {file:?}"))?;

    let mut engine = create_session(
        jit,
        false,
        cli.snapshot.as_ref(),
        config,
        "Failed to create execution engine",
    )?;
    engine.set_source_name_override(Some(file.to_string_lossy().to_string()));
    let mut bench_run = engine.telemetry_run(TelemetryRunConfig {
        kind: TelemetryRunKind::Benchmark,
        jit_enabled: jit,
        accelerate_enabled: config.accelerate.enabled,
    });

    let mut total_time = Duration::ZERO;
    let mut jit_executions: u64 = 0;
    let mut interpreter_executions: u64 = 0;

    println!("Warming up...");
    for _ in 0..3 {
        let _ = engine.execute(&content).await.map_err(anyhow::Error::new)?;
    }

    println!("Running benchmark...");
    for i in 1..=iterations {
        let result = engine.execute(&content).await.map_err(anyhow::Error::new)?;

        let iter_duration = Duration::from_millis(result.execution_time_ms);
        if let Some(error) = result
            .error
            .as_ref()
            .and_then(|err| err.identifier().map(|value| value.to_string()))
            .or_else(|| result.error.as_ref().map(|_| "runtime_error".to_string()))
        {
            total_time += iter_duration;
            let counters = RuntimeExecutionCounters {
                total_executions: i as u64,
                jit_compiled: jit_executions + if result.used_jit { 1 } else { 0 },
                interpreter_fallback: interpreter_executions + if result.used_jit { 0 } else { 1 },
            };
            if let Some(run) = bench_run.take() {
                run.finish(TelemetryRunFinish {
                    duration: Some(total_time),
                    success: false,
                    jit_used: result.used_jit,
                    error: Some(error.clone()),
                    failure: result
                        .error
                        .as_ref()
                        .map(runtime_error_telemetry_failure_info),
                    host: Some(TelemetryHost::Cli),
                    counters: Some(counters),
                    provider: capture_provider_snapshot(),
                });
            }
            if let Some(runtime_error) = result.error.as_ref() {
                eprintln!(
                    "{}",
                    runtime_error.format_diagnostic_with_source(
                        Some(file.to_string_lossy().as_ref()),
                        Some(&content),
                    )
                );
            } else {
                error!("Benchmark iteration {i} failed: {error}");
            }
            std::process::exit(1);
        }

        total_time += iter_duration;
        if result.used_jit {
            jit_executions += 1;
        } else {
            interpreter_executions += 1;
        }

        if i % 10 == 0 {
            println!("  Completed {i} iterations");
        }
    }

    let avg_time = total_time / iterations;
    println!("\nBenchmark Results:");
    println!("  Total iterations: {iterations}");
    println!("  JIT executions: {jit_executions}");
    println!("  Interpreter executions: {interpreter_executions}");
    println!("  Total time: {total_time:?}");
    println!("  Average time: {avg_time:?}");
    println!(
        "  Throughput: {:.2} executions/second",
        iterations as f64 / total_time.as_secs_f64()
    );

    let counters = RuntimeExecutionCounters {
        total_executions: iterations as u64,
        jit_compiled: jit_executions,
        interpreter_fallback: interpreter_executions,
    };
    if let Some(run) = bench_run.take() {
        run.finish(TelemetryRunFinish {
            duration: Some(total_time),
            success: true,
            jit_used: jit_executions > 0,
            error: None,
            failure: None,
            host: Some(TelemetryHost::Cli),
            counters: Some(counters),
            provider: capture_provider_snapshot(),
        });
    }

    engine.set_source_name_override(None);

    Ok(())
}
