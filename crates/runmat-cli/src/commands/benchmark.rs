use anyhow::{Context, Result};
use log::{error, info};
use runmat_config::{
    resolve_project_source_input_from, ResolveProjectSourceInputError, RunMatConfig,
};
use runmat_core::{
    abi::{DiagnosticSeverity, ExecutionRequest, HostExecutionPolicy, SourceInput},
    TelemetryHost, TelemetryRunConfig, TelemetryRunFinish,
};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use crate::cli::Cli;
use crate::commands::session::create_session;
use crate::diagnostics::format_frontend_error;
use crate::telemetry::{capture_provider_snapshot, RuntimeExecutionCounters, TelemetryRunKind};
use crate::AlreadyReportedCliError;

pub async fn execute_benchmark(
    file: PathBuf,
    iterations: u32,
    jit: bool,
    cli: &Cli,
    config: &RunMatConfig,
) -> Result<()> {
    let file = resolve_benchmark_input(file)?;
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
    let mut bench_run = engine.telemetry_run(TelemetryRunConfig {
        kind: TelemetryRunKind::Benchmark,
        jit_enabled: jit,
        accelerate_enabled: config.accelerate.enabled,
    });
    let source_name = file.to_string_lossy().to_string();

    let mut total_time = Duration::ZERO;
    let mut jit_executions: u64 = 0;
    let mut interpreter_executions: u64 = 0;

    println!("Warming up...");
    for _ in 0..3 {
        let request = ExecutionRequest::for_source(
            SourceInput::Text {
                name: source_name.clone(),
                text: content.clone(),
            },
            crate::diagnostics::parser_compat(config.language.compat),
            HostExecutionPolicy::default(),
            engine.workspace_handle(),
        );
        match engine.execute_request(request).await {
            Ok(outcome) if outcome_error_code(&outcome).is_none() => {}
            Ok(outcome) => {
                let error =
                    outcome_error_code(&outcome).unwrap_or_else(|| "runtime_error".to_string());
                if let Some(run) = bench_run.take() {
                    run.finish(TelemetryRunFinish {
                        duration: Some(total_time),
                        success: false,
                        jit_used: outcome.used_jit,
                        error: Some(error.clone()),
                        failure: None,
                        host: Some(TelemetryHost::Cli),
                        counters: Some(RuntimeExecutionCounters {
                            total_executions: 0,
                            jit_compiled: 0,
                            interpreter_fallback: 0,
                        }),
                        provider: capture_provider_snapshot(),
                    });
                }
                eprintln!("Benchmark error: {error}");
                return Err(AlreadyReportedCliError.into());
            }
            Err(err) => {
                let failure = err.telemetry_failure_info();
                if let Some(run) = bench_run.take() {
                    run.finish(TelemetryRunFinish {
                        duration: Some(total_time),
                        success: false,
                        jit_used: false,
                        error: Some(failure.code.clone()),
                        failure: Some(failure),
                        host: Some(TelemetryHost::Cli),
                        counters: Some(RuntimeExecutionCounters {
                            total_executions: 0,
                            jit_compiled: 0,
                            interpreter_fallback: 0,
                        }),
                        provider: capture_provider_snapshot(),
                    });
                }
                if let Some(diag) =
                    format_frontend_error(&err, file.to_string_lossy().as_ref(), &content)
                {
                    eprintln!("{diag}");
                } else {
                    eprintln!("Benchmark error: {err}");
                }
                return Err(AlreadyReportedCliError.into());
            }
        }
    }

    println!("Running benchmark...");
    for i in 1..=iterations {
        let request = ExecutionRequest::for_source(
            SourceInput::Text {
                name: source_name.clone(),
                text: content.clone(),
            },
            crate::diagnostics::parser_compat(config.language.compat),
            HostExecutionPolicy::default(),
            engine.workspace_handle(),
        );
        let outcome = match engine.execute_request(request).await {
            Ok(outcome) => outcome,
            Err(err) => {
                let failure = err.telemetry_failure_info();
                let counters = RuntimeExecutionCounters {
                    total_executions: i.saturating_sub(1) as u64,
                    jit_compiled: jit_executions,
                    interpreter_fallback: interpreter_executions,
                };
                if let Some(run) = bench_run.take() {
                    run.finish(TelemetryRunFinish {
                        duration: Some(total_time),
                        success: false,
                        jit_used: false,
                        error: Some(failure.code.clone()),
                        failure: Some(failure),
                        host: Some(TelemetryHost::Cli),
                        counters: Some(counters),
                        provider: capture_provider_snapshot(),
                    });
                }
                if let Some(diag) =
                    format_frontend_error(&err, file.to_string_lossy().as_ref(), &content)
                {
                    eprintln!("{diag}");
                } else {
                    eprintln!("Benchmark error: {err}");
                }
                return Err(AlreadyReportedCliError.into());
            }
        };

        let iter_duration = Duration::from_millis(outcome.execution_time_ms);
        if let Some(error) = outcome_error_code(&outcome) {
            total_time += iter_duration;
            let counters = RuntimeExecutionCounters {
                total_executions: i as u64,
                jit_compiled: jit_executions + if outcome.used_jit { 1 } else { 0 },
                interpreter_fallback: interpreter_executions + if outcome.used_jit { 0 } else { 1 },
            };
            if let Some(run) = bench_run.take() {
                run.finish(TelemetryRunFinish {
                    duration: Some(total_time),
                    success: false,
                    jit_used: outcome.used_jit,
                    error: Some(error.clone()),
                    failure: None,
                    host: Some(TelemetryHost::Cli),
                    counters: Some(counters),
                    provider: capture_provider_snapshot(),
                });
            }
            error!("Benchmark iteration {i} failed: {error}");
            return Err(AlreadyReportedCliError.into());
        }

        total_time += iter_duration;
        if outcome.used_jit {
            jit_executions += 1;
        } else {
            interpreter_executions += 1;
        }

        if i % 10 == 0 {
            println!("  Completed {i} iterations");
        }
    }

    fn outcome_error_code(outcome: &runmat_core::abi::ExecutionOutcome) -> Option<String> {
        outcome
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
            .map(|diagnostic| diagnostic.code.clone())
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

    Ok(())
}

fn resolve_benchmark_input(file: PathBuf) -> Result<PathBuf> {
    let cwd = std::env::current_dir().context("failed to resolve current working directory")?;
    resolve_project_source_input_from(&cwd, &file).map_err(|err| match err {
        ResolveProjectSourceInputError::EntrypointResolve { .. } => {
            anyhow::anyhow!(
                "failed to resolve benchmark target '{}': {}",
                file.display(),
                err
            )
        }
    })
}

#[cfg(test)]
mod tests {
    use super::resolve_benchmark_input;
    use crate::test_support::ScopedCurrentDir;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn resolves_named_entrypoint_to_manifest_path_target() {
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[[entrypoints]]
name = "main"
path = "src/main"
"#,
        )
        .unwrap();

        let _cwd = ScopedCurrentDir::enter(tmp.path());
        let resolved = resolve_benchmark_input(PathBuf::from("main")).expect("resolve entrypoint");

        assert_eq!(
            resolved.canonicalize().unwrap(),
            tmp.path().join("src/main.m").canonicalize().unwrap()
        );
    }

    #[test]
    fn resolve_benchmark_input_infers_m_extension_for_relative_path() {
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();

        let _cwd = ScopedCurrentDir::enter(tmp.path());

        let resolved =
            resolve_benchmark_input(PathBuf::from("src/main")).expect("should infer .m extension");

        assert_eq!(resolved, PathBuf::from("src/main.m"));
    }

    #[test]
    fn resolves_module_function_entrypoint_to_source_root_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src/app")).unwrap();
        fs::write(
            tmp.path().join("src/app/server.m"),
            "function y = main(); y = 1; end",
        )
        .unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[[entrypoints]]
name = "server"
module = "app.server"
function = "main"
"#,
        )
        .unwrap();

        let _cwd = ScopedCurrentDir::enter(tmp.path());
        let resolved = resolve_benchmark_input(PathBuf::from("server"))
            .expect("module/function entrypoint should resolve to module file");

        assert_eq!(
            resolved.canonicalize().unwrap(),
            tmp.path().join("src/app/server.m").canonicalize().unwrap()
        );
    }

    #[test]
    fn module_function_entrypoint_errors_when_module_file_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[[entrypoints]]
name = "server"
module = "app.server"
function = "main"
"#,
        )
        .unwrap();

        let _cwd = ScopedCurrentDir::enter(tmp.path());
        let err = resolve_benchmark_input(PathBuf::from("server"))
            .expect_err("missing module file should return explicit error");

        let message = err.to_string();
        assert!(
            message.contains("module/function target")
                || message.contains("did not resolve under configured source roots"),
            "unexpected error message: {message}"
        );
    }
}
