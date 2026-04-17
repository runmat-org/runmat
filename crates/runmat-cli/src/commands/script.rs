use anyhow::{Context, Result};
use clap::Parser;
use log::{info, warn};
use runmat_config::RunMatConfig;
use runmat_core::{
    runtime_error_telemetry_failure_info, TelemetryHost, TelemetryRunConfig, TelemetryRunFinish,
};
use runmat_time::Instant;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::cli::{CaptureFiguresMode, Cli, FigureSize};
use crate::commands::accel::dump_provider_telemetry_if_requested;
use crate::commands::bytecode::{emit_bytecode, write_bytecode_output};
use crate::commands::session::create_session;
use crate::commands::streams::emit_execution_streams;
use crate::diagnostics::format_frontend_error;
use crate::telemetry::{capture_provider_snapshot, TelemetryRunKind};

pub async fn execute_script(
    script: PathBuf,
    emit_bytecode_path: Option<PathBuf>,
    cli: &Cli,
    config: &RunMatConfig,
) -> Result<()> {
    execute_script_with_args(script, vec![], emit_bytecode_path, cli, config).await
}

pub(crate) async fn execute_script_with_current_cli(
    script: PathBuf,
    config: &RunMatConfig,
) -> Result<()> {
    let cli = Cli::parse();
    execute_script_with_args(script, vec![], None, &cli, config).await
}

pub async fn execute_script_with_args(
    script: PathBuf,
    _args: Vec<String>,
    emit_bytecode_path: Option<PathBuf>,
    cli: &Cli,
    config: &RunMatConfig,
) -> Result<()> {
    info!("Executing script: {script:?}");

    let content = fs::read_to_string(&script)
        .with_context(|| format!("Failed to read script file: {script:?}"))?;

    if let Some(path) = &emit_bytecode_path {
        let output = emit_bytecode(&content, config)
            .with_context(|| format!("Failed to emit bytecode for {script:?}"))?;
        write_bytecode_output(path, &output)?;
        return Ok(());
    }

    let enable_jit = config.jit.enabled;
    let mut engine = create_session(
        enable_jit,
        config.runtime.verbose,
        config.runtime.snapshot_path.as_ref(),
        config,
        "Failed to create execution engine",
    )?;
    engine.set_source_name_override(Some(script.to_string_lossy().to_string()));
    let mut script_run = engine.telemetry_run(TelemetryRunConfig {
        kind: TelemetryRunKind::Script,
        jit_enabled: config.jit.enabled,
        accelerate_enabled: config.accelerate.enabled,
    });
    let start_time = Instant::now();
    let result = match engine.execute(&content).await {
        Ok(result) => result,
        Err(err) => {
            let failure = err.telemetry_failure_info();
            if let Some(run) = script_run.take() {
                run.finish(TelemetryRunFinish {
                    duration: Some(start_time.elapsed()),
                    success: false,
                    jit_used: false,
                    error: Some(failure.code.clone()),
                    failure: Some(failure),
                    host: Some(TelemetryHost::Cli),
                    counters: None,
                    provider: capture_provider_snapshot(),
                });
            }
            if let Some(diag) =
                format_frontend_error(&err, script.to_string_lossy().as_ref(), &content)
            {
                eprintln!("{diag}");
            } else {
                eprintln!("Execution error: {err}");
            }
            std::process::exit(1);
        }
    };

    let execution_time = start_time.elapsed();
    emit_execution_streams(&result.streams);

    let provider_snapshot = capture_provider_snapshot();
    let failure = result
        .error
        .as_ref()
        .map(runtime_error_telemetry_failure_info);
    let error_payload = result
        .error
        .as_ref()
        .and_then(|err| err.identifier().map(|value| value.to_string()))
        .or_else(|| failure.as_ref().map(|info| info.code.clone()))
        .or_else(|| result.error.as_ref().map(|_| "runtime_error".to_string()));
    let success = error_payload.is_none();

    if let Some(artifacts_plan) = ScriptArtifactsPlan::from_cli(cli)? {
        if let Err(err) = write_script_artifacts(
            &artifacts_plan,
            &script,
            &result,
            execution_time,
            success,
            error_payload.as_deref(),
        )
        .await
        {
            warn!("Failed to write run artifacts: {err}");
            eprintln!("Warning: failed to write run artifacts: {err}");
        }
    }

    if let Some(run) = script_run.take() {
        run.finish(TelemetryRunFinish {
            duration: Some(execution_time),
            success,
            jit_used: result.used_jit,
            error: error_payload.clone(),
            failure,
            host: Some(TelemetryHost::Cli),
            counters: None,
            provider: provider_snapshot,
        });
    }

    if let Some(error) = result.error.as_ref() {
        eprintln!(
            "{}",
            error.format_diagnostic_with_source(
                Some(script.to_string_lossy().as_ref()),
                Some(&content),
            )
        );
        std::process::exit(1);
    } else if let Some(error) = error_payload {
        eprintln!("{error}");
        std::process::exit(1);
    } else {
        if result.used_jit {
            info!("Script executed successfully in {:?} (JIT)", execution_time);
        } else {
            info!("Script executed successfully in {:?}", execution_time);
        }
        if let Some(value) = result.value {
            if config.runtime.verbose {
                println!("{value:?}");
            }
        }
    }

    engine.set_source_name_override(None);
    dump_provider_telemetry_if_requested();

    Ok(())
}

#[derive(Clone, Debug)]
struct ScriptArtifactsPlan {
    artifacts_dir: PathBuf,
    manifest_path: PathBuf,
    capture_figures: CaptureFiguresMode,
    figure_size: FigureSize,
    max_figures: usize,
}

impl ScriptArtifactsPlan {
    fn from_cli(cli: &Cli) -> Result<Option<Self>> {
        let artifacts_dir = match (&cli.artifacts_dir, &cli.artifacts_manifest) {
            (Some(dir), _) => Some(dir.clone()),
            (None, Some(manifest)) => manifest.parent().map(normalize_manifest_parent),
            (None, None) => None,
        };

        let Some(artifacts_dir) = artifacts_dir else {
            return Ok(None);
        };

        let manifest_path = cli
            .artifacts_manifest
            .clone()
            .unwrap_or_else(|| artifacts_dir.join("run_manifest.json"));

        Ok(Some(Self {
            artifacts_dir,
            manifest_path,
            capture_figures: cli.capture_figures,
            figure_size: cli.figure_size.clone(),
            max_figures: cli.max_figures,
        }))
    }
}

fn normalize_manifest_parent(parent: &Path) -> PathBuf {
    if parent.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        parent.to_path_buf()
    }
}

async fn write_script_artifacts(
    plan: &ScriptArtifactsPlan,
    script: &Path,
    result: &runmat_core::ExecutionResult,
    execution_time: Duration,
    success: bool,
    error_identifier: Option<&str>,
) -> Result<()> {
    fs::create_dir_all(&plan.artifacts_dir).with_context(|| {
        format!(
            "Failed to create artifacts directory {}",
            plan.artifacts_dir.display()
        )
    })?;

    let figure_exports = export_touched_figures(plan, &result.figures_touched).await;

    let mut stdout_bytes: usize = 0;
    let mut stderr_bytes: usize = 0;
    for stream in &result.streams {
        match stream.stream {
            runmat_core::ExecutionStreamKind::Stdout => stdout_bytes += stream.text.len(),
            runmat_core::ExecutionStreamKind::Stderr => stderr_bytes += stream.text.len(),
            runmat_core::ExecutionStreamKind::ClearScreen => {}
        }
    }

    let manifest = serde_json::json!({
        "schema_version": "runmat.artifacts.v1",
        "script": script.to_string_lossy(),
        "success": success,
        "execution_time_ms": execution_time.as_millis() as u64,
        "used_jit": result.used_jit,
        "error_identifier": error_identifier,
        "figures_touched": result.figures_touched,
        "figure_exports": figure_exports,
        "stream_summary": {
            "entry_count": result.streams.len(),
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
        },
        "capture": {
            "capture_figures": format!("{:?}", plan.capture_figures).to_lowercase(),
            "figure_width": plan.figure_size.width,
            "figure_height": plan.figure_size.height,
            "max_figures": plan.max_figures,
        }
    });

    if let Some(parent) = plan.manifest_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "Failed to create manifest parent directory {}",
                parent.display()
            )
        })?;
    }
    fs::write(
        &plan.manifest_path,
        serde_json::to_string_pretty(&manifest)
            .context("Failed to serialize run artifacts manifest")?,
    )
    .with_context(|| {
        format!(
            "Failed to write artifacts manifest {}",
            plan.manifest_path.display()
        )
    })?;

    info!(
        "Wrote run artifacts manifest: {}",
        plan.manifest_path.display()
    );
    Ok(())
}

async fn export_touched_figures(
    plan: &ScriptArtifactsPlan,
    figures_touched: &[u32],
) -> Vec<serde_json::Value> {
    use runmat_runtime::builtins::plotting::{render_figure_snapshot, FigureHandle};

    let capture_enabled = match plan.capture_figures {
        CaptureFiguresMode::Off => false,
        CaptureFiguresMode::Auto => !figures_touched.is_empty(),
        CaptureFiguresMode::On => true,
    };
    if !capture_enabled {
        return Vec::new();
    }

    let figures_dir = plan.artifacts_dir.join("figures");
    if let Err(err) = fs::create_dir_all(&figures_dir) {
        warn!(
            "Failed to create figures artifact directory {}: {}",
            figures_dir.display(),
            err
        );
        return Vec::new();
    }

    let mut exports = Vec::new();

    for (index, handle_raw) in figures_touched.iter().enumerate().take(plan.max_figures) {
        let handle = FigureHandle::from(*handle_raw);
        match render_figure_snapshot(
            handle,
            plan.figure_size.width,
            plan.figure_size.height,
            None,
        )
        .await
        {
            Ok(bytes) => {
                let file_name = format!("figure_{:03}_h{}.png", index + 1, handle_raw);
                let file_path = figures_dir.join(file_name);
                if let Err(err) = fs::write(&file_path, bytes.as_slice()) {
                    exports.push(serde_json::json!({
                        "handle": handle_raw,
                        "ok": false,
                        "error": format!("failed_to_write: {}", err),
                    }));
                    continue;
                }
                exports.push(serde_json::json!({
                    "handle": handle_raw,
                    "ok": true,
                    "path": file_path.to_string_lossy(),
                    "format": "png",
                    "width": plan.figure_size.width,
                    "height": plan.figure_size.height,
                }));
            }
            Err(err) => {
                exports.push(serde_json::json!({
                    "handle": handle_raw,
                    "ok": false,
                    "error": err.to_string(),
                }));
            }
        }
    }

    exports
}
