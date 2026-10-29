use anyhow::{Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution_runner_native::NativeStructuralStudyPlacement;
use runmat_runtime::analysis::{
    analysis_execute_structural_study_with_placement,
    analysis_execute_structural_sweep_with_placement, AnalysisStudyRunData, AnalysisStudySweepData,
};
use runmat_runtime::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use serde::Serialize;
use std::path::PathBuf;

use crate::cli::{Cli, ExecutionTargetArgs};
use crate::presentation;
use crate::AlreadyReportedCliError;

pub(crate) mod command;
mod input_materialization;
mod plan;
mod preparation;
pub(crate) mod remote;
pub(crate) mod result_presentation;
mod run;
mod snapshot;
pub(crate) mod source;

pub use command::{plan, reassess, render_snapshot, snapshot, verify_snapshot};

#[derive(Debug)]
pub(crate) struct NativeAnalysisService {
    compatibility_mode: runmat_core::CompatMode,
}

impl NativeAnalysisService {
    pub(crate) const fn new(compatibility_mode: runmat_core::CompatMode) -> Self {
        Self { compatibility_mode }
    }
}

impl runmat_runtime::context::RuntimeAnalysisService for NativeAnalysisService {
    fn materialize_signal(
        &self,
        request: runmat_runtime::context::RuntimeSignalMaterializationRequest,
    ) -> runmat_runtime::context::RuntimeServiceFuture<
        Result<runmat_analysis_core::PhysicalSignal, runmat_runtime::RuntimeError>,
    > {
        Box::pin(async move {
            input_materialization::materialize_runtime_signal(request).map_err(|error| {
                runmat_runtime::build_runtime_error(error.to_string())
                    .with_identifier("RunMat:FEA:SignalMaterialization")
                    .build()
            })
        })
    }

    fn execute_study(
        &self,
        study: runmat_runtime::analysis::AnalysisStudySpec,
        context: OperationContext,
    ) -> runmat_runtime::context::RuntimeServiceFuture<
        Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope>,
    > {
        let compatibility_mode = self.compatibility_mode;
        Box::pin(async move {
            let error_context = context.clone();
            let placement =
                NativeStructuralStudyPlacement::for_current_executable(compatibility_mode)
                    .map_err(|error| host_placement_error("study", error, &error_context))?;
            analysis_execute_structural_study_with_placement(&study, context, &placement).await
        })
    }

    fn execute_sweep(
        &self,
        sweep: runmat_runtime::analysis::AnalysisStudySweepSpec,
        context: OperationContext,
    ) -> runmat_runtime::context::RuntimeServiceFuture<
        Result<OperationEnvelope<AnalysisStudySweepData>, OperationErrorEnvelope>,
    > {
        let compatibility_mode = self.compatibility_mode;
        Box::pin(async move {
            let error_context = context.clone();
            let placement =
                NativeStructuralStudyPlacement::for_current_executable(compatibility_mode)
                    .map_err(|error| host_placement_error("sweep", error, &error_context))?;
            analysis_execute_structural_sweep_with_placement(&sweep, context, &placement).await
        })
    }
    fn execute_document(
        &self,
        path: PathBuf,
    ) -> runmat_runtime::context::RuntimeServiceFuture<
        Result<AnalysisStudyRunData, runmat_runtime::RuntimeError>,
    > {
        Box::pin(async move { run_document_through_cli(&path) })
    }
}

/// Runs a `.fea` document with this executable's public `run` command, then reads the
/// authenticated publication it wrote. The study runs exactly as it does from a shell.
#[expect(
    clippy::result_large_err,
    reason = "RuntimeAnalysisService requires the canonical owned RuntimeError; changing its storage belongs to runmat-runtime"
)]
fn run_document_through_cli(
    path: &std::path::Path,
) -> Result<AnalysisStudyRunData, runmat_runtime::RuntimeError> {
    let failure = |message: String| {
        runmat_runtime::build_runtime_error(message)
            .with_identifier("RunMat:fea:OperationFailed")
            .build()
    };
    let executable = std::env::current_exe()
        .map_err(|error| failure(format!("cannot locate the RunMat executable: {error}")))?;
    let output = std::process::Command::new(executable)
        .arg("run")
        .arg(path)
        .arg("--json")
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|error| failure(format!("cannot start `runmat run`: {error}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let reason = stderr
            .lines()
            .rev()
            .find(|line| line.trim_start().starts_with("Error"))
            .or_else(|| stderr.lines().rev().find(|line| !line.trim().is_empty()))
            .unwrap_or("no diagnostic was printed");
        return Err(failure(format!(
            "`runmat run {}` failed: {reason}",
            path.display()
        )));
    }
    #[derive(serde::Deserialize)]
    struct PrintedStudy {
        study_definition_digest: String,
        result_manifest_artifact_path: String,
    }
    let printed: PrintedStudy = serde_json::from_slice(&output.stdout).map_err(|error| {
        failure(format!(
            "`runmat run {}` did not print one study result: {error}",
            path.display()
        ))
    })?;
    let definition: runmat_execution::Digest = printed
        .study_definition_digest
        .parse()
        .map_err(|error| failure(format!("invalid study definition digest: {error}")))?;
    runmat_runtime::analysis::study_result_manifest::read_saved_study_publication(
        definition,
        std::path::Path::new(&printed.result_manifest_artifact_path),
        16 * 1024 * 1024,
    )
    .map_err(failure)
}

fn host_placement_error(
    kind: &str,
    error: std::io::Error,
    context: &OperationContext,
) -> OperationErrorEnvelope {
    runmat_runtime::operations::operation_error(
        "fea.run_study",
        "fea.run_study/v1",
        context,
        runmat_runtime::operations::OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY.HOST_PLACEMENT_UNAVAILABLE",
            error_type: runmat_runtime::operations::OperationErrorType::Backend,
            retryable: false,
            severity: runmat_runtime::operations::OperationErrorSeverity::Fatal,
        },
        format!("native {kind} placement is unavailable: {error}"),
        std::collections::BTreeMap::new(),
    )
}

pub async fn execute_fea_path(
    path: PathBuf,
    root: crate::cli::FeaRootArgs,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    json: bool,
    execution: ExecutionTargetArgs,
) -> Result<()> {
    command::run(path, root, execution, json, cli, config).await
}

pub(crate) fn print_study_run_summary(data: &AnalysisStudyRunData, verbose: bool) {
    let styles = presentation::stdout();
    println!(
        "{}: {}",
        styles.success("FEA run complete"),
        styles.identifier(&data.study_id)
    );
    println!("  {} {}", styles.label("steps:"), data.steps.len());
    println!("  {} {:?}", styles.label("status:"), data.run_status);
    println!(
        "  {} {}",
        styles.label("numerically acceptable:"),
        data.numerically_acceptable
    );
    println!(
        "  {} {:?}",
        styles.label("solver:"),
        data.solver_convergence
    );
    println!("  {} {:?}", styles.label("quality:"), data.result_quality);
    println!(
        "  {} {}",
        styles.label("evidence:"),
        styles.path(&data.evidence_artifact_path)
    );
    for step in &data.steps {
        println!(
            "  step {}: {:?} / {:?} / {:?} on {:?}",
            styles.identifier(step.procedure_step.step_id.as_str()),
            step.procedure_step.family,
            step.procedure_step.procedure,
            step.procedure_step.solution_space,
            step.execution.backend
        );
        println!("    run id: {}", styles.identifier(&step.run_id));
        println!(
            "    results: {}",
            styles.help(format!("fea.results(\"{}\")", step.run_id))
        );
        for reason in &step.quality_reasons {
            println!("    quality {:?}: {}", reason.code, reason.detail);
        }
    }
    if verbose {
        println!("  sequence identity: {:?}", data.sequence_identity);
        println!("  study fingerprint: {}", data.study_fingerprint);
        println!("  operations:");
        for operation in &data.operation_sequence {
            println!("    {operation}");
        }
    }
}

pub(crate) fn print_sweep_run_summary(data: &AnalysisStudySweepData) {
    let styles = presentation::stdout();
    println!(
        "{}: {}",
        styles.success("FEA sweep complete"),
        styles.identifier(&data.sweep_id)
    );
    println!("  studies: {}", data.study_count);
    println!("  succeeded: {}", data.success_count);
    println!("  failed: {}", data.failed_count);
    println!(
        "  {} {}",
        styles.label("evidence:"),
        styles.path(&data.evidence_artifact_path)
    );
    if !data.run_entries.is_empty() {
        println!("  runs:");
        for entry in &data.run_entries {
            println!(
                "    {}  {} steps  {:?}  numerically acceptable: {}",
                entry.study_id,
                entry.result.steps.len(),
                entry.result.run_status,
                entry.result.numerically_acceptable
            );
        }
    }
    if !data.failure_entries.is_empty() {
        println!("  {}", styles.error("failures:"));
        for entry in &data.failure_entries {
            println!(
                "    {} [{}]: {} ({})",
                entry.study_id,
                entry.kind.as_str(),
                entry.message,
                entry.error_code
            );
        }
    }
}

pub(crate) fn print_envelope<T: Serialize>(
    envelope: &OperationEnvelope<T>,
    include_metadata: bool,
) -> Result<()> {
    if include_metadata {
        println!(
            "{}",
            serde_json::to_string_pretty(envelope).context("Failed to serialize FEA result")?
        );
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(&envelope.data)
                .context("Failed to serialize FEA result")?
        );
    }
    Ok(())
}

fn report_operation_error(error: OperationErrorEnvelope) -> anyhow::Error {
    match serde_json::to_string_pretty(&error) {
        Ok(payload) => eprintln!("{payload}"),
        Err(_) => eprintln!("{}: {}", error.error_code, error.message.replace('\n', " ")),
    }
    AlreadyReportedCliError.into()
}
