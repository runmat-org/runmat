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

pub(super) mod callables;
mod command;
mod input_materialization;
mod plan;
mod preparation;
pub(crate) mod remote;
mod run;
mod selection;
pub(crate) mod source;

pub async fn execute_command(
    command: crate::cli::FeaCommand,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    command::execute(command, cli, config).await
}

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
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    json: bool,
    execution: ExecutionTargetArgs,
) -> Result<()> {
    command::execute(
        crate::cli::FeaCommand::Run {
            file: path,
            root: crate::cli::FeaRootArgs::default(),
            execution,
            json,
        },
        cli,
        config,
    )
    .await
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
        "  {} {}",
        styles.label("production publishable:"),
        data.production_publishable
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
        for reason in &step.production_publication_reasons {
            println!("    publication {:?}: {}", reason.code, reason.detail);
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
                "    {}  {} steps  {:?}  numerically acceptable: {}  production publishable: {}",
                entry.study_id,
                entry.result.steps.len(),
                entry.result.run_status,
                entry.result.numerically_acceptable,
                entry.result.production_publishable
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
