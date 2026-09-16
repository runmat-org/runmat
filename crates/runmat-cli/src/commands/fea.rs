use anyhow::{Context, Result};
use log::info;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution_runner_native::NativeStructuralStudyPlacement;
use runmat_runtime::analysis::{
    analysis_execute_structural_study_with_placement,
    analysis_execute_structural_sweep_with_placement, load_fea_document_from_path_async,
    AnalysisStudyRunData, AnalysisStudySweepData, FeaResolvedDocument,
};
use runmat_runtime::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use serde::Serialize;
use std::path::PathBuf;

use crate::cli::{Cli, ExecutionTarget};
use crate::commands::accel::dump_provider_telemetry_if_requested;
use crate::presentation;
use crate::AlreadyReportedCliError;

pub(super) mod callables;
pub(crate) mod remote;
mod signal_materialization;

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
            signal_materialization::materialize_runtime_signal(request).map_err(|error| {
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
    execution: ExecutionTarget,
) -> Result<()> {
    info!("Executing FEA document: {path:?}");
    if !json {
        let styles = presentation::stderr();
        eprintln!(
            "{} {}",
            styles.heading("Running"),
            styles.path(path.display())
        );
        eprintln!("  {}", styles.muted("loading study document"));
    }
    let document = load_fea_document_from_path_async(&path)
        .await
        .map_err(|err| anyhow::anyhow!("Failed to load FEA file {}: {err}", path.display()))?;
    let context = OperationContext::new(None, None);

    match document {
        FeaResolvedDocument::Study(mut spec) => {
            if !json {
                let styles = presentation::stderr();
                eprintln!(
                    "  {} {} ({} procedure step{})",
                    styles.label("study:"),
                    styles.identifier(&spec.study_id),
                    spec.execution.steps.len(),
                    if spec.execution.steps.len() == 1 {
                        ""
                    } else {
                        "s"
                    }
                );
                eprintln!(
                    "  {} {} regions, {} meshes",
                    styles.label("geometry:"),
                    spec.geometry.regions.len(),
                    spec.geometry.meshes.len()
                );
                eprintln!("  {}", styles.info("validating, planning, and solving"));
            }
            let envelope = match execution {
                ExecutionTarget::Local => {
                    signal_materialization::materialize_study(&path, &mut spec, cli, config)
                        .await?;
                    let placement = NativeStructuralStudyPlacement::for_current_executable(
                        crate::diagnostics::parser_compat(config.language.compat),
                    )?;
                    Some(
                        analysis_execute_structural_study_with_placement(
                            &spec,
                            context.clone(),
                            &placement,
                        )
                        .await
                        .map_err(report_operation_error)?,
                    )
                }
                ExecutionTarget::Remote(target) => {
                    remote::execute_study(&path, *spec, context, target, cli, config, json).await?
                }
            };
            if let Some(envelope) = envelope {
                if json {
                    print_envelope(&envelope, config.runtime.verbose)?;
                } else {
                    print_study_run_summary(&envelope.data, config.runtime.verbose);
                }
            }
        }
        FeaResolvedDocument::Sweep(mut spec) => {
            if !json {
                let styles = presentation::stderr();
                eprintln!(
                    "  {} {} ({} studies, fail_fast: {})",
                    styles.label("sweep:"),
                    styles.identifier(&spec.sweep_id),
                    spec.studies.len(),
                    spec.fail_fast
                );
                eprintln!(
                    "  {}",
                    styles.info("validating, planning, and solving studies")
                );
            }
            let envelope = match execution {
                ExecutionTarget::Local => {
                    signal_materialization::materialize_sweep(&path, &mut spec, cli, config)
                        .await?;
                    let placement = NativeStructuralStudyPlacement::for_current_executable(
                        crate::diagnostics::parser_compat(config.language.compat),
                    )?;
                    Some(
                        analysis_execute_structural_sweep_with_placement(
                            &spec, context, &placement,
                        )
                        .await
                        .map_err(report_operation_error)?,
                    )
                }
                ExecutionTarget::Remote(target) => {
                    remote::execute_sweep(&path, spec.clone(), context, target, cli, config, json)
                        .await?
                }
            };
            if let Some(envelope) = envelope {
                if json {
                    print_envelope(&envelope, config.runtime.verbose)?;
                } else {
                    print_sweep_run_summary(&envelope.data);
                }
            }
        }
    }

    dump_provider_telemetry_if_requested();
    Ok(())
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
