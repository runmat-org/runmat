use anyhow::{Context, Result};
use log::info;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_runtime::analysis::{
    analysis_complete_structural_study_op, analysis_complete_study_sweep_op,
    analysis_prepare_structural_study_op, analysis_study_sweep_identity,
    load_fea_document_from_path_async, AnalysisStudyRunData, AnalysisStudySweepData,
    AnalysisStudySweepRunOutcome, AnalysisStudySweepSpec, FeaResolvedDocument,
};
use runmat_runtime::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use serde::Serialize;
use std::path::PathBuf;

use crate::cli::{Cli, ExecutionTarget};
use crate::commands::accel::dump_provider_telemetry_if_requested;
use crate::presentation;
use crate::AlreadyReportedCliError;

pub(super) mod callables;
mod portable;
pub(crate) mod remote;
mod signal_materialization;

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
                    let prepared = analysis_prepare_structural_study_op(&spec, context.clone())
                        .map_err(report_operation_error)?;
                    let numerical =
                        portable::execute_structural_study(&spec, &prepared, config).await;
                    Some(
                        analysis_complete_structural_study_op(prepared, numerical)
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
                    Some(execute_local_sweep(&spec, context, config).await?)
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

async fn execute_local_sweep(
    spec: &AnalysisStudySweepSpec,
    context: OperationContext,
    config: &RunMatRuntimeConfig,
) -> Result<OperationEnvelope<AnalysisStudySweepData>> {
    analysis_study_sweep_identity(spec, &context).map_err(report_operation_error)?;
    let mut prepared = spec
        .studies
        .iter()
        .map(|study| analysis_prepare_structural_study_op(study, context.clone()))
        .map(|result| match result {
            Ok(prepared) => (Some(prepared), None),
            Err(error) => (None, Some(AnalysisStudySweepRunOutcome::Failed(error))),
        })
        .collect::<Vec<_>>();

    if spec.fail_fast && prepared.iter().any(|(_, outcome)| outcome.is_some()) {
        let outcomes = prepared
            .into_iter()
            .map(|(_, outcome)| outcome.unwrap_or(AnalysisStudySweepRunOutcome::NotRun))
            .collect();
        return analysis_complete_study_sweep_op(spec, context, outcomes)
            .map_err(report_operation_error);
    }

    let placement_indices = prepared
        .iter()
        .enumerate()
        .filter_map(|(index, (entry, _))| entry.as_ref().map(|_| index))
        .collect::<Vec<_>>();
    if !placement_indices.is_empty() {
        let numerical_results = {
            let placement_entries = placement_indices
                .iter()
                .map(|index| {
                    prepared[*index]
                        .0
                        .as_ref()
                        .map(|structural| portable::LocalSweepEntry {
                            study: &spec.studies[*index],
                            prepared: structural,
                        })
                        .ok_or_else(|| {
                            anyhow::anyhow!("prepared structural sweep entry {index} disappeared")
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            portable::execute_structural_sweep(&placement_entries, config, spec.fail_fast)
                .await
                .unwrap_or_else(|error| {
                    placement_entries
                        .iter()
                        .enumerate()
                        .map(|(index, _)| {
                            if spec.fail_fast && index > 0 {
                                portable::LocalSweepOutcome::NotRun
                            } else {
                                portable::LocalSweepOutcome::Completed(Err(error.clone()))
                            }
                        })
                        .collect()
                })
        };
        if numerical_results.len() != placement_indices.len() {
            anyhow::bail!(
                "local structural sweep returned {} results for {} prepared studies",
                numerical_results.len(),
                placement_indices.len()
            );
        }
        for (index, numerical_result) in placement_indices.into_iter().zip(numerical_results) {
            let portable::LocalSweepOutcome::Completed(numerical_result) = numerical_result else {
                prepared[index].1 = Some(AnalysisStudySweepRunOutcome::NotRun);
                continue;
            };
            let structural = prepared[index].0.take().ok_or_else(|| {
                anyhow::anyhow!("prepared structural sweep entry {index} disappeared")
            })?;
            prepared[index].1 = Some(
                match analysis_complete_structural_study_op(structural, numerical_result) {
                    Ok(run) => AnalysisStudySweepRunOutcome::Succeeded(Box::new(run)),
                    Err(error) => AnalysisStudySweepRunOutcome::Failed(error),
                },
            );
        }
    }

    let outcomes = prepared
        .into_iter()
        .enumerate()
        .map(|(index, (_, outcome))| {
            outcome.ok_or_else(|| anyhow::anyhow!("sweep entry {index} has no terminal outcome"))
        })
        .collect::<Result<Vec<_>>>()?;
    analysis_complete_study_sweep_op(spec, context, outcomes).map_err(report_operation_error)
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
