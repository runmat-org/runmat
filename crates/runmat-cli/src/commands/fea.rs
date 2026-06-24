use anyhow::{Context, Result};
use log::info;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_runtime::analysis::{
    analysis_run_study_op, analysis_run_study_sweep_op, load_fea_document_from_path_async,
    AnalysisStudyRunData, AnalysisStudySweepData, FeaResolvedDocument,
};
use runmat_runtime::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use serde::Serialize;
use std::path::PathBuf;

use crate::cli::Cli;
use crate::commands::accel::dump_provider_telemetry_if_requested;
use crate::AlreadyReportedCliError;

pub async fn execute_fea_path(
    path: PathBuf,
    _cli: &Cli,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<()> {
    info!("Executing FEA document: {path:?}");
    if !json {
        eprintln!("Running {}", path.display());
        eprintln!("  loading study document");
    }
    let document = load_fea_document_from_path_async(&path)
        .await
        .map_err(|err| anyhow::anyhow!("Failed to load FEA file {}: {err}", path.display()))?;
    let context = OperationContext::new(None, None);

    match document {
        FeaResolvedDocument::Study(spec) => {
            if !json {
                eprintln!(
                    "  study: {} ({:?}, {:?})",
                    spec.study_id, spec.run_kind, spec.backend
                );
                eprintln!(
                    "  geometry: {} regions, {} meshes",
                    spec.geometry.regions.len(),
                    spec.geometry.meshes.len()
                );
                eprintln!("  validating, planning, and solving");
            }
            let envelope = analysis_run_study_op(&spec, context).map_err(report_operation_error)?;
            if json {
                print_envelope(&envelope, config.runtime.verbose)?;
            } else {
                print_study_run_summary(&envelope.data, config.runtime.verbose);
            }
        }
        FeaResolvedDocument::Sweep(spec) => {
            if !json {
                eprintln!(
                    "  sweep: {} ({} studies, fail_fast: {})",
                    spec.sweep_id,
                    spec.studies.len(),
                    spec.fail_fast
                );
                eprintln!("  validating, planning, and solving studies");
            }
            let envelope =
                analysis_run_study_sweep_op(&spec, context).map_err(report_operation_error)?;
            if json {
                print_envelope(&envelope, config.runtime.verbose)?;
            } else {
                print_sweep_run_summary(&envelope.data);
            }
        }
    }

    dump_provider_telemetry_if_requested();
    Ok(())
}

fn print_study_run_summary(data: &AnalysisStudyRunData, verbose: bool) {
    println!("FEA run complete: {}", data.study_id);
    println!("  run id: {}", data.run_id);
    println!("  kind: {:?}", data.run_kind);
    println!("  backend: {:?}", data.backend);
    println!(
        "  status: {:?} (publishable: {})",
        data.run_status, data.publishable
    );
    println!("  solver: {:?}", data.solver_convergence);
    println!("  quality: {:?}", data.result_quality);
    println!("  evidence: {}", data.evidence_artifact_path);
    println!("  results: fea.results(\"{}\")", data.run_id);
    if !data.quality_reasons.is_empty() {
        println!("  quality reasons:");
        for reason in &data.quality_reasons {
            println!("    {:?}: {}", reason.code, reason.detail);
        }
    }
    if verbose {
        println!(
            "  operation: {} ({})",
            data.run_operation, data.run_op_version
        );
        println!("  study fingerprint: {}", data.study_fingerprint);
        println!("  operations:");
        for operation in &data.operation_sequence {
            println!("    {operation}");
        }
    }
}

fn print_sweep_run_summary(data: &AnalysisStudySweepData) {
    println!("FEA sweep complete: {}", data.sweep_id);
    println!("  studies: {}", data.study_count);
    println!("  succeeded: {}", data.success_count);
    println!("  failed: {}", data.failed_count);
    println!("  evidence: {}", data.evidence_artifact_path);
    if !data.run_entries.is_empty() {
        println!("  runs:");
        for entry in &data.run_entries {
            println!(
                "    {}  {}  {:?}  publishable: {}",
                entry.study_id, entry.run_id, entry.run_status, entry.publishable
            );
        }
    }
    if !data.failure_entries.is_empty() {
        println!("  failures:");
        for entry in &data.failure_entries {
            println!(
                "    {}: {} ({})",
                entry.study_id, entry.message, entry.error_code
            );
        }
    }
}

fn print_envelope<T: Serialize>(
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
