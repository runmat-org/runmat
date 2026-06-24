use anyhow::{Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_runtime::analysis::{
    analysis_plan_study_op, analysis_plan_study_sweep_op, analysis_validate_study_op,
    analysis_validate_study_sweep_op, is_fea_file_path, load_fea_document_from_path_async,
    AnalysisStudyPlanData, AnalysisStudySweepPlanData, AnalysisStudySweepValidateData,
    AnalysisStudyValidateResult, FeaResolvedDocument,
};
use runmat_runtime::operations::{OperationContext, OperationErrorEnvelope};
use serde::Serialize;
use serde_json::json;
use std::path::{Path, PathBuf};

use crate::cli::Cli;
use crate::commands::bytecode::emit_bytecode;
use crate::commands::script::resolve_script_input;
use crate::AlreadyReportedCliError;

pub async fn execute_check(
    file: PathBuf,
    _cli: &Cli,
    config: &RunMatRuntimeConfig,
    json: bool,
) -> Result<()> {
    let file = resolve_script_input(file)?;
    if is_fea_file_path(&file) {
        return check_fea_file(file, config, json).await;
    }
    if is_matlab_file(&file) {
        return check_m_file(file, config, json).await;
    }

    anyhow::bail!(
        "runmat check supports .m scripts and .fea documents; got {}",
        file.display()
    )
}

async fn check_fea_file(path: PathBuf, config: &RunMatRuntimeConfig, json: bool) -> Result<()> {
    if !json {
        eprintln!("Checking {}", path.display());
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
                eprintln!("  validating");
            }
            let validation = analysis_validate_study_op(&spec, context.clone())
                .map_err(report_operation_error)?;
            if !validation.data.valid {
                if json {
                    print_payload("study", "invalid", &validation.data, config.runtime.verbose)?;
                } else {
                    print_study_validation_failure(&validation.data);
                }
                return Err(AlreadyReportedCliError.into());
            }
            if !json {
                eprintln!("  planning");
            }
            let plan = analysis_plan_study_op(&spec, context).map_err(report_operation_error)?;
            if json {
                print_payload(
                    "study",
                    "valid",
                    &json!({
                        "validation": validation.data,
                        "plan": plan.data,
                    }),
                    config.runtime.verbose,
                )?;
            } else {
                print_study_check_summary(&validation.data, &plan.data, config.runtime.verbose);
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
                eprintln!("  validating");
            }
            let validation = analysis_validate_study_sweep_op(&spec, context.clone())
                .map_err(report_operation_error)?;
            if !validation.data.valid {
                if json {
                    print_payload("sweep", "invalid", &validation.data, config.runtime.verbose)?;
                } else {
                    print_sweep_validation_failure(&validation.data);
                }
                return Err(AlreadyReportedCliError.into());
            }
            if !json {
                eprintln!("  planning");
            }
            let plan =
                analysis_plan_study_sweep_op(&spec, context).map_err(report_operation_error)?;
            if json {
                print_payload(
                    "sweep",
                    "valid",
                    &json!({
                        "validation": validation.data,
                        "plan": plan.data,
                    }),
                    config.runtime.verbose,
                )?;
            } else {
                print_sweep_check_summary(&validation.data, &plan.data);
            }
        }
    }
    Ok(())
}

async fn check_m_file(path: PathBuf, config: &RunMatRuntimeConfig, json: bool) -> Result<()> {
    let content = runmat_filesystem::read_to_string_async(&path)
        .await
        .with_context(|| format!("Failed to read script file: {}", path.display()))?;
    emit_bytecode(&content, config, Some(path.to_string_lossy().as_ref()))
        .with_context(|| format!("Failed to check {}", path.display()))?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "document_kind": "script",
                "status": "valid",
                "path": path,
            }))
            .context("Failed to serialize check result")?
        );
    } else {
        println!("OK {}", path.display());
    }
    Ok(())
}

fn print_study_check_summary(
    validation: &AnalysisStudyValidateResult,
    plan: &AnalysisStudyPlanData,
    verbose: bool,
) {
    println!("OK {}", plan.study_id);
    println!("  kind: {:?}", plan.run_kind);
    println!("  backend: {:?}", plan.backend);
    println!("  model: {}", plan.model_id);
    println!("  validation: passed ({} issues)", validation.issues.len());
    println!("  run op: {} ({})", plan.run_operation, plan.run_op_version);
    println!("  evidence: {}", plan.evidence_artifact_path);
    if verbose {
        println!("  study fingerprint: {}", plan.study_fingerprint);
        println!("  operations:");
        for operation in &plan.operation_sequence {
            println!("    {operation}");
        }
    }
}

fn print_study_validation_failure(validation: &AnalysisStudyValidateResult) {
    println!("FAILED validation");
    println!("  evidence: {}", validation.evidence_artifact_path);
    for issue in &validation.issues {
        println!("  {}: {}", issue.code, issue.message);
    }
}

fn print_sweep_check_summary(
    validation: &AnalysisStudySweepValidateData,
    plan: &AnalysisStudySweepPlanData,
) {
    println!("OK {}", plan.sweep_id);
    println!("  studies: {}", plan.study_count);
    println!("  planned: {}", plan.planned_count);
    println!("  failed: {}", plan.failed_count);
    println!(
        "  validation: passed ({} study entries)",
        validation.study_entries.len()
    );
    println!("  evidence: {}", plan.evidence_artifact_path);
    for entry in &plan.plan_entries {
        println!(
            "  {}: {} ({})",
            entry.study_id, entry.run_operation, entry.run_op_version
        );
    }
}

fn print_sweep_validation_failure(validation: &AnalysisStudySweepValidateData) {
    println!("FAILED {}", validation.sweep_id);
    println!("  evidence: {}", validation.evidence_artifact_path);
    for entry in &validation.study_entries {
        if entry.valid {
            continue;
        }
        println!("  {}", entry.study_id);
        for issue in &entry.issues {
            println!("    {}: {}", issue.code, issue.message);
        }
    }
}

fn print_payload<T: Serialize>(
    document_kind: &'static str,
    status: &'static str,
    data: &T,
    include_metadata: bool,
) -> Result<()> {
    if include_metadata {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "document_kind": document_kind,
                "status": status,
                "data": data,
            }))
            .context("Failed to serialize check result")?
        );
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(data).context("Failed to serialize check result")?
        );
    }
    Ok(())
}

fn is_matlab_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("m"))
}

fn report_operation_error(error: OperationErrorEnvelope) -> anyhow::Error {
    match serde_json::to_string_pretty(&error) {
        Ok(payload) => eprintln!("{payload}"),
        Err(_) => eprintln!("{}: {}", error.error_code, error.message.replace('\n', " ")),
    }
    AlreadyReportedCliError.into()
}
