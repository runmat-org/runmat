//! Governed family orchestration; numerical and publication owners retain authority.
use super::prepared_validation::validate_prepared_execution;
use super::*;

/// Executes an authentic prepared study through the governed family coordinators.
/// Imported assessment functions require checked-source or frozen-callable completion.
pub async fn execute_prepared_fea_study_locally(
    prepared: PreparedFeaStudy,
    options: &LocalFeaExecutionOptions,
) -> Result<CompletedLocalFeaCase, String> {
    validate_prepared_execution(&prepared, options)?;
    use runmat_runtime::analysis::fea_run_admission::{
        FeaRunKind, FeaRunOutcome, FeaRunRequest, FeaRunScope,
    };
    let workers = options.workers.unwrap_or(prepared.workers).max(1);
    let threads = prepared.root.local_execution.thread_count();
    crate::machine_capacity::study_allocation(workers, Some(u32::from(threads.get())))?;
    let mut request = FeaRunRequest::local(
        FeaRunKind::MeshAndSolve,
        u64::from(workers) * u64::from(threads.get()) * 1000,
        prepared.root.authored_study_digest,
    )
    .map_err(|e| format!("{}: {e}", e.error_code()))?;
    request.site = FeaRunScope::admission_site().map_err(|e| format!("{}: {e}", e.error_code()))?;
    let scope = FeaRunScope::admit(request).map_err(|e| format!("{}: {e}", e.error_code()))?;
    let result = match scope
        .run_checked_result_future(execute_prepared_with_source(prepared, options, None))
        .await
    {
        Ok(result) => result,
        Err(failure) => {
            let _ = scope.settle(failure.outcome());
            return Err(failure.to_string());
        }
    };
    let outcome = if runmat_runtime::interrupt::is_cancelled() {
        FeaRunOutcome::Cancelled
    } else if result.is_ok() {
        FeaRunOutcome::Completed
    } else {
        FeaRunOutcome::Failed
    };
    let settlement = scope.settle(outcome);
    match result {
        Err(original) => Err(original),
        Ok(value) => settlement
            .map(|()| value)
            .map_err(|error| format!("{}: {error}", error.error_code())),
    }
}

pub(super) async fn execute_prepared_with_source(
    prepared: PreparedFeaStudy,
    options: &LocalFeaExecutionOptions,
    source: Option<&runmat_core::CheckedFeaSessionSource>,
) -> Result<CompletedLocalFeaCase, String> {
    validate_prepared_execution(&prepared, options)?;
    let case_id = prepared.realized.study_id.clone();
    let workers = options.workers.unwrap_or(prepared.workers).max(1);
    let root = crate::prepare_structural_study_root(prepared.root).map_err(|e| e.to_string())?;
    let completed = crate::execute_prepared_study_locally_with_observers(
        crate::NativeExecutionConfig::for_current_executable().map_err(|e| e.to_string())?,
        root,
        workers,
        crate::LocalStructuralStudyObservers {
            progress: options.progress_observer.clone(),
            mesh: options.mesh_observer.clone(),
        },
    )
    .await
    .map_err(|error| error.to_string())?;
    let result = match completed {
        crate::LocalStudyCompletion::Structural(result) => {
            CompletedFeaStudyResult::Structural(result)
        }
        crate::LocalStudyCompletion::Family(result) => *result,
    };
    let mut publication = prepare_study_publication(&result, "typed-local")?;
    publication
        .admit_accelerator_placement(&options.accelerator_placement)
        .map_err(|error| error.to_string())?;
    let publication = complete_prepared_publication(source, publication).await?;
    Ok(CompletedLocalFeaCase {
        case_id,
        result,
        publication,
    })
}

/// Prepare the atomic publication of any completed authored study.
///
/// Local and remote completions use this one function. `placement` only labels the completion
/// identity; the definition digest and the authenticated completion root fence the run.
pub fn prepare_study_publication(
    result: &CompletedFeaStudyResult,
    placement: &str,
) -> Result<runmat_runtime::analysis::PreparedStudyPublication, String> {
    use runmat_runtime::analysis::{AnalysisStudyCompletionIdentity, PrecisionMode, QualityPolicy};
    let context = || runmat_runtime::operations::OperationContext::new(None, None);
    // Every execution is its own attempt of the definition: a rerun publishes a fresh attempt
    // rather than colliding with, or silently reusing, an earlier publication.
    let attempt = AnalysisStudyCompletionIdentity::fresh_attempt();
    // The definition run id is derived from the authored definition alone; the attempt fences
    // each execution's publication (its authenticated completion root is part of that attempt).
    let identity = |digest: runmat_execution::Digest, _root: &runmat_execution::Digest| {
        AnalysisStudyCompletionIdentity::with_attempt(
            format!(
                "{placement}:{}",
                digest.to_string().trim_start_matches("sha256:")
            ),
            attempt.clone(),
        )
    };
    let publication = match result {
        CompletedFeaStudyResult::Electrical(result) => {
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            runmat_runtime::analysis::prepare_electrical_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &runmat_runtime::analysis::AnalysisElectricalRunOptions {
                    deterministic_mode: true,
                    precision_mode: PrecisionMode::Fp64,
                    quality_policy: QualityPolicy::Strict,
                    residual_target: result.completion.workload.relative_residual_target,
                },
                digest,
                context(),
                identity(digest, &result.completion.root.logical_digest)?,
            )
        }
        CompletedFeaStudyResult::Magnetic(result) => {
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            runmat_runtime::analysis::prepare_magnetic_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &runmat_runtime::analysis::AnalysisMagneticRunOptions {
                    deterministic_mode: true,
                    precision_mode: PrecisionMode::Fp64,
                    quality_policy: QualityPolicy::Strict,
                    residual_target: result.completion.workload.relative_residual_target,
                },
                digest,
                context(),
                identity(digest, &result.completion.root.logical_digest)?,
            )
        }
        CompletedFeaStudyResult::Maxwell(result) => {
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            runmat_runtime::analysis::prepare_maxwell_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &runmat_runtime::analysis::AnalysisMaxwellRunOptions {
                    precision_mode: PrecisionMode::Fp64,
                    residual_target: result.completion.workload.relative_residual_target,
                },
                digest,
                context(),
                identity(digest, &result.completion.root.logical_digest)?,
            )
        }
        CompletedFeaStudyResult::Nonlinear(result) => {
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            runmat_runtime::analysis::prepare_nonlinear_solid_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                digest,
                context(),
                identity(digest, &result.completion.root.logical_digest)?,
            )
        }
        CompletedFeaStudyResult::Fluid(result) => {
            let study = result
                .completion
                .compiled()
                .definition()
                .map_err(|e| e.to_string())?;
            let digest = study.digest();
            runmat_runtime::analysis::prepare_fluid_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &runmat_runtime::analysis::AnalysisCfdRunOptions {
                    precision_mode: PrecisionMode::Fp64,
                    numerical: result.completion.workload().policy,
                    geometry: result.completion.workload().geometry_policy,
                },
                study,
                context(),
                identity(digest, &result.completion.root.logical_digest)?,
            )
        }
        CompletedFeaStudyResult::Acoustic(result) => {
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            runmat_runtime::analysis::prepare_acoustic_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &runmat_runtime::analysis::AnalysisAcousticRunOptions {
                    deterministic_mode: true,
                    precision_mode: PrecisionMode::Fp64,
                    quality_policy: QualityPolicy::Strict,
                    mode_count: 1,
                    residual_warn_threshold: result
                        .completion
                        .entries
                        .iter()
                        .map(|entry| entry.workload.solver.relative_tolerance)
                        .reduce(f64::min)
                        .ok_or("acoustic completion omits its frequency entries")?,
                },
                digest,
                context(),
                identity(digest, &result.completion.root.logical_digest)?,
            )
        }
        CompletedFeaStudyResult::AcousticResponse(result) => {
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            runmat_runtime::analysis::prepare_acoustic_response_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &runmat_runtime::analysis::AnalysisAcousticResponseRunOptions {
                    precision_mode: PrecisionMode::Fp64,
                    residual_target: result.completion.workload.solver.relative_tolerance(),
                },
                digest,
                context(),
                identity(digest, &result.completion.root.logical_digest)?,
            )
        }
        CompletedFeaStudyResult::Structural(result) => {
            let digest = result.definition_digest()?;
            runmat_runtime::analysis::prepare_structural_study_publication(
                result,
                context(),
                AnalysisStudyCompletionIdentity::with_attempt(
                    format!(
                        "{placement}:{}",
                        digest.to_string().trim_start_matches("sha256:")
                    ),
                    attempt.clone(),
                )?,
            )
        }
        CompletedFeaStudyResult::LineParameters(result) => {
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            runmat_runtime::analysis::prepare_line_parameters_study_publication(
                &result.completion,
                &result.mesh,
                &result.meshing_evidence,
                &result.options,
                context(),
                identity(digest, &digest)?,
            )
        }
        CompletedFeaStudyResult::Thermal(result) => {
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            let workload = &result.completion.workload;
            // A fresh execution can retain different attempt/accounting objects even when its
            // scientific definition and fields are unchanged; it publishes as its own attempt.
            runmat_runtime::analysis::prepare_thermal_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &runmat_runtime::analysis::AnalysisThermalRunOptions {
                    deterministic_mode: true,
                    precision_mode: PrecisionMode::Fp64,
                    quality_policy: QualityPolicy::Strict,
                    snapshot_count: match workload.procedure {
                        runmat_analysis_fea::ThermalProcedureControl::Steady => 1,
                        runmat_analysis_fea::ThermalProcedureControl::BackwardEuler {
                            snapshot_count,
                        } => snapshot_count,
                    },
                    residual_warn_threshold: workload.residual_target,
                    solver_mesh_artifact_path: None,
                },
                digest,
                context(),
                identity(digest, &result.completion.root.logical_digest)?,
            )
        }
    };
    publication.map_err(|error| {
        // This native-host boundary carries text. Preserve the complete
        // operation envelope, including recovery receipts and the original
        // durability cause, rather than only its user-facing summary.
        serde_json::to_string(&error).unwrap_or_else(|encoding| {
            format!("operation error {error:?}; diagnostic encoding failed: {encoding}")
        })
    })
}

async fn complete_prepared_publication(
    source: Option<&runmat_core::CheckedFeaSessionSource>,
    publication: runmat_runtime::analysis::PreparedStudyPublication,
) -> Result<runmat_runtime::analysis::AnalysisStudyRunData, String> {
    match source {
        Some(source) => assessment::complete(source, publication).await,
        None => assessment::complete_frozen_locally(publication, &[]).await,
    }
    .map_err(|error| error.to_string())
}
