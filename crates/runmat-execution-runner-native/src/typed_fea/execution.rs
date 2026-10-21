//! Governed family orchestration; numerical and publication owners retain authority.
use super::prepared_validation::validate_prepared_execution;
use super::*;

/// Executes an authentic prepared study through the governed family coordinators.
/// Imported assessment functions require checked-source or frozen-callable completion.
pub async fn execute_prepared_fea_study_locally(
    prepared: PreparedFeaStudy,
    options: &LocalFeaExecutionOptions,
) -> Result<CompletedLocalFeaCase, String> {
    execute_prepared_with_source(prepared, options, None).await
}

pub(super) async fn execute_prepared_with_source(
    prepared: PreparedFeaStudy,
    options: &LocalFeaExecutionOptions,
    source: Option<&runmat_core::CheckedFeaSessionSource>,
) -> Result<CompletedLocalFeaCase, String> {
    validate_prepared_execution(&prepared, options)?;
    let case_id = prepared.realized.study_id.clone();
    let workers = options.workers.unwrap_or(prepared.workers).max(1);
    let (result, publication) = match prepared.root {
        PreparedFeaStudyRoot::Electrical(root) => {
            let result = electrical_execution::execute(root, workers, options).await?;
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            let publication_options = runmat_runtime::analysis::AnalysisElectricalRunOptions {
                deterministic_mode: true,
                precision_mode: runmat_runtime::analysis::PrecisionMode::Fp64,
                quality_policy: runmat_runtime::analysis::QualityPolicy::Strict,
                residual_target: result.completion.workload.relative_residual_target,
            };
            let identity =
                runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
                    "typed-local:{}:{}",
                    digest.to_string().trim_start_matches("sha256:"),
                    result.completion.root.logical_digest,
                ))?;
            let publication = runmat_runtime::analysis::prepare_electrical_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &publication_options,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?;
            let publication = complete_prepared_publication(source, publication).await?;
            (
                CompletedFeaStudyResult::Electrical(Box::new(result)),
                publication,
            )
        }
        PreparedFeaStudyRoot::AcousticResponse(root) => {
            let result = acoustic_response_execution::execute(root, workers, options).await?;
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            let publication_options = runmat_runtime::analysis::AnalysisAcousticResponseRunOptions {
                precision_mode: runmat_runtime::analysis::PrecisionMode::Fp64,
                residual_target: result.completion.workload.solver.relative_tolerance(),
            };
            let identity =
                runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
                    "typed-local:{}:{}",
                    digest.to_string().trim_start_matches("sha256:"),
                    result.completion.root.logical_digest,
                ))?;
            let publication =
                runmat_runtime::analysis::prepare_acoustic_response_study_publication(
                    &result.completion,
                    &result.source,
                    &result.mesh,
                    &result.meshing_evidence,
                    &publication_options,
                    digest,
                    runmat_runtime::operations::OperationContext::new(None, None),
                    identity,
                )
                .map_err(|e| e.message)?;
            let publication = complete_prepared_publication(source, publication).await?;
            (
                CompletedFeaStudyResult::AcousticResponse(Box::new(result)),
                publication,
            )
        }
        PreparedFeaStudyRoot::Magnetic(root) => {
            let result = magnetic_execution::execute(root, workers, options).await?;
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            let publication_options = runmat_runtime::analysis::AnalysisMagneticRunOptions {
                deterministic_mode: true,
                precision_mode: runmat_runtime::analysis::PrecisionMode::Fp64,
                quality_policy: runmat_runtime::analysis::QualityPolicy::Strict,
                residual_target: result.completion.workload.relative_residual_target,
            };
            let identity =
                runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
                    "typed-local:{}:{}",
                    digest.to_string().trim_start_matches("sha256:"),
                    result.completion.root.logical_digest,
                ))?;
            let publication = runmat_runtime::analysis::prepare_magnetic_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &publication_options,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?;
            let publication = complete_prepared_publication(source, publication).await?;
            (
                CompletedFeaStudyResult::Magnetic(Box::new(result)),
                publication,
            )
        }
        PreparedFeaStudyRoot::Maxwell(root) => {
            let result = maxwell_execution::execute(root, workers, options).await?;
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            let publication_options = runmat_runtime::analysis::AnalysisMaxwellRunOptions {
                precision_mode: runmat_runtime::analysis::PrecisionMode::Fp64,
                residual_target: result.completion.workload.relative_residual_target,
            };
            let identity =
                runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
                    "typed-local:{}:{}",
                    digest.to_string().trim_start_matches("sha256:"),
                    result.completion.root.logical_digest,
                ))?;
            let publication = runmat_runtime::analysis::prepare_maxwell_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &publication_options,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?;
            let publication = complete_prepared_publication(source, publication).await?;
            (
                CompletedFeaStudyResult::Maxwell(Box::new(result)),
                publication,
            )
        }
        PreparedFeaStudyRoot::Nonlinear(root) => {
            let result = nonlinear_execution::execute(*root, workers, options).await?;
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            let identity =
                runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
                    "typed-local:{}:{}",
                    digest, result.completion.root.logical_digest,
                ))?;
            let publication = runmat_runtime::analysis::prepare_nonlinear_solid_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?;
            let publication = complete_prepared_publication(source, publication).await?;
            (
                CompletedFeaStudyResult::Nonlinear(Box::new(result)),
                publication,
            )
        }
        PreparedFeaStudyRoot::Fluid(root) => {
            let result = fluid_execution::execute(root, workers, options).await?;
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            let publication_options = runmat_runtime::analysis::AnalysisCfdRunOptions {
                precision_mode: runmat_runtime::analysis::PrecisionMode::Fp64,
                numerical: result.completion.workload.policy,
                geometry: result.completion.workload.geometry_policy,
            };
            let identity =
                runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
                    "typed-local:{}:{}",
                    digest.to_string().trim_start_matches("sha256:"),
                    result.completion.root.logical_digest,
                ))?;
            let publication = runmat_runtime::analysis::prepare_fluid_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &publication_options,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?;
            let publication = complete_prepared_publication(source, publication).await?;
            (
                CompletedFeaStudyResult::Fluid(Box::new(result)),
                publication,
            )
        }
        PreparedFeaStudyRoot::Acoustic(root) => {
            let result = acoustic_execution::execute(root, workers, options).await?;
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            let publication_options = runmat_runtime::analysis::AnalysisAcousticRunOptions {
                deterministic_mode: true,
                precision_mode: runmat_runtime::analysis::PrecisionMode::Fp64,
                quality_policy: runmat_runtime::analysis::QualityPolicy::Strict,
                mode_count: 1,
                residual_warn_threshold: result
                    .completion
                    .entries
                    .iter()
                    .map(|entry| entry.workload.solver.relative_tolerance)
                    .reduce(f64::min)
                    .ok_or("acoustic completion omits its frequency entries")?,
            };
            let identity =
                runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
                    "typed-local:{}:{}",
                    digest.to_string().trim_start_matches("sha256:"),
                    result.completion.root.logical_digest,
                ))?;
            let publication = runmat_runtime::analysis::prepare_acoustic_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &publication_options,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?;
            let publication = complete_prepared_publication(source, publication).await?;
            (
                CompletedFeaStudyResult::Acoustic(Box::new(result)),
                publication,
            )
        }
        PreparedFeaStudyRoot::Structural(root) => {
            let result = crate::execute_prepared_structural_study_locally_with_observers(
                crate::NativeExecutionConfig::for_current_executable()
                    .map_err(|error| error.to_string())?,
                root,
                workers,
                crate::LocalStructuralStudyObservers {
                    progress: options.progress_observer.clone(),
                    mesh: options.mesh_observer.clone(),
                },
            )
            .await
            .map_err(|error| error.to_string())?;
            let digest = result.definition_digest()?;
            let identity =
                runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
                    "typed-local:{}",
                    digest.to_string().trim_start_matches("sha256:")
                ))?;
            let publication = runmat_runtime::analysis::prepare_structural_study_publication(
                &result,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|error| error.message)?;
            let publication = complete_prepared_publication(source, publication).await?;
            (
                CompletedFeaStudyResult::Structural(Box::new(result)),
                publication,
            )
        }
        PreparedFeaStudyRoot::Thermal(root) => {
            let result = thermal_execution::execute(root, workers, options).await?;
            let digest = result
                .completion
                .compiled
                .definition_digest()
                .map_err(|e| e.to_string())?;
            let workload = &result.completion.workload;
            let publication_options = runmat_runtime::analysis::AnalysisThermalRunOptions {
                deterministic_mode: true,
                precision_mode: runmat_runtime::analysis::PrecisionMode::Fp64,
                quality_policy: runmat_runtime::analysis::QualityPolicy::Strict,
                snapshot_count: match workload.procedure {
                    runmat_analysis_fea::ThermalProcedureControl::Steady => 1,
                    runmat_analysis_fea::ThermalProcedureControl::BackwardEuler {
                        snapshot_count,
                    } => snapshot_count,
                },
                residual_warn_threshold: workload.residual_target,
                solver_mesh_artifact_path: None,
            };
            // A fresh execution can retain different attempt/accounting objects even
            // when its scientific definition and fields are unchanged. Fence publication
            // by the authenticated completion, not the authored definition alone.
            let identity =
                runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!(
                    "typed-local:{}:{}",
                    digest.to_string().trim_start_matches("sha256:"),
                    result.completion.root.logical_digest
                ))?;
            let publication = runmat_runtime::analysis::prepare_thermal_study_publication(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &publication_options,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?;
            let publication = complete_prepared_publication(source, publication).await?;
            (
                CompletedFeaStudyResult::Thermal(Box::new(result)),
                publication,
            )
        }
    };
    Ok(CompletedLocalFeaCase {
        case_id,
        result,
        publication,
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
