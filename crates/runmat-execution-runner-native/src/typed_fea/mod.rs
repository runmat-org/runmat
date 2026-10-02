//! Native execution of checked `.fea` sources.
//!
//! This is the shared host boundary used by the CLI and Desktop. It consumes the exact checked
//! session product, resolves a typed root, materializes geometry, meshes, executes, and publishes
//! without interpreting authoring syntax or introducing host-specific physics defaults.

mod acoustic_execution;
mod callable;
mod electrical_execution;
mod fluid_execution;
mod prepared_validation;
use prepared_validation::validate_prepared_execution;
mod geometry;
mod input;
mod magnetic_execution;
mod maxwell_execution;
mod preparation;
mod thermal_execution;

#[cfg(all(test, feature = "occt-native"))]
mod tests;

#[cfg(all(test, feature = "occt-native"))]
#[path = "tests/prepared_execution.rs"]
mod prepared_execution_tests;

#[cfg(test)]
#[path = "tests/thermal_signals.rs"]
mod thermal_signals;

#[cfg(test)]
#[path = "tests/spatial_signals.rs"]
mod spatial_signals;

pub use acoustic_execution::CompletedAuthoredAcousticStudy;
pub use electrical_execution::CompletedAuthoredElectricalStudy;
pub use fluid_execution::CompletedAuthoredFluidStudy;
pub use geometry::MaterializedFeaGeometry;
pub use magnetic_execution::CompletedAuthoredMagneticStudy;
pub use maxwell_execution::CompletedAuthoredMaxwellStudy;
pub use preparation::{
    checked_fea_artifact_access, PreparedAuthoredAcousticStudy, PreparedAuthoredElectricalStudy,
    PreparedAuthoredFluidStudy, PreparedAuthoredMagneticStudy, PreparedAuthoredMaxwellStudy,
    PreparedAuthoredThermalStudy, PreparedFeaStudy, PreparedFeaStudyRoot,
};
pub use thermal_execution::CompletedAuthoredThermalStudy;

pub async fn materialize_checked_fea_study_geometry(
    source: &runmat_core::CheckedFeaSessionSource,
    study_id: &str,
    inputs: &[AuthoredStudyInputBinding],
    authority: &str,
) -> Result<MaterializedFeaGeometry, String> {
    let selector = FeaRootSelector::Study {
        id: study_id.to_string(),
    };
    let resolved = source
        .resolve_root(Some(&selector))
        .map_err(|error| error.to_string())?;
    let ResolvedFeaRoot::Study(study) = resolved else {
        return Err(format!("`{study_id}` did not resolve to an FEA study"));
    };
    geometry::materialize_study_geometry(source, &study, inputs, authority).await
}

pub async fn prepare_checked_fea_study(
    source: &runmat_core::CheckedFeaSessionSource,
    resolved: runmat_analysis_language::ResolvedAnalysisStudy,
    inputs: &[AuthoredStudyInputBinding],
    authority: &str,
    options: &LocalFeaExecutionOptions,
) -> Result<PreparedFeaStudy, String> {
    preparation::prepare_study(source, resolved, inputs, authority, options).await
}

use futures::StreamExt as _;
use runmat_analysis_core::AuthoredStudyInputBinding;
use runmat_runtime::analysis::{
    AnalysisStudyRunData, FeaRootSelector, ResolvedFeaRoot, StructuralStudyResult,
};
use std::sync::Arc;

pub type LocalFeaProgressObserver =
    Arc<dyn Fn(runmat_analysis_fea::FeaProgressEvent) + Send + Sync + 'static>;
pub type LocalFeaMeshObserver =
    Arc<dyn Fn(&runmat_meshing_core::SolverMeshArtifact) + Send + Sync + 'static>;

#[derive(Clone)]
pub struct LocalFeaExecutionOptions {
    pub root: Option<FeaRootSelector>,
    pub workers: Option<u32>,
    pub backend: runmat_analysis_fea::ComputeBackend,
    pub accelerators: Vec<runmat_execution::resource::AcceleratorRequest>,
    /// Receives coarse, ordered pipeline progress directly from local execution.
    /// This avoids relying on thread-local propagation across async/native boundaries.
    pub progress_observer: Option<LocalFeaProgressObserver>,
    /// Receives the canonical solver mesh as soon as meshing completes, before assembly/solve.
    pub mesh_observer: Option<LocalFeaMeshObserver>,
}

impl Default for LocalFeaExecutionOptions {
    fn default() -> Self {
        Self {
            root: None,
            workers: None,
            backend: runmat_analysis_fea::ComputeBackend::Cpu,
            accelerators: Vec::new(),
            progress_observer: None,
            mesh_observer: None,
        }
    }
}

impl std::fmt::Debug for LocalFeaExecutionOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalFeaExecutionOptions")
            .field("root", &self.root)
            .field("workers", &self.workers)
            .field("backend", &self.backend)
            .field("accelerators", &self.accelerators)
            .field("progress_observer", &self.progress_observer.is_some())
            .field("mesh_observer", &self.mesh_observer.is_some())
            .finish()
    }
}

#[derive(Debug)]
pub struct CompletedLocalFeaCase {
    pub case_id: String,
    pub result: CompletedFeaStudyResult,
    pub publication: AnalysisStudyRunData,
}

#[derive(Debug)]
pub enum CompletedFeaStudyResult {
    Fluid(Box<CompletedAuthoredFluidStudy>),
    Acoustic(Box<CompletedAuthoredAcousticStudy>),
    Electrical(Box<CompletedAuthoredElectricalStudy>),
    Magnetic(Box<CompletedAuthoredMagneticStudy>),
    Maxwell(Box<CompletedAuthoredMaxwellStudy>),
    Structural(Box<StructuralStudyResult>),
    Thermal(Box<CompletedAuthoredThermalStudy>),
}

impl CompletedFeaStudyResult {
    pub fn mesh(&self) -> &runmat_meshing_core::SolverMeshArtifact {
        match self {
            Self::Fluid(result) => &result.mesh,
            Self::Acoustic(result) => &result.mesh,
            Self::Electrical(result) => &result.mesh,
            Self::Magnetic(result) => &result.mesh,
            Self::Maxwell(result) => &result.mesh,
            Self::Structural(result) => result.mesh(),
            Self::Thermal(result) => &result.mesh,
        }
    }
}

#[derive(Debug)]
pub enum LocalFeaExecution {
    Study(Box<CompletedLocalFeaCase>),
    Sweep {
        sweep_id: String,
        cases: Vec<CompletedLocalFeaCase>,
    },
}

pub async fn execute_checked_fea_locally(
    source: &runmat_core::CheckedFeaSessionSource,
    options: LocalFeaExecutionOptions,
) -> Result<LocalFeaExecution, String> {
    let root = source
        .resolve_root(options.root.as_ref())
        .map_err(|error| error.to_string())?;
    match root {
        ResolvedFeaRoot::Study(study) => execute_case(source, study, Vec::new(), &options)
            .await
            .map(Box::new)
            .map(LocalFeaExecution::Study),
        ResolvedFeaRoot::Sweep(sweep) => {
            if sweep.study.study.sequence.steps.iter().any(|step| {
                step.family == runmat_analysis_core::PhysicsFamily::Electromagnetic
                    && step.procedure == runmat_analysis_core::AnalysisProcedure::HarmonicResponse
            }) {
                return Err("authored Maxwell sweeps are outside the qualified singleton-frequency execution envelope".into());
            }
            let expanded = sweep
                .sweep
                .expand_cases(&sweep.study.study)
                .map_err(|error| error.to_string())?;
            if expanded.is_empty() {
                return Err("typed FEA sweep contains no cases".into());
            }
            let concurrency = sweep_concurrency(&sweep, &options, expanded.len());
            let mut cases =
                futures::stream::iter(expanded.into_iter().enumerate().map(|(index, case)| {
                    let study = sweep.study.clone();
                    let options = &options;
                    async move {
                        let inputs = case
                            .bindings
                            .into_iter()
                            .map(|binding| AuthoredStudyInputBinding {
                                input_id: binding.input_id,
                                value: binding.value,
                            })
                            .collect();
                        execute_case(source, study, inputs, options)
                            .await
                            .map(|mut completed| {
                                completed.case_id = case.case_id;
                                (index, completed)
                            })
                    }
                }))
                .buffer_unordered(concurrency)
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            cases.sort_by_key(|(index, _)| *index);
            Ok(LocalFeaExecution::Sweep {
                sweep_id: sweep.sweep.sweep_id,
                cases: cases.into_iter().map(|(_, case)| case).collect(),
            })
        }
    }
}

fn sweep_concurrency(
    sweep: &runmat_analysis_language::ResolvedAnalysisSweep,
    options: &LocalFeaExecutionOptions,
    case_count: usize,
) -> usize {
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    let requested = sweep
        .study
        .execution_policy
        .as_ref()
        .map(|policy| policy.fanout.sweep_cases)
        .and_then(|limit| match limit {
            runmat_analysis_core::ParallelismLimit::Auto => None,
            runmat_analysis_core::ParallelismLimit::Fixed(value) => Some(value as usize),
        })
        .unwrap_or(available);
    let workers_per_case = options
        .workers
        .or_else(|| {
            sweep
                .study
                .execution_policy
                .as_ref()
                .and_then(|policy| policy.resources.workers)
                .map(|range| range.maximum)
        })
        .unwrap_or(1)
        .max(1) as usize;
    requested
        .min(available / workers_per_case)
        .max(1)
        .min(case_count)
}

async fn execute_case(
    source: &runmat_core::CheckedFeaSessionSource,
    study: runmat_analysis_language::ResolvedAnalysisStudy,
    inputs: Vec<AuthoredStudyInputBinding>,
    options: &LocalFeaExecutionOptions,
) -> Result<CompletedLocalFeaCase, String> {
    let authority = format!("typed-fea-local:{}", uuid::Uuid::new_v4());
    let prepared = prepare_checked_fea_study(source, study, &inputs, &authority, options).await?;
    execute_prepared_fea_study_locally(prepared, options).await
}

/// Execute an authentic prepared study through governed family coordinators.
pub async fn execute_prepared_fea_study_locally(
    prepared: PreparedFeaStudy,
    options: &LocalFeaExecutionOptions,
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
            let publication = runmat_runtime::analysis::publish_electrical_study_result(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &publication_options,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?
            .data;
            (
                CompletedFeaStudyResult::Electrical(Box::new(result)),
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
            let publication = runmat_runtime::analysis::publish_magnetic_study_result(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &publication_options,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?
            .data;
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
            let publication = runmat_runtime::analysis::publish_maxwell_study_result(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &publication_options,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?
            .data;
            (
                CompletedFeaStudyResult::Maxwell(Box::new(result)),
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
            let publication = runmat_runtime::analysis::publish_fluid_study_result(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &publication_options,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?
            .data;
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
            let publication = runmat_runtime::analysis::publish_acoustic_study_result(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &publication_options,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?
            .data;
            (
                CompletedFeaStudyResult::Acoustic(Box::new(result)),
                publication,
            )
        }
        PreparedFeaStudyRoot::Structural(root) => {
            let result = super::execute_prepared_structural_study_locally_with_observers(
                super::NativeExecutionConfig::for_current_executable()
                    .map_err(|error| error.to_string())?,
                root,
                workers,
                super::LocalStructuralStudyObservers {
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
            let publication = runmat_runtime::analysis::publish_structural_study_result(
                &result,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|error| error.message)?
            .data;
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
            let publication = runmat_runtime::analysis::publish_thermal_study_result(
                &result.completion,
                &result.source,
                &result.mesh,
                &result.meshing_evidence,
                &publication_options,
                digest,
                runmat_runtime::operations::OperationContext::new(None, None),
                identity,
            )
            .map_err(|e| e.message)?
            .data;
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
