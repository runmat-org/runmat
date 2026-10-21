//! Native execution of checked `.fea` sources.
//!
//! This is the shared host boundary used by the CLI and Desktop. It consumes the exact checked
//! session product, resolves a typed root, materializes geometry, meshes, executes, and publishes
//! without interpreting authoring syntax or introducing host-specific physics defaults.

mod assessment;
mod callable;
mod execution;
mod family_execution;
mod geometry;
mod input;
mod preparation;
mod prepared_validation;
pub use execution::{execute_prepared_fea_study_locally, prepare_study_publication};
use execution::execute_prepared_with_source;
pub(crate) use family_execution::local_result as local_family_result;
pub use family_execution::{
    imported_family_result, CompletedAuthoredAcousticResponseStudy, CompletedAuthoredAcousticStudy,
    CompletedAuthoredElectricalStudy,
    CompletedAuthoredFluidStudy, CompletedAuthoredMagneticStudy, CompletedAuthoredMaxwellStudy,
    CompletedAuthoredNonlinearSolidStudy, CompletedAuthoredThermalStudy,
};

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

#[cfg(test)]
#[path = "tests/assessment_functions.rs"]
mod assessment_functions;

pub use assessment::{
    complete as complete_local_fea_assessments,
    complete_frozen_locally as complete_frozen_fea_assessments,
    complete_remotely as complete_remote_fea_assessments,
    freeze_functions as freeze_fea_assessment_functions,
    reassess_saved_locally as reassess_saved_fea_locally,
};
pub use geometry::MaterializedFeaGeometry;
pub use preparation::{checked_fea_artifact_access, PreparedFeaStudy};

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
    Nonlinear(Box<CompletedAuthoredNonlinearSolidStudy>),
    Acoustic(Box<CompletedAuthoredAcousticStudy>),
    AcousticResponse(Box<CompletedAuthoredAcousticResponseStudy>),
    Electrical(Box<CompletedAuthoredElectricalStudy>),
    Magnetic(Box<CompletedAuthoredMagneticStudy>),
    Maxwell(Box<CompletedAuthoredMaxwellStudy>),
    Structural(Box<StructuralStudyResult>),
    Thermal(Box<CompletedAuthoredThermalStudy>),
}

impl CompletedFeaStudyResult {
    /// Compiled study of a non-structural family completion.
    pub fn family_compiled_study(
        &self,
    ) -> Option<&runmat_analysis_fea::study::CompiledAnalysisStudy> {
        match self {
            Self::Fluid(result) => Some(&result.completion.compiled),
            Self::Nonlinear(result) => Some(&result.completion.compiled),
            Self::Acoustic(result) => Some(&result.completion.compiled),
            Self::AcousticResponse(result) => Some(&result.completion.compiled),
            Self::Electrical(result) => Some(&result.completion.compiled),
            Self::Magnetic(result) => Some(&result.completion.compiled),
            Self::Maxwell(result) => Some(&result.completion.compiled),
            Self::Thermal(result) => Some(&result.completion.compiled),
            Self::Structural(_) => None,
        }
    }

    /// Digest of the solved numerical content, independent of placement and attempt evidence.
    ///
    /// Local and remote runs of one study under one artifact authority produce the same value.
    pub fn numerical_digest(&self) -> Result<runmat_execution::Digest, String> {
        let bytes = match self {
            Self::Thermal(result) => serde_json::to_vec(&(
                &result.completion.manifest,
                &result.completion.field_partitions,
                &result.completion.states,
            )),
            Self::Electrical(result) => serde_json::to_vec(&result.completion.result),
            Self::Magnetic(result) => serde_json::to_vec(&result.completion.result),
            Self::Maxwell(result) => serde_json::to_vec(&result.completion.result),
            Self::Acoustic(result) => serde_json::to_vec(&result.completion.result),
            Self::AcousticResponse(result) => serde_json::to_vec(&result.completion.result),
            Self::Fluid(result) => serde_json::to_vec(&result.completion.result),
            Self::Nonlinear(result) => serde_json::to_vec(&(
                &result.completion.terminal_block,
                &result.completion.states,
            )),
            Self::Structural(result) => serde_json::to_vec(result.numerical()),
        }
        .map_err(|error| error.to_string())?;
        Ok(runmat_execution::Digest::sha256(bytes))
    }

    pub fn mesh(&self) -> &runmat_meshing_core::SolverMeshArtifact {
        match self {
            Self::Fluid(result) => &result.mesh,
            Self::Nonlinear(result) => &result.mesh,
            Self::Acoustic(result) => &result.mesh,
            Self::AcousticResponse(result) => &result.mesh,
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
        /// Every authored case in authored order, each with its own terminal status.
        cases: Vec<LocalSweepCase>,
    },
}

/// One sweep case: its authored identity and its own success or failure. A failed case never
/// disappears from the sweep and never hides the results of its siblings.
#[derive(Debug)]
pub struct LocalSweepCase {
    pub case_id: String,
    pub outcome: Result<CompletedLocalFeaCase, String>,
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
            // Independent cases overlap up to the admitted concurrency; completion order is
            // irrelevant because results are restored to authored order below.
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
                        let outcome = execute_case(source, study, inputs, options)
                            .await
                            .map(|mut completed| {
                                completed.case_id = case.case_id.clone();
                                completed
                            });
                        (
                            index,
                            LocalSweepCase {
                                case_id: case.case_id,
                                outcome,
                            },
                        )
                    }
                }))
                .buffer_unordered(concurrency)
                .collect::<Vec<_>>()
                .await;
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
    let policy = sweep.study.execution_policy.as_ref();
    let requested = policy
        .map(|policy| policy.fanout.sweep_cases)
        .and_then(|limit| match limit {
            runmat_analysis_core::ParallelismLimit::Auto => None,
            runmat_analysis_core::ParallelismLimit::Fixed(value) => Some(value as usize),
        });
    let workers_per_case = options
        .workers
        .or_else(|| {
            policy
                .and_then(|policy| policy.resources.workers)
                .map(|range| range.maximum)
        })
        .unwrap_or(1)
        .max(1);
    // Per-case threads come from the same authority as the case's own plan.
    let threads_per_worker = crate::machine_capacity::study_allocation(
        workers_per_case,
        policy.and_then(|policy| policy.resources.cpu_threads_per_worker),
    )
    .unwrap_or(1);
    crate::machine_capacity::case_concurrency(workers_per_case, threads_per_worker, case_count, requested)
}

async fn execute_case(
    source: &runmat_core::CheckedFeaSessionSource,
    study: runmat_analysis_language::ResolvedAnalysisStudy,
    inputs: Vec<AuthoredStudyInputBinding>,
    options: &LocalFeaExecutionOptions,
) -> Result<CompletedLocalFeaCase, String> {
    let authority = format!("typed-fea-local:{}", uuid::Uuid::new_v4());
    let prepared = prepare_checked_fea_study(source, study, &inputs, &authority, options).await?;
    execute_prepared_with_source(prepared, options, Some(source)).await
}
