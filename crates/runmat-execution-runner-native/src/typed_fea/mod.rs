//! Native execution of checked `.fea` sources.
//!
//! This is the shared host boundary used by the CLI and Desktop. It consumes the exact checked
//! session product, resolves a typed root, materializes geometry, meshes, executes, and publishes
//! without interpreting authoring syntax or introducing host-specific physics defaults.

mod assessment;
mod run_admission;
mod callable;
mod chain;
mod execution;
mod family_execution;
mod geometry;
mod input;
mod line_parameters;
pub use line_parameters::CompletedLineParametersStudy;
mod preparation;
mod prepared_validation;
mod duration;
pub use duration::execute_checked_fea_locally_with_timing;
mod planning;
use execution::execute_prepared_with_source;
pub use execution::{execute_prepared_fea_study_locally, prepare_study_publication};
pub(crate) use family_execution::local_result as local_family_result;
pub use family_execution::{
    imported_family_result, CompletedAuthoredAcousticResponseStudy, CompletedAuthoredAcousticStudy,
    CompletedAuthoredElectricalStudy, CompletedAuthoredFluidStudy, CompletedAuthoredMagneticStudy,
    CompletedAuthoredMaxwellStudy, CompletedAuthoredNonlinearSolidStudy,
    CompletedAuthoredThermalStudy, FamilyResultImportLimits,
};
pub use planning::{plan_checked_fea_locally, plan_checked_fea_locally_with_timing};

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
    /// Explicit host allocation; independent of worker-slot count.
    pub cpu_threads_per_worker: Option<runmat_analysis_fea::planning::resource::LocalThreadCount>,
    pub max_concurrent_cases: Option<std::num::NonZeroU16>,
    pub backend: runmat_analysis_fea::ComputeBackend,
    pub accelerators: Vec<runmat_execution::resource::AcceleratorRequest>,
    /// The placement the host admitted for the study: the authored admission policy and the
    /// fallback warning of a study planned on `cpu_reference` although the accelerator was
    /// requested. Applied when the study publication is prepared.
    pub accelerator_placement: runmat_analysis_fea::accelerator_admission::AcceleratorPlacement,
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
            cpu_threads_per_worker: None,
            max_concurrent_cases: None,
            backend: runmat_analysis_fea::ComputeBackend::Cpu,
            accelerators: Vec::new(),
            accelerator_placement: Default::default(),
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
            .field("cpu_threads_per_worker", &self.cpu_threads_per_worker)
            .field("max_concurrent_cases", &self.max_concurrent_cases)
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
    LineParameters(Box<CompletedLineParametersStudy>),
}

impl CompletedFeaStudyResult {
    /// Compiled study of a non-structural family completion.
    pub fn family_compiled_study(
        &self,
    ) -> Option<&runmat_analysis_fea::study::CompiledAnalysisStudy> {
        match self {
            Self::Fluid(result) => Some(result.completion.compiled()),
            Self::Nonlinear(result) => Some(&result.completion.compiled),
            Self::Acoustic(result) => Some(&result.completion.compiled),
            Self::AcousticResponse(result) => Some(&result.completion.compiled),
            Self::Electrical(result) => Some(&result.completion.compiled),
            Self::Magnetic(result) => Some(&result.completion.compiled),
            Self::Maxwell(result) => Some(&result.completion.compiled),
            Self::Thermal(result) => Some(&result.completion.compiled),
            Self::LineParameters(result) => Some(&result.completion.compiled),
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
            Self::Nonlinear(result) => {
                serde_json::to_vec(&(&result.completion.terminal_block, &result.completion.states))
            }
            Self::Structural(result) => serde_json::to_vec(result.numerical()),
            Self::LineParameters(result) => serde_json::to_vec(&(
                &result.completion.parameters,
                &result.completion.observation_results,
            )),
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
            Self::LineParameters(result) => &result.mesh,
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

impl LocalFeaExecution {
    /// A completed sweep can carry failed cases even when orchestration itself
    /// returned Ok. Hosts must not classify that envelope as a successful run.
    pub fn is_successful(&self) -> bool {
        match self {
            Self::Study(_) => true,
            Self::Sweep { cases, .. } => cases.iter().all(|case| case.outcome.is_ok()),
        }
    }
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
    let (scope, options) = run_admission::admit_root(source, &root, options)?;
    let result = match scope
        .run_checked_result_future(execute_admitted_root(source, root, options))
        .await
    {
        Ok(result) => result,
        Err(failure) => {
            let _ = scope.settle(failure.outcome());
            return Err(failure.to_string());
        }
    };
    let outcome = run_admission::outcome(&result);
    let settlement = scope.settle(outcome);
    match result {
        Err(original) => Err(original),
        Ok(value) => settlement
            .map(|()| value)
            .map_err(|error| format!("{}: {error}", error.error_code())),
    }
}

async fn execute_admitted_root(
    source: &runmat_core::CheckedFeaSessionSource,
    root: ResolvedFeaRoot,
    options: LocalFeaExecutionOptions,
) -> Result<LocalFeaExecution, String> {
    match root {
        ResolvedFeaRoot::Study(study) => {
            let frequencies = harmonic_frequency_cases(&study);
            let chained = chain::heating_chain(&study).is_some()
                || chain::electrothermal_analysis(&study).is_some();
            if frequencies.len() <= 1 && !chained {
                return execute_case(source, study, Vec::new(), &options)
                    .await
                    .map(Box::new)
                    .map(LocalFeaExecution::Study);
            }
            // Every authored frequency of a harmonic electromagnetic analysis is an
            // independent solve (and drives its own downstream heating analysis); run them
            // as cases in parallel, each keyed by its frequency.
            let sweep_id = format!("{}.cases", study.study.study_id);
            let study_id = study.study.study_id.clone();
            let frequencies = if frequencies.is_empty() {
                vec![f64::NAN]
            } else {
                frequencies
            };
            let allocation = run_admission::study_allocation(&study, &options)?;
            let concurrency = crate::machine_capacity::case_concurrency(
                allocation.workers,
                allocation.threads,
                frequencies.len(),
                options.max_concurrent_cases.map(|v| usize::from(v.get())),
            );
            let run_group = |(index, frequency): (usize, f64)| {
                let case = if frequency.is_nan() {
                    study.clone()
                } else {
                    single_frequency_study(&study, frequency)
                };
                let prefix = if frequency.is_nan() {
                    String::new()
                } else {
                    frequency_case_id(frequency)
                };
                let options = &options;
                let study_id = study_id.clone();
                async move {
                    // A failed frequency keeps its place in the sweep and never hides
                    // the results of its siblings.
                    let cases = match execute_study_cases(source, case, &prefix, options).await {
                        Ok(cases) => cases
                            .into_iter()
                            .map(|case| LocalSweepCase {
                                case_id: case.case_id.clone(),
                                outcome: Ok(case),
                            })
                            .collect(),
                        Err(error) => vec![LocalSweepCase {
                            case_id: if prefix.is_empty() {
                                study_id.clone()
                            } else {
                                prefix.clone()
                            },
                            outcome: Err(error),
                        }],
                    };
                    (index, cases)
                }
            };
            // Every frequency shares one geometry and mesh. The first case meshes and fills the
            // accepted-stage meshing cache; the remaining frequencies then reuse that mesh
            // instead of each meshing the same geometry at the same time.
            let mut indexed = frequencies.into_iter().enumerate();
            let mut groups = Vec::new();
            if let Some(first) = indexed.next() {
                groups.push(run_group(first).await);
            }
            groups.extend(
                futures::stream::iter(indexed.map(run_group))
                    .buffer_unordered(concurrency)
                    .collect::<Vec<_>>()
                    .await,
            );
            groups.sort_by_key(|(index, _)| *index);
            Ok(LocalFeaExecution::Sweep {
                sweep_id,
                cases: groups.into_iter().flat_map(|(_, cases)| cases).collect(),
            })
        }
        ResolvedFeaRoot::Sweep(sweep) => {
            if harmonic_frequency_cases(&sweep.study).len() > 1 {
                return Err("a parameter sweep over a multi-frequency harmonic analysis is not supported; sweep one frequency per study or list the frequencies without a sweep".into());
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
                        let outcome = execute_case(source, study, inputs, options).await.map(
                            |mut completed| {
                                completed.case_id = case.case_id.clone();
                                completed
                            },
                        );
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

/// Authored frequencies of the study's harmonic electromagnetic analysis (empty if none).
fn harmonic_frequency_cases(study: &runmat_analysis_language::ResolvedAnalysisStudy) -> Vec<f64> {
    study
        .study
        .sequence
        .steps
        .iter()
        .find(|step| {
            step.family == runmat_analysis_core::PhysicsFamily::Electromagnetic
                && step.procedure == runmat_analysis_core::AnalysisProcedure::HarmonicResponse
        })
        .and_then(|step| match &step.independent_domain {
            runmat_analysis_core::IndependentDomain::FrequencyHertz(values) => Some(values.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// The authored study restricted to one of its harmonic frequencies.
fn single_frequency_study(
    study: &runmat_analysis_language::ResolvedAnalysisStudy,
    frequency: f64,
) -> runmat_analysis_language::ResolvedAnalysisStudy {
    let mut case = study.clone();
    for step in &mut case.study.sequence.steps {
        if step.family == runmat_analysis_core::PhysicsFamily::Electromagnetic
            && step.procedure == runmat_analysis_core::AnalysisProcedure::HarmonicResponse
        {
            step.independent_domain =
                runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![frequency]);
        }
    }
    case
}

fn frequency_case_id(frequency: f64) -> String {
    format!("frequency={frequency}Hz")
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
        options
            .cpu_threads_per_worker
            .map(|v| u32::from(v.get()))
            .or_else(|| policy.and_then(|policy| policy.resources.cpu_threads_per_worker)),
    )
    .unwrap_or(1);
    crate::machine_capacity::case_concurrency(
        workers_per_case,
        threads_per_worker,
        case_count,
        options
            .max_concurrent_cases
            .map(|v| usize::from(v.get()))
            .or(requested),
    )
}

/// One study case, or the two analyses of a harmonic-heating chain run in order.
async fn execute_study_cases(
    source: &runmat_core::CheckedFeaSessionSource,
    study: runmat_analysis_language::ResolvedAnalysisStudy,
    prefix: &str,
    options: &LocalFeaExecutionOptions,
) -> Result<Vec<CompletedLocalFeaCase>, String> {
    let label = |analysis: &str| {
        if prefix.is_empty() {
            analysis.to_string()
        } else {
            format!("{prefix}/{analysis}")
        }
    };
    if let Some(analysis) = chain::electrothermal_analysis(&study) {
        // Heat transfer iterated with conduction to the two-way fixed point, then the
        // conduction field at the temperatures the heat case published (final time of a
        // transient). The conduction case carries those temperatures in its planning policy, so
        // it runs on any placement without solving the coupled loop again.
        let mut heat = execute_case(source, study.clone(), Vec::new(), options).await?;
        heat.case_id = label(&analysis);
        let CompletedFeaStudyResult::Thermal(thermal) = &heat.result else {
            return Err(
                "the heat case of an electrothermal analysis must be a thermal study".into(),
            );
        };
        let state = thermal
            .completion
            .states
            .iter()
            .filter_map(|state| state.payload.thermal())
            .max_by(|a, b| {
                a.sample_time_s
                    .canonical_value()
                    .total_cmp(&b.sample_time_s.canonical_value())
            })
            .ok_or("the electrothermal heat case published no thermal state")?;
        let transient = !matches!(
            thermal.completion.workload.procedure,
            runmat_analysis_fea::ThermalProcedureControl::Steady
        );
        let (element_temperature_k, operating_time_s) =
            runmat_analysis_fea::execution::conduction_operating_temperatures(
                state,
                &thermal.mesh,
                &thermal.completion.compiled.model.interfaces,
                transient,
            )
            .map_err(|error| error.to_string())?;
        let authority = format!("typed-fea-local:{}", uuid::Uuid::new_v4());
        let prepared = preparation::prepare_study_with_role(
            source,
            study,
            &[],
            &authority,
            options,
            preparation::PreparationRole::ElectrothermalConduction {
                element_temperature_k,
                operating_time_s,
            },
        )
        .await?;
        let mut conduction = execute_prepared_with_source(prepared, options, Some(source)).await?;
        conduction.case_id = label(&format!("{analysis}/conduction"));
        return Ok(vec![heat, conduction]);
    }
    let Some(chain) = chain::heating_chain(&study) else {
        let mut case = execute_case(source, study, Vec::new(), options).await?;
        case.case_id = prefix.to_string();
        return Ok(vec![case]);
    };
    let mut harmonic = execute_case(
        source,
        chain::sub_study(&study, &chain.harmonic),
        Vec::new(),
        options,
    )
    .await?;
    harmonic.case_id = label(&chain.harmonic);
    let CompletedFeaStudyResult::Maxwell(maxwell) = &harmonic.result else {
        return Err("the heat source of a thermal analysis must come from a harmonic electromagnetic analysis".into());
    };
    let density = maxwell
        .completion
        .result
        .element_loss_density_w_per_m3(&maxwell.mesh)
        .map_err(|error| error.to_string())?;
    let authority = format!("typed-fea-local:{}", uuid::Uuid::new_v4());
    let prepared = preparation::prepare_study_with_role(
        source,
        chain::sub_study(&study, &chain.thermal),
        &[],
        &authority,
        options,
        preparation::PreparationRole::Heated(runmat_analysis_fea::study::ElementHeatSource {
            step_id: chain.thermal.clone(),
            source_id: format!("{}.{}", chain.harmonic, chain.state_id),
            element_density_w_per_m3: density,
        }),
    )
    .await?;
    let mut thermal = execute_prepared_with_source(prepared, options, Some(source)).await?;
    thermal.case_id = label(&chain.thermal);
    Ok(vec![harmonic, thermal])
}

async fn execute_case(
    source: &runmat_core::CheckedFeaSessionSource,
    study: runmat_analysis_language::ResolvedAnalysisStudy,
    inputs: Vec<AuthoredStudyInputBinding>,
    options: &LocalFeaExecutionOptions,
) -> Result<CompletedLocalFeaCase, String> {
    // A cross-section is meshed and solved in place: no exact geometry or prepared root.
    if line_parameters::is_cross_section(&study) {
        return line_parameters::execute_case(source, study, inputs, options).await;
    }
    let authority = format!("typed-fea-local:{}", uuid::Uuid::new_v4());
    let prepared = prepare_checked_fea_study(source, study, &inputs, &authority, options).await?;
    execute_prepared_with_source(prepared, options, Some(source)).await
}
