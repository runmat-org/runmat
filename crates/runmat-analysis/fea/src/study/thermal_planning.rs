//! Authored thermal intent, independent of task placement and transport.
use runmat_analysis_core::{
    AccuracyTolerance, AnalysisProcedure, ErrorContributionKind, IndependentDomain,
    ObservationRequest, PhysicsFamily, ProcedureSettings, RestartPolicy, SamplingPolicy,
    SolutionSpace, StateKind,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::CompiledAnalysisStudy;
use crate::{execution::ThermalWorkload, ComputeBackend, ThermalProcedureControl};

mod initial_selection;
mod sources;
mod validation;
pub use initial_selection::thermal_step_initial_selection_supported;
#[cfg(test)]
mod tests;

/// Backend admission is resolved by the caller; physical and accuracy controls remain authored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalStudyPlanningPolicy {
    pub backend: ComputeBackend,
    /// A solved upstream loss field that heats this analysis (`heat_source = analysis.<p>.state.<s>`).
    pub element_heat_source: Option<ElementHeatSource>,
}

/// Element heat generation solved by an upstream analysis on the same canonical mesh.
///
/// The density of each element, in canonical element order, is bound as the heat source
/// `source_id` of the thermal step `step_id`, so the total heat equals the upstream loss.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElementHeatSource {
    pub step_id: String,
    pub source_id: String,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub element_density_w_per_m3: runmat_types::physical::Quantity<
        runmat_types::physical::kind::HeatGenerationDensity,
        Vec<f64>,
    >,
}

impl ElementHeatSource {
    /// Bind the solved density into the compiled thermal model as the step's heat source.
    pub fn bind(
        &self,
        model: &mut runmat_analysis_core::AnalysisModel,
        mesh: &runmat_meshing_core::SolverMeshArtifact,
    ) -> crate::execution::AnalysisExecutionResult<()> {
        let transient = model.sequence.steps.iter().any(|step| {
            step.step_id.as_str() == self.step_id
                && step.procedure == AnalysisProcedure::TransientResponse
        });
        crate::execution::bind_element_heat_source(
            model,
            &self.step_id,
            &self.source_id,
            runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()),
            &[(
                runmat_types::physical::Time::new(0.0),
                self.element_density_w_per_m3.clone(),
            )],
            transient,
        )
    }
}

/// Plan one fresh steady or backward-Euler thermal step without rewriting its authored model.
/// Mesh-dependent support, source projection and observation resolution remain admission work.
pub fn plan_thermal_study(
    study: &CompiledAnalysisStudy,
    policy: ThermalStudyPlanningPolicy,
) -> Result<ThermalWorkload, ThermalStudyPlanningError> {
    plan_thermal_study_with_fields(study, policy, Vec::new())
}

/// Plan with closed original fields resolved by the host before numerical admission.
pub fn plan_thermal_study_with_fields(
    study: &CompiledAnalysisStudy,
    policy: ThermalStudyPlanningPolicy,
    initial_fields: Vec<crate::execution::InitialFieldBinding>,
) -> Result<ThermalWorkload, ThermalStudyPlanningError> {
    let step = validation::standalone_step(study)?;
    plan_selected(study, &step.step_id.0, policy, false, initial_fields)
}

/// Plan an independent thermal producer in its full authored mixed-physics sequence.
/// This selects one task; it does not schedule or execute the surrounding study.
pub fn plan_thermal_step(
    study: &CompiledAnalysisStudy,
    step_id: &str,
    policy: ThermalStudyPlanningPolicy,
) -> Result<ThermalWorkload, ThermalStudyPlanningError> {
    plan_thermal_step_with_fields(study, step_id, policy, Vec::new())
}

/// Selected mixed-sequence producer with the same original-field admission.
pub fn plan_thermal_step_with_fields(
    study: &CompiledAnalysisStudy,
    step_id: &str,
    policy: ThermalStudyPlanningPolicy,
    initial_fields: Vec<crate::execution::InitialFieldBinding>,
) -> Result<ThermalWorkload, ThermalStudyPlanningError> {
    plan_selected(study, step_id, policy, true, initial_fields)
}

fn plan_selected(
    study: &CompiledAnalysisStudy,
    step_id: &str,
    policy: ThermalStudyPlanningPolicy,
    mixed_context: bool,
    initial_fields: Vec<crate::execution::InitialFieldBinding>,
) -> Result<ThermalWorkload, ThermalStudyPlanningError> {
    let (step, procedure, residual_target) = authored_controls(&study.model, step_id)?;
    let workload = ThermalWorkload {
        initial_fields,
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        backend: policy.backend,
        procedure,
        residual_target,
    };
    validate_sources(&workload, mixed_context)?;
    Ok(workload)
}

/// Shared selected-step authority for canonical heterogeneous inventories.
impl ThermalWorkload {
    pub(crate) fn validate_selected_study_intent(&self) -> Result<(), ThermalStudyPlanningError> {
        let workload = self;
        let (_, procedure, residual_target) =
            authored_controls(&workload.model, &workload.step_id.0)?;
        if workload.procedure != procedure || workload.residual_target != residual_target {
            return Err(ThermalStudyPlanningError::Source(
                "thermal numerical controls differ from authored intent".into(),
            ));
        }
        validate_sources(workload, true)
    }
}

fn authored_controls<'a>(
    model: &'a runmat_analysis_core::AnalysisModel,
    step_id: &str,
) -> Result<
    (
        &'a runmat_analysis_core::ProcedureStep,
        ThermalProcedureControl,
        f64,
    ),
    ThermalStudyPlanningError,
> {
    let sequence = &model.sequence;
    let mut matches = sequence
        .steps
        .iter()
        .filter(|step| step.step_id.0 == step_id);
    let step = matches
        .next()
        .ok_or_else(|| ThermalStudyPlanningError::InvalidSelection(step_id.to_owned()))?;
    if matches.next().is_some() {
        return Err(ThermalStudyPlanningError::InvalidSelection(
            step_id.to_owned(),
        ));
    }
    if sequence
        .dependencies
        .iter()
        .any(|dependency| dependency.successor == step.step_id)
        || sequence
            .transitions
            .iter()
            .any(|transition| transition.consumer_step == step.step_id)
        || sequence
            .branches
            .iter()
            .any(|branch| branch.consumer_steps.contains(&step.step_id))
        || !initial_selection::supported(model, step)
        || step.restart != RestartPolicy::Fresh
        || step
            .state_outputs
            .iter()
            .any(|output| output.kind != StateKind::Thermal)
    {
        return Err(ThermalStudyPlanningError::UnsupportedSequence);
    }
    if step.family != PhysicsFamily::Thermal
        || step.solution_space != SolutionSpace::Direct
        || step.settings != ProcedureSettings::None
    {
        return Err(ThermalStudyPlanningError::UnsupportedProcedure);
    }
    let procedure = match (&step.procedure, &step.independent_domain) {
        (AnalysisProcedure::Static, IndependentDomain::Static) => ThermalProcedureControl::Steady,
        (
            AnalysisProcedure::TransientResponse,
            IndependentDomain::TimeSeconds { interval_count, .. },
        ) => ThermalProcedureControl::BackwardEuler {
            snapshot_count: interval_count
                .checked_add(1)
                .ok_or(ThermalStudyPlanningError::SampleCountOverflow)?,
        },
        _ => return Err(ThermalStudyPlanningError::UnsupportedProcedure),
    };
    let definitions = crate::physics::thermal::history_fields::definitions()
        .map_err(|error| ThermalStudyPlanningError::Observation(error.to_string()))?;
    for request in &step.observations {
        let spec = match request {
            ObservationRequest::History(spec) => spec,
            ObservationRequest::Field(spec) | ObservationRequest::Reduction(spec)
                if spec.sampling == SamplingPolicy::Final =>
            {
                spec
            }
            _ => {
                return Err(ThermalStudyPlanningError::Observation(format!(
                    "observation {:?} is not a supported thermal history, final field or final spatial reduction",
                    request.observation_id()
                )))
            }
        };
        if matches!(spec.sampling, SamplingPolicy::Event(_))
            || !definitions.iter().any(|definition| {
                definition.quantity() == spec.quantity
                    && definition.location() == spec.source_location
            })
        {
            return Err(ThermalStudyPlanningError::Observation(format!(
                "history {:?} has unsupported sampling or source field",
                spec.observation_id
            )));
        }
    }
    let residual_target = step
        .accuracy
        .criteria
        .iter()
        .filter(|criterion| criterion.contribution == ErrorContributionKind::Algebraic)
        .filter_map(|criterion| match criterion.tolerance {
            AccuracyTolerance::Relative(value) => Some(value),
            AccuracyTolerance::Absolute { .. } => None,
        })
        .reduce(f64::min)
        .ok_or(ThermalStudyPlanningError::MissingRelativeAlgebraicTarget)?;
    Ok((step, procedure, residual_target))
}

fn validate_sources(
    workload: &ThermalWorkload,
    mixed_context: bool,
) -> Result<(), ThermalStudyPlanningError> {
    workload.validate()?;
    let schedule = workload
        .solve_schedule()
        .map_err(|error| ThermalStudyPlanningError::Schedule(error.to_string()))?;
    sources::validate(workload, &schedule, mixed_context)?;
    Ok(())
}

#[derive(Debug, Error)]
pub enum ThermalStudyPlanningError {
    #[error("thermal workload differs from the compiled authored intent")]
    WorkloadMismatch,
    #[error("thermal planning requires a fresh independent step with only authored physical thermal initial selection; standalone planning additionally requires a single-step sequence")]
    UnsupportedSequence,
    #[error("thermal step selection must identify exactly one authored step: {0}")]
    InvalidSelection(String),
    #[error("thermal study requires a direct steady or backward-Euler time-domain procedure")]
    UnsupportedProcedure,
    #[error("authored thermal interval count overflows the sample contract")]
    SampleCountOverflow,
    #[error("thermal step requires an authored relative algebraic accuracy target")]
    MissingRelativeAlgebraicTarget,
    #[error("unsupported thermal observation: {0}")]
    Observation(String),
    #[error("thermal schedule is invalid: {0}")]
    Schedule(String),
    #[error("thermal source or boundary is unsupported: {0}")]
    Source(String),
    #[error(transparent)]
    Workload(#[from] crate::execution::AnalysisExecutionError),
}
