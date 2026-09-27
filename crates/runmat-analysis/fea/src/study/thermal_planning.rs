//! Authored thermal intent, independent of task placement and transport.
use runmat_analysis_core::{
    AccuracyTolerance, AnalysisProcedure, ErrorContributionKind, FormulationRequirement,
    IndependentDomain, ObservationRequest, PhysicsFamily, ProcedureSettings, RestartPolicy,
    SamplingPolicy, SolutionSpace, StateKind,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::CompiledAnalysisStudy;
use crate::{execution::ThermalWorkload, ComputeBackend, ThermalProcedureControl};

mod sources;
#[cfg(test)]
mod tests;

/// Backend admission is resolved by the caller; physical and accuracy controls remain authored.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalStudyPlanningPolicy {
    pub backend: ComputeBackend,
    pub formulation_requirement: FormulationRequirement,
}

/// Plan one fresh steady or backward-Euler thermal step without rewriting its authored model.
/// Mesh-dependent support, source projection and observation resolution remain admission work.
pub fn plan_thermal_study(
    study: &CompiledAnalysisStudy,
    policy: ThermalStudyPlanningPolicy,
) -> Result<ThermalWorkload, ThermalStudyPlanningError> {
    if policy.formulation_requirement != study.formulation_requirement {
        return Err(ThermalStudyPlanningError::FormulationRequirementMismatch);
    }
    let sequence = &study.model.sequence;
    let [step] = sequence.steps.as_slice() else {
        return Err(ThermalStudyPlanningError::UnsupportedSequence);
    };
    if !sequence.dependencies.is_empty()
        || !sequence.transitions.is_empty()
        || !sequence.branches.is_empty()
    {
        return Err(ThermalStudyPlanningError::UnsupportedSequence);
    }
    plan_selected(study, &step.step_id.0, policy, false)
}

/// Plan an independent thermal producer in its full authored mixed-physics sequence.
/// This selects one task; it does not schedule or execute the surrounding study.
pub fn plan_thermal_step(
    study: &CompiledAnalysisStudy,
    step_id: &str,
    policy: ThermalStudyPlanningPolicy,
) -> Result<ThermalWorkload, ThermalStudyPlanningError> {
    plan_selected(study, step_id, policy, true)
}

fn plan_selected(
    study: &CompiledAnalysisStudy,
    step_id: &str,
    policy: ThermalStudyPlanningPolicy,
    mixed_context: bool,
) -> Result<ThermalWorkload, ThermalStudyPlanningError> {
    if policy.formulation_requirement != study.formulation_requirement {
        return Err(ThermalStudyPlanningError::FormulationRequirementMismatch);
    }
    let sequence = &study.model.sequence;
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
        || !step.activations.is_empty()
        || !step.state_inputs.is_empty()
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
            ObservationRequest::Field(spec) if spec.sampling == SamplingPolicy::Final => spec,
            _ => {
                return Err(ThermalStudyPlanningError::Observation(format!(
                    "observation {:?} is not a supported thermal history or final field",
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
    let workload = ThermalWorkload {
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        backend: policy.backend,
        formulation_requirement: policy.formulation_requirement,
        procedure,
        residual_target,
    };
    workload.validate()?;
    let schedule = workload
        .solve_schedule()
        .map_err(|error| ThermalStudyPlanningError::Schedule(error.to_string()))?;
    sources::validate(&workload, &schedule, mixed_context)?;
    Ok(workload)
}

#[derive(Debug, Error)]
pub enum ThermalStudyPlanningError {
    #[error("thermal planning policy differs from the compiled formulation requirement")]
    FormulationRequirementMismatch,
    #[error("thermal planning requires a fresh independent step without inbound state or lifecycle controls; standalone planning additionally requires a single-step sequence")]
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
