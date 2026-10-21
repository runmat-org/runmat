//! Authored canonical acoustic intent, independent of task placement and transport.

use runmat_analysis_core::{
    AccuracyTolerance, AnalysisProcedure, ErrorContributionKind, Frequency, IndependentDomain,
    PhysicsFamily, ProcedureSettings, RestartPolicy, SolutionSpace, StateCompleteness, StateKind,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    execution::AcousticHarmonicWorkload,
    frequency::{FrequencyResponsePlan, FrequencyResponsePlanEntry},
    physics::acoustic::AcousticHarmonicPolicy,
};

use super::CompiledAnalysisStudy;

#[cfg(test)]
mod tests;

/// Placement-neutral numerical controls for the currently supported canonical CPU formulation.
///
/// The convergence tolerance is not duplicated here: it is derived from the strictest authored
/// relative algebraic criterion. Power closure remains an execution admission until the authored
/// accuracy vocabulary grows a dedicated acoustic-power criterion.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticStudyPlanningPolicy {
    pub power_balance_relative_tolerance: f64,
    /// Limit on the factorization certificate's contraction bound for every frequency.
    pub contraction_limit: f64,
}

impl AcousticStudyPlanningPolicy {
    pub const fn standard() -> Self {
        Self {
            power_balance_relative_tolerance: 1.0e-8,
            contraction_limit: crate::solve::certificate::DEFAULT_CONTRACTION_LIMIT,
        }
    }

    fn validate(self) -> Result<(), AcousticStudyPlanningError> {
        if !open_unit_interval(self.power_balance_relative_tolerance)
            || !open_unit_interval(self.contraction_limit)
        {
            return Err(AcousticStudyPlanningError::InvalidNumericalPolicy);
        }
        Ok(())
    }
}

/// One immutable member of the canonical frequency plan and its exact numerical workload.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedAcousticFrequency {
    entry: FrequencyResponsePlanEntry,
    workload: AcousticHarmonicWorkload,
}

impl PlannedAcousticFrequency {
    pub const fn entry(&self) -> FrequencyResponsePlanEntry {
        self.entry
    }

    pub const fn workload(&self) -> &AcousticHarmonicWorkload {
        &self.workload
    }
}

/// Canonically ordered independent-frequency work for one authored acoustic step.
#[derive(Clone, Debug, PartialEq)]
pub struct AcousticStudyPlan {
    frequency_plan: FrequencyResponsePlan,
    frequencies: Vec<PlannedAcousticFrequency>,
}

impl AcousticStudyPlan {
    pub const fn frequency_plan(&self) -> &FrequencyResponsePlan {
        &self.frequency_plan
    }

    pub fn frequencies(&self) -> &[PlannedAcousticFrequency] {
        &self.frequencies
    }
}

/// Resolve exactly one standalone authored acoustic harmonic step into independently executable
/// canonical workloads. This does not schedule the entries or claim parallel execution.
pub fn plan_acoustic_study(
    study: &CompiledAnalysisStudy,
    policy: AcousticStudyPlanningPolicy,
) -> Result<AcousticStudyPlan, AcousticStudyPlanningError> {
    policy.validate()?;
    study.definition_digest()?;
    let [step] = study.model.sequence.steps.as_slice() else {
        return Err(AcousticStudyPlanningError::UnsupportedSequence);
    };
    if !study.model.sequence.dependencies.is_empty()
        || !study.model.sequence.transitions.is_empty()
        || !study.model.sequence.branches.is_empty()
        || !step.activations.is_empty()
        || !step.state_inputs.is_empty()
        || step.restart != RestartPolicy::Fresh
    {
        return Err(AcousticStudyPlanningError::UnsupportedSequence);
    }
    if !matches!(
        step.state_outputs.as_slice(),
        [output]
            if output.kind == StateKind::AcousticComplex
                && output.completeness == StateCompleteness::Complete
    ) {
        return Err(AcousticStudyPlanningError::UnsupportedStatePublication);
    }
    if step.family != PhysicsFamily::Acoustic
        || step.procedure != AnalysisProcedure::HarmonicResponse
        || step.solution_space != SolutionSpace::Direct
        || step.settings != ProcedureSettings::None
    {
        return Err(AcousticStudyPlanningError::UnsupportedProcedure);
    }
    let IndependentDomain::FrequencyHertz(authored_frequencies) = &step.independent_domain else {
        return Err(AcousticStudyPlanningError::UnsupportedProcedure);
    };
    let reference = authored_frequencies
        .first()
        .copied()
        .ok_or(AcousticStudyPlanningError::EmptyFrequencyDomain)?;
    let frequency_plan = FrequencyResponsePlan::new(reference, authored_frequencies, &[])
        .map_err(|error| AcousticStudyPlanningError::Frequency(error.to_string()))?;
    if frequency_plan.frequencies_hz().len() > step.accuracy.resources.maximum_samples {
        return Err(AcousticStudyPlanningError::SampleLimitExceeded);
    }
    let relative_tolerance = step
        .accuracy
        .criteria
        .iter()
        .filter(|criterion| criterion.contribution == ErrorContributionKind::Algebraic)
        .filter_map(|criterion| match criterion.tolerance {
            AccuracyTolerance::Relative(value) => Some(value),
            AccuracyTolerance::Absolute { .. } => None,
        })
        .reduce(f64::min)
        .filter(|value| open_unit_interval(*value))
        .ok_or(AcousticStudyPlanningError::MissingRelativeAlgebraicTarget)?;
    let solver = AcousticHarmonicPolicy {
        contraction_limit: policy.contraction_limit,
        relative_tolerance,
    };
    let frequencies = frequency_plan
        .entries()
        .map_err(|error| AcousticStudyPlanningError::Frequency(error.to_string()))?
        .into_iter()
        .map(|entry| {
            let workload = AcousticHarmonicWorkload {
                model: study.model.clone(),
                step_id: step.step_id.clone(),
                frequency: Frequency::new(entry.frequency_hz()),
                solver,
                power_balance_relative_tolerance: policy.power_balance_relative_tolerance,
            };
            workload.validate()?;
            Ok(PlannedAcousticFrequency { entry, workload })
        })
        .collect::<Result<Vec<_>, AcousticStudyPlanningError>>()?;
    Ok(AcousticStudyPlan {
        frequency_plan,
        frequencies,
    })
}

const fn open_unit_interval(value: f64) -> bool {
    value.is_finite() && value > 0.0 && value < 1.0
}

#[derive(Debug, Error)]
pub enum AcousticStudyPlanningError {
    #[error("acoustic planning requires exactly one fresh independent procedure step")]
    UnsupportedSequence,
    #[error("acoustic planning currently supports only direct harmonic response")]
    UnsupportedProcedure,
    #[error("acoustic planning requires exactly one complete acoustic-complex state output")]
    UnsupportedStatePublication,
    #[error("acoustic harmonic response requires at least one authored frequency")]
    EmptyFrequencyDomain,
    #[error("acoustic frequency count exceeds the authored resource ceiling")]
    SampleLimitExceeded,
    #[error("acoustic step requires an authored relative algebraic accuracy target")]
    MissingRelativeAlgebraicTarget,
    #[error("acoustic numerical policy is outside its finite bounds")]
    InvalidNumericalPolicy,
    #[error("acoustic frequency planning failed: {0}")]
    Frequency(String),
    #[error(transparent)]
    Compilation(#[from] super::AnalysisStudyCompilationError),
    #[error(transparent)]
    Workload(#[from] crate::execution::AnalysisExecutionError),
}
