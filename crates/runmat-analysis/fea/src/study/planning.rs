use crate::pipeline::modal_governance::ModalConvergencePolicy;
use std::collections::BTreeMap;

use runmat_analysis_core::{
    AccuracyTolerance, AnalysisProcedure, ErrorContributionKind, PhysicsFamily, ProcedureStep,
    SolutionSpace,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::execution::{
    AnalysisExecutionError, StructuralLinearWorkload, StructuralStudyWorkloads,
};
use crate::results::observation::ObservationExecutionPolicy;
use crate::solve::backend::kind::LinearAlgebraBackendKind;
use crate::solve::preconditioner::SpdPreconditionerKind;
use crate::ComputeBackend;

use super::CompiledAnalysisStudy;

mod dynamics;
mod initial_fields;
#[cfg(test)]
mod tests;

/// Portable numerical limits selected after project/CLI policy resolution.
///
/// Physical domains and accuracy tolerances remain in the compiled study. These values govern
/// algorithms and task partitioning only. Physical domains, accuracy tolerances, damping, and
/// modal-mass controls remain in the compiled study.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralNumericalPlanningPolicy {
    pub maximum_transient_intervals_per_block: u64,
    pub maximum_frequency_intervals_per_task: usize,
    pub maximum_modal_iterations: usize,
    pub maximum_modal_subspace_dimension: usize,
    pub modal_eigenspace_relative_tolerance: f64,
    pub harmonic_restart: usize,
    pub maximum_harmonic_iterations: usize,
    pub maximum_buckling_iterations: usize,
}

impl StructuralNumericalPlanningPolicy {
    pub const fn production() -> Self {
        Self {
            maximum_transient_intervals_per_block: 256,
            maximum_frequency_intervals_per_task: 32,
            maximum_modal_iterations: 256,
            maximum_modal_subspace_dimension: ModalConvergencePolicy::production()
                .maximum_subspace_dimension(),
            modal_eigenspace_relative_tolerance: 1.0e-6,
            harmonic_restart: 32,
            maximum_harmonic_iterations: 2_048,
            maximum_buckling_iterations: 256,
        }
    }

    pub fn validate(self) -> Result<(), StructuralStudyPlanningError> {
        if self.maximum_transient_intervals_per_block == 0
            || self.maximum_frequency_intervals_per_task == 0
            || self.maximum_modal_iterations == 0
            || self.maximum_modal_subspace_dimension == 0
            || !open_unit_interval(self.modal_eigenspace_relative_tolerance)
            || self.harmonic_restart == 0
            || self.maximum_harmonic_iterations == 0
            || self.maximum_buckling_iterations == 0
        {
            return Err(StructuralStudyPlanningError::InvalidNumericalPolicy);
        }
        Ok(())
    }
}

const fn open_unit_interval(value: f64) -> bool {
    value.is_finite() && value > 0.0 && value < 1.0
}

/// Already-admitted choices for translating a compiled study definition into portable workloads.
///
/// The runtime resolves `auto`, project policy, CLI policy, and available capabilities before this
/// boundary. No worker or study compiler performs a hardware fallback.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralStudyPlanningPolicy {
    pub backend: ComputeBackend,
    pub numerics: StructuralNumericalPlanningPolicy,
}

/// Build portable structural workloads from the canonical compiled procedure sequence.
pub fn plan_structural_study(
    study: &CompiledAnalysisStudy,
    policy: StructuralStudyPlanningPolicy,
) -> Result<StructuralStudyWorkloads, StructuralStudyPlanningError> {
    plan_structural_study_with_fields(study, policy, &[])
}

/// Plan with closed original sources. Each consumer receives only its active bindings;
/// topology and quantity admission remain with the existing numerical workload owners.
pub fn plan_structural_study_with_fields(
    study: &CompiledAnalysisStudy,
    policy: StructuralStudyPlanningPolicy,
    fields: &[crate::execution::InitialFieldBinding],
) -> Result<StructuralStudyWorkloads, StructuralStudyPlanningError> {
    policy.numerics.validate()?;
    initial_fields::validate_inventory(&study.model, fields)?;
    if study.model.sequence.steps.iter().all(is_linear_static_step) {
        return plan_linear_sequence(study, policy);
    }
    dynamics::plan(study, policy, fields)
}

/// Linear static structural steps, including thermal stress from a prescribed temperature
/// field (a thermomechanical static step that consumes no solved thermal state).
fn is_linear_static_step(step: &ProcedureStep) -> bool {
    let prescribed_thermal_stress = step.family
        == PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical)
        && step.state_inputs.is_empty();
    (step.family == PhysicsFamily::Structural || prescribed_thermal_stress)
        && step.procedure == AnalysisProcedure::Static
        && step.solution_space == SolutionSpace::Direct
}

fn plan_linear_sequence(
    study: &CompiledAnalysisStudy,
    policy: StructuralStudyPlanningPolicy,
) -> Result<StructuralStudyWorkloads, StructuralStudyPlanningError> {
    let workloads = study
        .model
        .sequence
        .steps
        .iter()
        .map(|step| Ok((step.step_id.clone(), linear_workload(study, step, policy)?)))
        .collect::<Result<BTreeMap<_, _>, StructuralStudyPlanningError>>()?;
    let planned = StructuralStudyWorkloads::LinearSequence { workloads };
    planned.validate()?;
    Ok(planned)
}

fn linear_workload(
    study: &CompiledAnalysisStudy,
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
) -> Result<StructuralLinearWorkload, StructuralStudyPlanningError> {
    let thermal_preload = step.family
        == PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical)
        && step.procedure == AnalysisProcedure::Static
        && step.solution_space == SolutionSpace::Direct
        && matches!(step.state_inputs.as_slice(), [input] if input.required_kind == runmat_analysis_core::StateKind::Thermal && input.use_as == runmat_analysis_core::StateUse::Prestress);
    if !(is_linear_static_step(step) || thermal_preload) {
        return Err(unsupported(step));
    }
    let algebra_backend = match policy.backend {
        ComputeBackend::Cpu => LinearAlgebraBackendKind::CpuReference,
        ComputeBackend::Gpu => LinearAlgebraBackendKind::Accelerator,
    };
    let workload = StructuralLinearWorkload {
        schema_version: 2,
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        backend: policy.backend,
        preconditioner: SpdPreconditionerKind::Auto,
        algebra_backend,
        observation_execution: ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    workload.validate_sequence_template()?;
    Ok(workload)
}

/// The measure and tolerance of the step's algebraic residual criteria: one shared measure,
/// the tightest relative tolerance.
fn algebraic_residual_target(
    step: &ProcedureStep,
) -> Result<(runmat_analysis_core::AlgebraicAccuracyMeasure, f64), StructuralStudyPlanningError> {
    let missing = || StructuralStudyPlanningError::MissingAlgebraicResidualTarget {
        step_id: step.step_id.0.clone(),
    };
    let measure = step.accuracy.algebraic_measure().ok_or_else(missing)?;
    let tolerance = step
        .accuracy
        .criteria
        .iter()
        .filter(|criterion| {
            criterion.contribution == ErrorContributionKind::Algebraic
                && criterion.evidence == measure.evidence()
        })
        .filter_map(|criterion| match criterion.tolerance {
            AccuracyTolerance::Relative(value) => Some(value),
            AccuracyTolerance::Absolute { .. } => None,
        })
        .reduce(f64::min)
        .ok_or_else(missing)?;
    Ok((measure, tolerance))
}

fn relative_accuracy_target(
    step: &ProcedureStep,
    contribution: ErrorContributionKind,
) -> Result<f64, StructuralStudyPlanningError> {
    step.accuracy
        .criteria
        .iter()
        .filter(|criterion| criterion.contribution == contribution)
        .filter_map(|criterion| match criterion.tolerance {
            AccuracyTolerance::Relative(value) => Some(value),
            AccuracyTolerance::Absolute { .. } => None,
        })
        .reduce(f64::min)
        .ok_or_else(|| match contribution {
            ErrorContributionKind::Algebraic => {
                StructuralStudyPlanningError::MissingAlgebraicResidualTarget {
                    step_id: step.step_id.0.clone(),
                }
            }
            _ => StructuralStudyPlanningError::MissingRelativeAccuracyTarget {
                step_id: step.step_id.0.clone(),
                contribution,
            },
        })
}

fn unsupported(step: &ProcedureStep) -> StructuralStudyPlanningError {
    StructuralStudyPlanningError::UnsupportedProcedure {
        step_id: step.step_id.0.clone(),
        family: step.family,
        procedure: step.procedure,
        solution_space: step.solution_space,
    }
}

#[derive(Debug, Error)]
pub enum StructuralStudyPlanningError {
    #[error(
        "procedure step `{step_id}` ({family:?}/{procedure:?}/{solution_space:?}) has no typed structural workload planner"
    )]
    UnsupportedProcedure {
        step_id: String,
        family: PhysicsFamily,
        procedure: AnalysisProcedure,
        solution_space: SolutionSpace,
    },
    #[error("procedure step `{step_id}` has an incompatible independent domain")]
    InvalidIndependentDomain { step_id: String },
    #[error("procedure step `{step_id}` has incompatible procedure-specific settings")]
    InvalidProcedureSettings { step_id: String },
    #[error(
        "transient step `{step_id}` interval count {interval_count} exceeds the portable execution contract"
    )]
    TransientIntervalCountOverflow {
        step_id: String,
        interval_count: usize,
    },
    #[error("procedure step `{step_id}` has no relative algebraic accuracy criterion")]
    MissingAlgebraicResidualTarget { step_id: String },
    #[error("procedure step `{step_id}` has no relative {contribution:?} accuracy criterion")]
    MissingRelativeAccuracyTarget {
        step_id: String,
        contribution: ErrorContributionKind,
    },
    #[error("structural numerical planning policy is outside its finite bounds")]
    InvalidNumericalPolicy,
    #[error("procedure step `{step_id}` refinement limit exceeds the execution contract")]
    RefinementLimitOverflow { step_id: String },
    #[error(transparent)]
    Workload(#[from] AnalysisExecutionError),
    #[error("frequency planning failed: {0}")]
    Frequency(String),
}
