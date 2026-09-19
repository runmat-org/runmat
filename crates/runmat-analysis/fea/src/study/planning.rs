use std::collections::BTreeMap;

use runmat_analysis_core::{
    AccuracyTolerance, AnalysisProcedure, ErrorContributionKind, FormulationRequirement,
    IndependentDomain, PhysicsFamily, ProcedureStep, SolutionSpace,
};
use thiserror::Error;

use crate::execution::{
    AnalysisExecutionError, StructuralLinearWorkload, StructuralStudyWorkloads,
    StructuralTransientBlockPlan, StructuralTransientSolverPlan, StructuralTransientWorkload,
};
use crate::results::observation::ObservationExecutionPolicy;
use crate::solve::backend::kind::LinearAlgebraBackendKind;
use crate::solve::preconditioner::SpdPreconditionerKind;
use crate::solve::transient::StructuralTimeStepPolicy;
use crate::ComputeBackend;

use super::CompiledAnalysisStudy;

/// Already-admitted numerical choices for translating scientific intent into portable workloads.
///
/// The runtime resolves `auto`, project policy, CLI policy, and available capabilities before this
/// boundary. No worker or scientific compiler performs a hardware fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StructuralStudyPlanningPolicy {
    pub backend: ComputeBackend,
    pub formulation_requirement: FormulationRequirement,
}

/// Build portable structural workloads from the canonical compiled procedure sequence.
///
/// Static sequences and a standalone direct transient study are admitted without reconstructing
/// scientific inputs from runtime options. Other typed procedures remain valid scientific
/// contracts but fail planning until their workload constructors move behind this same owner.
pub fn plan_structural_study(
    study: &CompiledAnalysisStudy,
    policy: StructuralStudyPlanningPolicy,
) -> Result<StructuralStudyWorkloads, StructuralStudyPlanningError> {
    if let [step] = study.model.sequence.steps.as_slice() {
        if step.family == PhysicsFamily::Structural
            && step.procedure == AnalysisProcedure::TransientResponse
            && step.solution_space == SolutionSpace::Direct
        {
            return plan_direct_transient(study, step, policy);
        }
    }

    let algebra_backend = match policy.backend {
        ComputeBackend::Cpu => LinearAlgebraBackendKind::CpuReference,
        ComputeBackend::Gpu => LinearAlgebraBackendKind::Accelerator,
    };
    let mut workloads = BTreeMap::new();
    for step in &study.model.sequence.steps {
        if step.family != PhysicsFamily::Structural
            || step.procedure != AnalysisProcedure::Static
            || step.solution_space != SolutionSpace::Direct
        {
            return Err(StructuralStudyPlanningError::UnsupportedProcedure {
                step_id: step.step_id.0.clone(),
                family: step.family,
                procedure: step.procedure,
                solution_space: step.solution_space,
            });
        }
        let workload = StructuralLinearWorkload {
            schema_version: 1,
            model: study.model.clone(),
            step_id: step.step_id.clone(),
            backend: policy.backend,
            formulation_requirement: policy.formulation_requirement,
            preconditioner: SpdPreconditionerKind::Jacobi,
            algebra_backend,
            observation_execution: ObservationExecutionPolicy::scalable(),
            require_solver_mesh_for_solid: true,
            initial_guess_transition: None,
        };
        workload.validate_sequence_template()?;
        workloads.insert(step.step_id.clone(), workload);
    }
    let planned = StructuralStudyWorkloads::LinearSequence { workloads };
    planned.validate()?;
    Ok(planned)
}

fn plan_direct_transient(
    study: &CompiledAnalysisStudy,
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
) -> Result<StructuralStudyWorkloads, StructuralStudyPlanningError> {
    let IndependentDomain::TimeSeconds { interval_count, .. } = &step.independent_domain else {
        return Err(StructuralStudyPlanningError::InvalidTransientDomain {
            step_id: step.step_id.0.clone(),
        });
    };
    let interval_count = *interval_count;
    let interval_count_u64 = u64::try_from(interval_count).map_err(|_| {
        StructuralStudyPlanningError::TransientIntervalCountOverflow {
            step_id: step.step_id.0.clone(),
            interval_count,
        }
    })?;
    let residual_target = algebraic_residual_target(step)?;
    let workload = StructuralTransientWorkload {
        schema_version: 1,
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        backend: policy.backend,
        formulation_requirement: policy.formulation_requirement,
        solver: StructuralTransientSolverPlan::Direct,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count },
        residual_target,
    };
    let block_plan = StructuralTransientBlockPlan::new(interval_count_u64.min(256))?;
    let planned = StructuralStudyWorkloads::StructuralTransient {
        workload: Box::new(workload),
        block_plan,
    };
    planned.validate()?;
    Ok(planned)
}

fn algebraic_residual_target(step: &ProcedureStep) -> Result<f64, StructuralStudyPlanningError> {
    step.accuracy
        .criteria
        .iter()
        .filter(|criterion| criterion.contribution == ErrorContributionKind::Algebraic)
        .filter_map(|criterion| match criterion.tolerance {
            AccuracyTolerance::Relative(value) => Some(value),
            AccuracyTolerance::Absolute { .. } => None,
        })
        .reduce(f64::min)
        .ok_or_else(
            || StructuralStudyPlanningError::MissingAlgebraicResidualTarget {
                step_id: step.step_id.0.clone(),
            },
        )
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
    #[error("transient step `{step_id}` does not declare a physical time domain")]
    InvalidTransientDomain { step_id: String },
    #[error(
        "transient step `{step_id}` interval count {interval_count} exceeds the portable execution contract"
    )]
    TransientIntervalCountOverflow {
        step_id: String,
        interval_count: usize,
    },
    #[error("transient step `{step_id}` has no relative algebraic accuracy criterion")]
    MissingAlgebraicResidualTarget { step_id: String },
    #[error(transparent)]
    Workload(#[from] AnalysisExecutionError),
}
