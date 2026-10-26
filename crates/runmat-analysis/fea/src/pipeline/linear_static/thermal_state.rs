//! Static physical consumption of an immutable, topology-bound thermal solution.

use runmat_analysis_core::{AnalysisModel, AnalysisProcedure, ProcedureStep, ProcedureStepId};

#[cfg(test)]
mod qualification;
#[cfg(test)]
mod tests;

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    contracts::{ComputeBackend, FeaRunError, FeaRunResult, LinearStaticSolveOptions},
    execution::{CanonicalSemanticState, CanonicalStateTransition, SolverResultIdentity},
};

/// Authored thermal source and exact prepared numerical admission for one selected consumer.
pub struct PreparedThermoelasticStaticInput<'a> {
    pub consumer_step_id: &'a ProcedureStepId,
    pub source: &'a CanonicalSemanticState,
    pub constraint_plan: &'a CanonicalStructuralConstraintPlan,
    pub admitted: Option<super::StaticResourceAdmission<'a>>,
}

pub(super) struct ThermalStateSelection<'a> {
    pub consumer_step_id: &'a ProcedureStepId,
    pub source: &'a CanonicalSemanticState,
}

pub struct ThermoelasticStaticResult {
    pub run: FeaRunResult,
    pub thermal_transition: CanonicalStateTransition,
    pub solver_result_identity: SolverResultIdentity,
    /// Physical initial-stress operator from this exact solve and thermal transition.
    pub linearization: crate::pipeline::structural_linearization::StructuralLinearizationArtifact,
}

/// Consume the exact retained nodal temperature field, not a prescribed profile or solver guess.
pub fn run_linear_static_from_thermal_state(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
    source: &CanonicalSemanticState,
) -> Result<ThermoelasticStaticResult, FeaRunError> {
    let step = super::super::admission::require_structural(model, AnalysisProcedure::Static)?;
    let result = super::run_linear_static_with_constraint_plan(
        model,
        backend,
        options,
        None,
        None,
        Some(ThermalStateSelection {
            consumer_step_id: &step.step_id,
            source,
        }),
    )?;
    finalize(model, result)
}

/// Selects a consumer and binds its immutable source against the complete authored sequence.
/// Constraints, resource admission, assembly and recovery use that same selected step; the
/// producer and transition are never erased to manufacture a single-step source identity.
pub fn run_linear_static_from_thermal_state_with_prepared_constraint_plan(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
    input: PreparedThermoelasticStaticInput<'_>,
) -> Result<ThermoelasticStaticResult, FeaRunError> {
    require_selected_step(model, input.consumer_step_id)?;
    let mesh = options.solver_mesh.as_ref().ok_or_else(|| {
        FeaRunError::InvalidModel(
            "thermal state consumption requires a canonical solver mesh".into(),
        )
    })?;
    input
        .constraint_plan
        .validate_against(model, mesh)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let result = super::run_linear_static_with_constraint_plan(
        model,
        backend,
        options,
        Some(input.constraint_plan),
        input.admitted,
        Some(ThermalStateSelection {
            consumer_step_id: input.consumer_step_id,
            source: input.source,
        }),
    )?;
    finalize(model, result)
}

pub(super) fn require_selected_step<'a>(
    model: &'a AnalysisModel,
    step_id: &ProcedureStepId,
) -> Result<&'a ProcedureStep, FeaRunError> {
    runmat_analysis_core::validate_model_sequence(model)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let step = model
        .sequence
        .steps
        .iter()
        .find(|step| &step.step_id == step_id)
        .ok_or_else(|| {
            FeaRunError::InvalidModel("selected thermoelastic consumer is absent".into())
        })?;
    if step.procedure != AnalysisProcedure::Static
        || step.family
            != runmat_analysis_core::PhysicsFamily::Coupled(
                runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
            )
    {
        return Err(FeaRunError::InvalidModel(
            "selected thermal consumer must be a thermomechanical static procedure".into(),
        ));
    }
    super::super::admission::require_structural_step(model, step)?;
    Ok(step)
}

pub(super) fn bind_source(
    model: &AnalysisModel,
    step_id: &ProcedureStepId,
    options: &LinearStaticSolveOptions,
    source: &CanonicalSemanticState,
) -> Result<crate::execution::BoundThermalStateTransfer, FeaRunError> {
    let mesh = options.solver_mesh.as_ref().ok_or_else(|| {
        FeaRunError::InvalidModel(
            "thermal state consumption requires a canonical solver mesh".into(),
        )
    })?;
    crate::execution::BoundThermalStateTransfer::bind(model, step_id, source, mesh)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))
}

fn finalize(
    model: &AnalysisModel,
    result: super::LinearStaticComputation,
) -> Result<ThermoelasticStaticResult, FeaRunError> {
    let thermal_transition = result.thermal_transition.ok_or_else(|| {
        FeaRunError::Assembly(
            "solved thermoelastic execution did not retain its source transition".into(),
        )
    })?;
    let linearization = crate::pipeline::structural_linearization::StructuralLinearizationArtifact::from_thermal_solution(
        &result.summary,
        &result.run,
        &result.solution,
        thermal_transition.clone(),
        model,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let solver_result_identity = linearization
        .thermal_source()
        .ok_or_else(|| {
            FeaRunError::Assembly(
                "thermal linearization omitted its physical source identity".into(),
            )
        })?
        .structural_result();
    Ok(ThermoelasticStaticResult {
        run: result.run,
        thermal_transition,
        solver_result_identity,
        linearization,
    })
}
