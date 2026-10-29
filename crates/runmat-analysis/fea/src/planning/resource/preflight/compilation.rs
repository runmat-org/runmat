//! Fresh original construction evidence for canonical static and derived nonlinear requests.
use super::*;

/// Reuse original bounded construction without recompiling or accepting serialized inventory.
pub fn build_structural_resource_plan_from_compilation(
    mesh: Option<&SolverMeshArtifact>,
    compilation: &crate::constraints::ConstraintCompilation<'_>,
    observations: &ResolvedObservationPlan,
    workload: &StructuralLinearWorkload,
    accelerator_requests: &[AcceleratorRequest],
    local_execution: LocalExecutionPlan,
    prepared_inputs: Option<&PreparedInputResourceFacts>,
    selection: StaticSolveSelection,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    workload
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    build_static_resource_plan_selecting(
        StaticResourcePreflight {
            compilation: Some(compilation),
            model: &workload.model,
            step: workload
                .procedure_step()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            backend: workload.backend,
            preconditioner: workload.preconditioner,
            observation_execution: workload.observation_execution,
            mesh,
            constraints: compilation.plan(),
            observations,
            accelerator_requests,
            local_execution,
            prepared_inputs,
        },
        selection,
    )
}

pub(super) fn original_peak(
    input: &StaticResourcePreflight<'_>,
) -> Result<ByteCount, ResourcePlanningError> {
    let owned = if input.compilation.is_none() {
        Some(
            CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
                input.model,
                input.mesh,
                input.step.accuracy.resources.resident_byte_budget(),
            )
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        )
    } else {
        None
    };
    let compiled = input
        .compilation
        .or(owned.as_ref())
        .expect("fresh or carried compilation");
    super::super::constraint_compilation::original_compilation_peak(
        compiled,
        input.model,
        input.mesh,
        input.constraints,
    )
}
