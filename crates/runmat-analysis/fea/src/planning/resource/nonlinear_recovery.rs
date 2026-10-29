//! Original J2 recovery reserves the shared physical basis, raw result and field construction.
use super::*;
use crate::execution::{
    nonlinear_native_field_definitions, FieldRecoveryRequest, NonlinearSolidBlockArtifact,
    NonlinearSolidWorkload,
};
use runmat_meshing_core::SolverMeshArtifact;

pub fn nonlinear_field_recovery_shape(
    request: &FieldRecoveryRequest,
) -> Result<(Vec<u64>, ByteCount), ResourcePlanningError> {
    let definitions = nonlinear_native_field_definitions(request)
        .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
    super::recovered_field_shape::shape(request, definitions)
}

/// Shared read-only physical preparation, accepted-state validation, raw history
/// copies, field projection and observations. No Newton/Krylov work or new state catalog.
/// This is requested storage, not measured RSS.
pub fn nonlinear_recovery_workspace(
    workload: &NonlinearSolidWorkload,
    mesh: &SolverMeshArtifact,
    block: &NonlinearSolidBlockArtifact,
) -> Result<ByteCount, ResourcePlanningError> {
    let input =
        super::nonlinear::block::derive_recovery(workload, mesh, block.result.load_factors.len())?;
    Ok(ByteCount(build_resource_plan(input)?.request.memory_bytes))
}

pub fn build_nonlinear_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    recovery_workspace_bytes: ByteCount,
    facts: PreparedPhaseResourceFacts,
    local_execution: super::LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let (partitions, construction_workspace_bytes) = nonlinear_field_recovery_shape(request)?;
    let bytes = partitions.iter().try_fold(0u64, |n, v| {
        n.checked_add(*v)
            .ok_or(ResourcePlanningError::Overflow("nonlinear recovery output"))
    })?;
    if facts.output.encoded_payload_bytes_upper_bound.0 != bytes {
        return Err(ResourcePlanningError::Invalid(
            "nonlinear field output profile differs".into(),
        ));
    }
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::NonlinearFieldRecovery(request.identity()),
        demand: FeaResourceDemand::NonlinearFieldRecovery(NonlinearFieldRecoveryResourceDemand {
            request: request.identity(),
            source_result: request.source.solver_result().ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "nonlinear recovery requires original block result".into(),
                )
            })?,
            topology_digest: request.topology.discretization_digest(),
            construction_workspace_bytes,
            recovery_workspace_bytes,
            inputs: facts.inputs,
            output: facts.output,
        }),
        output_set: Some(facts.output_set),
        worker: super::post_process::post_process_worker_on(local_execution)?,
        accuracy_ceiling: super::post_process::post_process_ceiling(),
    })
}
