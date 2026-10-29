//! Retained native copies: shared identity/catalog/partition materialization.
use super::*;
use crate::execution::{maxwell_native_field_definitions, FieldRecoveryRequest};
use crate::results::field::RECOVERED_FIELD_PARTITION_ENTITY_COUNT;
#[cfg(test)]
mod tests;

/// Catalog plus canonical storage partitions, without visiting projected values.
/// Transport may split each partition into multiple objects, so this is a lower
/// bound for the output object inventory, excluding its transport manifest.
pub fn maxwell_field_recovery_partition_count(
    request: &FieldRecoveryRequest,
) -> Result<usize, ResourcePlanningError> {
    maxwell_native_field_definitions(request)
        .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
    request
        .plan
        .requirements
        .iter()
        .try_fold(1usize, |total, r| {
            let count = usize::try_from(r.cost.output_entity_count)
                .map_err(|_| ResourcePlanningError::Overflow("maxwell native entities"))?;
            total
                .checked_add(count.div_ceil(RECOVERED_FIELD_PARTITION_ENTITY_COUNT))
                .ok_or(ResourcePlanningError::Overflow(
                    "maxwell recovery partitions",
                ))
        })
}

pub fn maxwell_field_recovery_shape(
    request: &FieldRecoveryRequest,
) -> Result<(Vec<u64>, ByteCount), ResourcePlanningError> {
    let invalid = |error: &dyn std::fmt::Display| ResourcePlanningError::Invalid(error.to_string());
    let definitions = maxwell_native_field_definitions(request).map_err(|e| invalid(&e))?;
    super::recovered_field_shape::shape(request, definitions)
}

pub fn build_maxwell_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    source_validation_workspace_bytes: ByteCount,
    facts: PreparedPhaseResourceFacts,
    local_execution: super::LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let (partitions, construction_workspace_bytes) = maxwell_field_recovery_shape(request)?;
    let bytes = partitions.iter().try_fold(0u64, |n, v| {
        n.checked_add(*v)
            .ok_or(ResourcePlanningError::Overflow("maxwell recovery output"))
    })?;
    if facts.output.encoded_payload_bytes_upper_bound.0 != bytes {
        return Err(ResourcePlanningError::Invalid(
            "maxwell field output profile differs".into(),
        ));
    }
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::MaxwellFieldRecovery(request.identity()),
        demand: FeaResourceDemand::MaxwellFieldRecovery(MaxwellFieldRecoveryResourceDemand {
            request: request.identity(),
            source_result: request.source.solver_result().ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "maxwell recovery requires original solver result".into(),
                )
            })?,
            topology_digest: request.topology.discretization_digest(),
            construction_workspace_bytes,
            source_validation_workspace_bytes,
            inputs: facts.inputs,
            output: facts.output,
        }),
        output_set: Some(facts.output_set),
        worker: super::post_process::post_process_worker_on(local_execution)?,
        accuracy_ceiling: super::post_process::post_process_ceiling(),
    })
}

/// Exact count/profile authority before source validation allocates derived maps.
pub fn maxwell_recovery_source_validation_workspace(
    workload: &crate::execution::MaxwellHarmonicWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> Result<ByteCount, ResourcePlanningError> {
    crate::physics::maxwell::resources::require_selected_ceiling(
        &workload.model,
        mesh,
        workload.step_id.as_str(),
    )
    .map(|reservation| ByteCount(reservation.total_bytes))
    .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))
}
