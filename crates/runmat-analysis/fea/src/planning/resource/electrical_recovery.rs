//! Retained native electrical node/element copies for authored observations.
use super::*;
use crate::execution::{electrical_native_field_definitions, FieldRecoveryRequest};
use crate::results::field::RECOVERED_FIELD_PARTITION_ENTITY_COUNT;

/// Catalog plus canonical storage partitions, without visiting projected values.
pub fn electrical_field_recovery_partition_count(
    request: &FieldRecoveryRequest,
) -> Result<usize, ResourcePlanningError> {
    electrical_native_field_definitions(request)
        .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
    request
        .plan
        .requirements
        .iter()
        .try_fold(1usize, |total, r| {
            let count = usize::try_from(r.cost.output_entity_count)
                .map_err(|_| ResourcePlanningError::Overflow("electrical native entities"))?;
            total
                .checked_add(count.div_ceil(RECOVERED_FIELD_PARTITION_ENTITY_COUNT))
                .ok_or(ResourcePlanningError::Overflow(
                    "electrical recovery partitions",
                ))
        })
}

pub fn electrical_field_recovery_shape(
    request: &FieldRecoveryRequest,
) -> Result<(Vec<u64>, ByteCount), ResourcePlanningError> {
    let invalid = |error: &dyn std::fmt::Display| ResourcePlanningError::Invalid(error.to_string());
    let definitions = electrical_native_field_definitions(request).map_err(|e| invalid(&e))?;
    super::recovered_field_shape::shape(request, definitions)
}

pub fn build_electrical_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    source_validation_workspace_bytes: ByteCount,
    facts: PreparedPhaseResourceFacts,
    local_execution: super::LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let (partitions, construction_workspace_bytes) = electrical_field_recovery_shape(request)?;
    let bytes = partitions.iter().try_fold(0u64, |n, v| {
        n.checked_add(*v).ok_or(ResourcePlanningError::Overflow(
            "electrical recovery output",
        ))
    })?;
    if facts.output.encoded_payload_bytes_upper_bound.0 != bytes {
        return Err(ResourcePlanningError::Invalid(
            "electrical field output profile differs".into(),
        ));
    }
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::ElectricalFieldRecovery(request.identity()),
        demand: FeaResourceDemand::ElectricalFieldRecovery(ElectricalFieldRecoveryResourceDemand {
            request: request.identity(),
            source_result: request.source.solver_result().ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "electrical recovery requires original solver result".into(),
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

/// Decoding the retained result and validating it against the workload and mesh projection.
pub fn electrical_recovery_source_validation_workspace(
    workload: &crate::execution::ElectricalStaticWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> Result<ByteCount, ResourcePlanningError> {
    let result = super::electrical_static_result_payload_bytes_upper_bound(workload, mesh)?;
    let projection = crate::topology::AnalysisProjection::derivation_resident_bytes_estimate(mesh)
        .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
    result
        .checked_add(projection)
        .and_then(|bytes| bytes.checked_mul(2))
        .map(ByteCount)
        .ok_or(ResourcePlanningError::Overflow(
            "electrical recovery source validation",
        ))
}
