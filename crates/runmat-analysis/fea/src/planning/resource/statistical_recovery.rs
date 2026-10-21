//! Retained native copies of statistical-step fields: catalog and partition materialization.
use super::*;
use crate::execution::{random_response_field_definitions, FieldRecoveryRequest};

/// Canonical catalog and storage partitions of a random-response field recovery.
pub fn random_response_field_recovery_shape(
    request: &FieldRecoveryRequest,
) -> Result<(Vec<u64>, ByteCount), ResourcePlanningError> {
    let definitions = random_response_field_definitions(request)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    super::recovered_field_shape::shape(request, definitions)
}

pub fn build_random_response_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    source_validation_workspace_bytes: ByteCount,
    facts: PreparedPhaseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let (partitions, construction_workspace_bytes) =
        random_response_field_recovery_shape(request)?;
    let bytes = partitions.iter().try_fold(0u64, |n, v| {
        n.checked_add(*v)
            .ok_or(ResourcePlanningError::Overflow("statistical recovery output"))
    })?;
    if facts.output.encoded_payload_bytes_upper_bound.0 != bytes {
        return Err(ResourcePlanningError::Invalid(
            "statistical field output profile differs".into(),
        ));
    }
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StatisticalFieldRecovery(request.identity()),
        demand: FeaResourceDemand::StatisticalFieldRecovery(MagneticFieldRecoveryResourceDemand {
            request: request.identity(),
            source_result: request.source.solver_result().ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "statistical recovery requires its original solver result".into(),
                )
            })?,
            topology_digest: request.topology.discretization_digest(),
            construction_workspace_bytes,
            source_validation_workspace_bytes,
            inputs: facts.inputs,
            output: facts.output,
        }),
        output_set: Some(facts.output_set),
        worker: super::post_process::post_process_worker(),
        accuracy_ceiling: super::post_process::post_process_ceiling(),
    })
}
