//! Retained native acoustic copies: shared identity/catalog/partition materialization.
use super::*;
use crate::execution::{AcousticHarmonicWorkload, FieldRecoveryRequest};
use crate::results::field::{RecoveredFieldSample, RECOVERED_FIELD_PARTITION_ENTITY_COUNT};

/// Catalog plus canonical storage partitions, without visiting projected values.
pub fn acoustic_field_recovery_partition_count(
    request: &FieldRecoveryRequest,
) -> Result<usize, ResourcePlanningError> {
    request
        .plan
        .requirements
        .iter()
        .try_fold(1usize, |total, r| {
            let count = usize::try_from(r.cost.output_entity_count)
                .map_err(|_| ResourcePlanningError::Overflow("acoustic native entities"))?;
            total
                .checked_add(count.div_ceil(RECOVERED_FIELD_PARTITION_ENTITY_COUNT))
                .ok_or(ResourcePlanningError::Overflow(
                    "acoustic recovery partitions",
                ))
        })
}

/// Exact partition byte profile of the native pressure/velocity fields at one sample.
pub fn acoustic_field_recovery_shape(
    request: &FieldRecoveryRequest,
    sample: RecoveredFieldSample,
) -> Result<(Vec<u64>, ByteCount), ResourcePlanningError> {
    let invalid = |error: &dyn std::fmt::Display| ResourcePlanningError::Invalid(error.to_string());
    let definitions =
        crate::execution::acoustic_native_field_definitions(request).map_err(|e| invalid(&e))?;
    super::recovered_field_shape::shape_at(request, definitions, sample)
}

pub fn build_acoustic_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    sample: RecoveredFieldSample,
    source_validation_workspace_bytes: ByteCount,
    facts: PreparedPhaseResourceFacts,
    local_execution: super::LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let (partitions, construction_workspace_bytes) =
        acoustic_field_recovery_shape(request, sample)?;
    let bytes = partitions.iter().try_fold(0u64, |n, v| {
        n.checked_add(*v)
            .ok_or(ResourcePlanningError::Overflow("acoustic recovery output"))
    })?;
    if facts.output.encoded_payload_bytes_upper_bound.0 != bytes {
        return Err(ResourcePlanningError::Invalid(
            "acoustic field output profile differs".into(),
        ));
    }
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::AcousticFieldRecovery(request.identity()),
        demand: FeaResourceDemand::AcousticFieldRecovery(AcousticFieldRecoveryResourceDemand {
            request: request.identity(),
            source_result: request.source.solver_result().ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "acoustic recovery requires original solver result".into(),
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
pub fn acoustic_recovery_source_validation_workspace(
    workload: &AcousticHarmonicWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> Result<ByteCount, ResourcePlanningError> {
    let result = super::acoustic_harmonic_result_payload_bytes_upper_bound(workload, mesh)?;
    let projection = crate::topology::AnalysisProjection::derivation_resident_bytes_estimate(mesh)
        .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
    result
        .checked_add(projection)
        .map(ByteCount)
        .ok_or(ResourcePlanningError::Overflow(
            "acoustic recovery source validation",
        ))
}
