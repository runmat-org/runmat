//! Retained native acoustic copies: shared identity/catalog/partition materialization.
use super::*;
use crate::execution::{
    AcousticHarmonicWorkload, AcousticResponseSolverResult, AcousticResponseWorkload,
    FieldRecoveryRequest,
};
use crate::results::field::{
    RecoveredField, RecoveredFieldArtifactManifest, RecoveredFieldPartition, RecoveredFieldSample,
    RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
};

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

/// Decoding and validating the retained response result alongside its mesh projection.
pub fn acoustic_response_recovery_source_validation_workspace(
    workload: &AcousticResponseWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> Result<ByteCount, ResourcePlanningError> {
    let result = super::acoustic_response_result_payload_bytes_upper_bound(workload, mesh)?;
    let projection = crate::topology::AnalysisProjection::derivation_resident_bytes_estimate(mesh)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    result
        .checked_add(projection)
        .map(ByteCount)
        .ok_or(ResourcePlanningError::Overflow(
            "acoustic response recovery source validation",
        ))
}

/// The response solver retains all selected samples. Recovery copies those samples into the
/// shared partitioned field format, so reserve every sample and partition before copying any.
pub fn acoustic_response_field_recovery_shape(
    request: &FieldRecoveryRequest,
    result: &AcousticResponseSolverResult,
) -> Result<(Vec<u64>, ByteCount), ResourcePlanningError> {
    request
        .validate()
        .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
    if request
        != &result
            .native_field_request_from_result()
            .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?
    {
        return Err(ResourcePlanningError::Invalid(
            "acoustic response recovery request differs from retained result".into(),
        ));
    }
    let definition = crate::execution::acoustic_response_pressure_definition()
        .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
    let sample = *result.samples().first().ok_or_else(|| {
        ResourcePlanningError::Invalid("acoustic response has no retained pressure sample".into())
    })?;
    let nodes = result.ordered_node_ids().len();
    let partitions_per_sample = nodes.div_ceil(RECOVERED_FIELD_PARTITION_ENTITY_COUNT);
    let partition_count = result
        .samples()
        .len()
        .checked_mul(partitions_per_sample)
        .ok_or(ResourcePlanningError::Overflow(
            "acoustic response partitions",
        ))?;
    let manifest = RecoveredFieldArtifactManifest::profile_sampled_history_shape(
        request,
        std::slice::from_ref(&definition),
        &[nodes],
        sample,
        result.samples().len(),
    )
    .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
    let mut partitions = Vec::with_capacity(partition_count.saturating_add(1));
    partitions.push(manifest.encoded_bytes_upper_bound);
    let mut workspace = manifest.materialization_resident_bytes_estimate;
    for _ in result.samples() {
        let identity = RecoveredField::identity_profile_shape(&definition, sample, nodes)
            .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
        workspace = workspace
            .checked_add(identity.materialization_resident_bytes_estimate)
            .and_then(|n| n.checked_add(identity.encoded_bytes_upper_bound))
            .ok_or(ResourcePlanningError::Overflow(
                "acoustic response field identity workspace",
            ))?;
        let mut remaining = nodes;
        while remaining > 0 {
            let count = remaining.min(RECOVERED_FIELD_PARTITION_ENTITY_COUNT);
            let profile = RecoveredFieldPartition::profile_shape(&definition, sample, count)
                .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
            partitions.push(profile.encoded_bytes_upper_bound);
            workspace = workspace
                .checked_add(profile.materialization_resident_bytes_estimate)
                .and_then(|n| n.checked_add(profile.encoded_bytes_upper_bound))
                .ok_or(ResourcePlanningError::Overflow(
                    "acoustic response partition workspace",
                ))?;
            remaining -= count;
        }
    }
    Ok((partitions, ByteCount(workspace)))
}

pub fn build_acoustic_response_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    result: &AcousticResponseSolverResult,
    source_validation_workspace_bytes: ByteCount,
    facts: PreparedPhaseResourceFacts,
    local_execution: super::LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let (partitions, construction_workspace_bytes) =
        acoustic_response_field_recovery_shape(request, result)?;
    let bytes = partitions.iter().try_fold(0_u64, |sum, n| {
        sum.checked_add(*n).ok_or(ResourcePlanningError::Overflow(
            "acoustic response recovery output",
        ))
    })?;
    if facts.output.encoded_payload_bytes_upper_bound.0 != bytes {
        return Err(ResourcePlanningError::Invalid(
            "acoustic response field output profile differs".into(),
        ));
    }
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::AcousticFieldRecovery(request.identity()),
        demand: FeaResourceDemand::AcousticFieldRecovery(AcousticFieldRecoveryResourceDemand {
            request: request.identity(),
            source_result: request.source.solver_result().ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "acoustic response recovery requires original solver result".into(),
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
