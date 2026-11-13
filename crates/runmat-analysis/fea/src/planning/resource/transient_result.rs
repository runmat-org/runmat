use crate::execution::{StructuralTransientResultAssemblyRequest, StructuralTransientSegment};

use super::{
    build_resource_plan, post_process, ByteCount, Count, FeaResourceDemand, FeaResourcePhase,
    FeaResourcePlan, FeaResourcePlanningInput, PreparedInputResourceFacts,
    PreparedOutputResourceFacts, PreparedOutputSetResourceFacts, ResourcePlanningError,
    StructuralTransientResultAssemblyResourceDemand, RESOURCE_ESTIMATION_MODEL_REVISION,
};

pub fn structural_transient_result_payload_bytes_upper_bound(
    inputs: &PreparedInputResourceFacts,
    sample_count: u64,
) -> Result<u64, ResourcePlanningError> {
    let records = sample_count
        .checked_add(1)
        .ok_or(ResourcePlanningError::Overflow(
            "transient result record count",
        ))?;
    post_process::derived_output_payload_bytes_upper_bound(inputs, records, 0)
}

pub fn structural_transient_receipt_payload_bytes_upper_bound(
    inputs: &PreparedInputResourceFacts,
    segment_count: u64,
) -> Result<u64, ResourcePlanningError> {
    let records = segment_count
        .checked_add(1)
        .ok_or(ResourcePlanningError::Overflow(
            "transient receipt record count",
        ))?;
    post_process::derived_output_payload_bytes_upper_bound(inputs, records, 0)
}

/// Admits the exact segment fan-in before the worker decodes or assembles physical samples.
pub fn build_structural_transient_result_assembly_resource_plan(
    request: &StructuralTransientResultAssemblyRequest,
    segments: &[StructuralTransientSegment],
    inputs: PreparedInputResourceFacts,
    result_output: PreparedOutputResourceFacts,
    receipt_output: PreparedOutputResourceFacts,
    output_set: PreparedOutputSetResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    request
        .validate_for(segments)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let segment_count = u64::try_from(segments.len())
        .map_err(|_| ResourcePlanningError::Overflow("transient segment count"))?;
    let sample_count = u64::try_from(request.scope.schedule.points.len())
        .map_err(|_| ResourcePlanningError::Overflow("transient sample count"))?;
    let physical_dof_count = u64::try_from(request.scope.dof_count)
        .map_err(|_| ResourcePlanningError::Overflow("transient physical DOF count"))?;
    let result_bytes =
        structural_transient_result_payload_bytes_upper_bound(&inputs, sample_count)?;
    let receipt_bytes =
        structural_transient_receipt_payload_bytes_upper_bound(&inputs, segment_count)?;
    post_process::validate_projected_output(&result_output, result_bytes, "transient result")?;
    post_process::validate_projected_output(&receipt_output, receipt_bytes, "transient receipt")?;
    // Fan-in temporarily owns both unordered and ordered segment vectors. Reserve another
    // complete decoded-input budget for their container/metadata overhead. This is canonical
    // across driver and worker targets; host `size_of` would not be portable plan authority.
    let ordering_workspace_bytes = inputs.decoded_resident_bytes_admission_budget.0;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralTransientResultAssembly(request.identity),
        demand: FeaResourceDemand::StructuralTransientResultAssembly(
            StructuralTransientResultAssemblyResourceDemand {
                scope_digest: request
                    .scope
                    .identity()
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
                    .digest(),
                segment_count: Count(segment_count),
                sample_count: Count(sample_count),
                physical_dof_count: Count(physical_dof_count),
                inputs,
                ordering_workspace_bytes_upper_bound: ByteCount(ordering_workspace_bytes),
                result_output,
                receipt_output,
            },
        ),
        output_set: Some(output_set),
        worker: post_process::post_process_worker(),
        accuracy_ceiling: post_process::post_process_ceiling(),
    })
}
