use super::*;

pub(super) fn implicit_resource_plan_input_facts(
) -> Result<PreparedInputResourceFacts, ResourcePlanningError> {
    let decoded = runmat_canonical_codec::projected_canonical_typed_decode_resident_bytes_estimate(
        RESOURCE_PLAN_MAX_ENCODED_BYTES,
        1,
    )
    .ok_or(ResourcePlanningError::Overflow(
        "resource-plan decoded residency budget",
    ))?;
    let transport = add(
        RESOURCE_PLAN_MAX_ENCODED_BYTES,
        crate::execution::ANALYSIS_MANIFEST_MAX_ENCODED_BYTES,
        "resource-plan transport closure budget",
    )?;
    Ok(PreparedInputResourceFacts {
        encoded_payload_bytes: ByteCount(RESOURCE_PLAN_MAX_ENCODED_BYTES),
        transport_closure_bytes: ByteCount(transport),
        decoded_resident_bytes_admission_budget: ByteCount(decoded),
    })
}

/// In-memory original recovery has no resource-plan input closure or decoder.
/// Its enclosing portable task, when present, owns that separate transport demand.
pub(super) fn operator_resource_plan_input_facts(
    phase: &FeaResourcePhase,
) -> Result<PreparedInputResourceFacts, ResourcePlanningError> {
    if *phase == FeaResourcePhase::StructuralNonlinearRecovery {
        return Ok(PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(0),
            transport_closure_bytes: ByteCount(0),
            decoded_resident_bytes_admission_budget: ByteCount(0),
        });
    }
    implicit_resource_plan_input_facts()
}
