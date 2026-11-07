use super::{PreparedPhaseResourceFacts, ResourcePlanningError};

pub(super) fn add(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_add(b)
        .ok_or(ResourcePlanningError::Overflow("fluid resource byte sum"))
}

/// Peak of one steady fluid phase. The retained source preparation and the decoded inputs
/// live throughout. Three stages follow one another and never coexist: the source's
/// finite-volume derivation scratch, the numerical solve (`numerical`), and result encoding,
/// which holds the retained solution (`verification`, the numerical bound without its
/// solver-only Krylov, preconditioner and coupled-block workspace) and the output buffers.
pub(super) fn phase_resident_bytes(
    numerical: u64,
    verification: u64,
    preparation: u64,
    preparation_scratch: u64,
    decoded_inputs: u64,
    result_encoded: u64,
    result_materialization: u64,
) -> Result<u64, ResourcePlanningError> {
    // Two buffers conservatively cover canonical result encoding plus transport
    // staging. The scheduler independently accounts its output-set/object closure.
    let output = add(add(result_encoded, result_encoded)?, result_materialization)?;
    let encoding = add(verification, output)?;
    let staged = preparation_scratch.max(numerical).max(encoding);
    add(add(preparation, decoded_inputs)?, staged)
}

pub(super) fn require_ceiling(required: u64, maximum: u64) -> Result<(), ResourcePlanningError> {
    if required > maximum || required > isize::MAX as u64 {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "steady fluid phase requested heap",
            required,
            maximum: maximum.min(isize::MAX as u64),
        });
    }
    Ok(())
}

pub(super) fn validate_prepared(
    input_bytes: u64,
    decode_bytes: u64,
    result_bytes: u64,
    result_materialization: u64,
    prepared: &PreparedPhaseResourceFacts,
) -> Result<(), ResourcePlanningError> {
    if input_bytes == 0
        || result_bytes == 0
        || prepared.inputs.encoded_payload_bytes.0 != input_bytes
        // Content-addressed repeated chunks can make unique transport smaller than the payload.
        || prepared.inputs.transport_closure_bytes.0 == 0
        || prepared.inputs.decoded_resident_bytes_admission_budget.0 < decode_bytes
        || prepared.output.encoded_payload_bytes_upper_bound.0 != result_bytes
        || prepared.output.transport_closure_bytes_upper_bound.0 < result_bytes
        || prepared
            .output
            .construction_resident_bytes_admission_budget
            .0
            < result_materialization.max(prepared.output.transport_closure_bytes_upper_bound.0)
        || prepared.output_set.encoded_bytes_upper_bound.0 == 0
    {
        return Err(ResourcePlanningError::Invalid(
            "steady fluid original wire or prepared transport facts disagree".into(),
        ));
    }
    Ok(())
}
