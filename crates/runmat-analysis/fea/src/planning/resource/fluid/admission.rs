use super::{PreparedPhaseResourceFacts, ResourcePlanningError};

pub(super) fn add(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_add(b)
        .ok_or(ResourcePlanningError::Overflow("fluid resource byte sum"))
}

pub(super) fn phase_resident_bytes(
    numerical: u64,
    preparation: u64,
    decoded_inputs: u64,
    result_encoded: u64,
    result_materialization: u64,
) -> Result<u64, ResourcePlanningError> {
    // Two buffers conservatively cover canonical result encoding plus transport
    // staging. The scheduler independently accounts its output-set/object closure.
    let output = add(add(result_encoded, result_encoded)?, result_materialization)?;
    add(add(numerical, preparation)?, add(decoded_inputs, output)?)
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
