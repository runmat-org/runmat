//! Shared immutable field-input residency projection, independent of consuming physics.
use super::{ByteCount, ResourcePlanningError};
use crate::execution::InitialFieldBinding;
use runmat_canonical_codec::CanonicalLimits;

pub(crate) fn require_initial_field_ceiling(
    fields: &[InitialFieldBinding],
    limits: CanonicalLimits,
    maximum: u64,
) -> Result<(), ResourcePlanningError> {
    let required = initial_field_reservation_bytes(fields, limits)?.0;
    if required > maximum {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "initial-field residency",
            required,
            maximum,
        });
    }
    Ok(())
}

pub(crate) fn initial_field_reservation_bytes(
    fields: &[InitialFieldBinding],
    limits: CanonicalLimits,
) -> Result<ByteCount, ResourcePlanningError> {
    if fields.is_empty() {
        return Ok(ByteCount(0));
    }
    let bytes =
        runmat_canonical_codec::serialized_materialization_resident_bytes_estimate(fields, limits)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    // Source closure, owned preparation and concurrent content/identity scratch.
    // This is an analytical estimate, not measured whole-process memory evidence.
    bytes
        .checked_mul(3)
        .map(ByteCount)
        .ok_or(ResourcePlanningError::Overflow("initial-field residency"))
}
