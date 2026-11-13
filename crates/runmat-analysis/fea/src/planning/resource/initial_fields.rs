//! Shared immutable field-input residency projection, independent of consuming physics.
use super::{ByteCount, ResourcePlanningError};
use crate::execution::InitialFieldBinding;
use runmat_canonical_codec::CanonicalLimits;

/// Source closures coexist at whole-study preparation. Honor every authored step ceiling.
pub fn initial_field_source_ceiling(
    sequence: &runmat_analysis_core::AnalysisSequence,
) -> Result<u64, ResourcePlanningError> {
    sequence
        .steps
        .iter()
        .map(|step| step.accuracy.resources.resident_byte_budget())
        .min()
        .ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "initial fields require an authored procedure sequence".into(),
            )
        })
}

pub fn require_initial_field_ceiling(
    fields: &[InitialFieldBinding],
    limits: CanonicalLimits,
    maximum: u64,
) -> Result<(), ResourcePlanningError> {
    let required = initial_field_reservation_bytes(fields, limits)?.0;
    require_ceiling(required, maximum)
}

/// Admit selected sources before cloning their numerical payloads into a workload.
pub(crate) fn require_borrowed_initial_field_ceiling(
    fields: &[&InitialFieldBinding],
    limits: CanonicalLimits,
    maximum: u64,
) -> Result<(), ResourcePlanningError> {
    let required = reservation(fields, fields.is_empty(), limits)?.0;
    require_ceiling(required, maximum)
}

fn require_ceiling(required: u64, maximum: u64) -> Result<(), ResourcePlanningError> {
    if required > maximum {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "initial-field residency",
            required,
            maximum,
        });
    }
    Ok(())
}

pub fn initial_field_reservation_bytes(
    fields: &[InitialFieldBinding],
    limits: CanonicalLimits,
) -> Result<ByteCount, ResourcePlanningError> {
    reservation(fields, fields.is_empty(), limits)
}

fn reservation<T: serde::Serialize + ?Sized>(
    fields: &T,
    empty: bool,
    limits: CanonicalLimits,
) -> Result<ByteCount, ResourcePlanningError> {
    if empty {
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
