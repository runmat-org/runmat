//! Admit the immutable native face-bound inventory before reconstruction.
use super::{EvaluatorBindings, GeometryEvaluationControl, GeometryEvaluationError};
use runmat_geometry_core::GeometryEvaluationErrorKind;

const NATIVE_FACE_BOUNDS_BYTES: u64 = 40;

pub(super) fn face_keys(
    bindings: &EvaluatorBindings,
    control: Option<&dyn GeometryEvaluationControl>,
) -> Result<Vec<u64>, GeometryEvaluationError> {
    admit_bytes(bindings.surfaces.len(), 8, control)?;
    let mut keys = bindings.surfaces.values().copied().collect::<Vec<_>>();
    keys.sort_unstable();
    keys.dedup();
    admit_bytes(keys.len(), NATIVE_FACE_BOUNDS_BYTES, control)?;
    Ok(keys)
}

fn admit_bytes(
    count: usize,
    width: u64,
    control: Option<&dyn GeometryEvaluationControl>,
) -> Result<(), GeometryEvaluationError> {
    let bytes = u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(width))
        .ok_or_else(|| {
            GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                "OCCT surface inventory storage overflows",
            )
        })?;
    if let Some(control) = control {
        control.consume_allocation_bytes(bytes)?;
    }
    Ok(())
}
