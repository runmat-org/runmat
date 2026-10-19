//! Flat borrowed dictionaries have explicitly bounded vector/sort workspace.
use super::*;
use runmat_canonical_codec::CanonicalCodecControl;
use std::cmp::Ordering;

pub(super) fn checkpoint(
    control: &dyn CanonicalCodecControl,
) -> Result<(), DelaunayVolumeMeshError> {
    if control.is_cancelled() {
        return Err(error(
            DelaunayVolumeMeshStage::Quality,
            DelaunayVolumeMeshErrorKind::Cancelled,
            "volume dictionary preparation cancelled",
        ));
    }
    Ok(())
}
pub(super) fn unique<T: Copy>(
    values: &mut Vec<T>,
    compare: impl Fn(&T, &T) -> Ordering,
    control: &dyn CanonicalCodecControl,
) -> Result<(), DelaunayVolumeMeshError> {
    checkpoint(control)?;
    let mut scratch = values.clone();
    runmat_cpu_execution::sort_copy_with_checkpoint(values, &mut scratch, &compare, || {
        checkpoint(control)
    })?;
    drop(scratch);
    let mut retained = 0;
    for index in 0..values.len() {
        if index.is_multiple_of(128) {
            checkpoint(control)?;
        }
        let value = values[index];
        if retained == 0 || compare(&values[retained - 1], &value) != Ordering::Equal {
            values[retained] = value;
            retained += 1;
        }
    }
    values.truncate(retained);
    Ok(())
}
pub(in crate::cdt::volume_mesh) fn overflow() -> DelaunayVolumeMeshError {
    error(
        DelaunayVolumeMeshStage::Quality,
        DelaunayVolumeMeshErrorKind::ResourceLimit,
        "volume dictionary workspace overflow",
    )
}
