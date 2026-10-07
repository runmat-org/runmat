//! Checked requested Rust slots and fallible reservations shared by constraint construction.
use super::ConstraintPlanError;
use std::mem::size_of;
pub(super) fn check_cancelled() -> Result<(), ConstraintPlanError> {
    if crate::is_fea_cancelled() {
        Err(ConstraintPlanError::Cancelled)
    } else {
        Ok(())
    }
}
pub(super) fn reserve<T>(count: usize) -> Result<Vec<T>, ConstraintPlanError> {
    check_cancelled()?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| overflow())?;
    Ok(values)
}
pub(super) fn slots<T>(count: usize) -> Result<u64, ConstraintPlanError> {
    u64::try_from(multiply(count, size_of::<T>())?).map_err(|_| overflow())
}
pub(super) fn multiply(a: usize, b: usize) -> Result<usize, ConstraintPlanError> {
    a.checked_mul(b).ok_or_else(overflow)
}
pub(super) fn count_add(a: usize, b: usize) -> Result<usize, ConstraintPlanError> {
    a.checked_add(b).ok_or_else(overflow)
}
pub(super) fn add(a: u64, b: u64) -> Result<u64, ConstraintPlanError> {
    a.checked_add(b).ok_or_else(overflow)
}
pub(super) fn overflow() -> ConstraintPlanError {
    ConstraintPlanError::Invalid("constraint requested storage exceeds addressable slots".into())
}
