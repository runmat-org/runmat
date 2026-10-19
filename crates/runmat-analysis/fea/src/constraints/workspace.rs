//! Domain mapping for shared counted topology storage and canonical stable ordering.
use super::ConstraintPlanError;
use crate::topology::storage::{self, TopologyStorageError};
impl From<TopologyStorageError> for ConstraintPlanError {
    fn from(error: TopologyStorageError) -> Self {
        match error {
            TopologyStorageError::Cancelled => Self::Cancelled,
            TopologyStorageError::Capacity => overflow(),
            TopologyStorageError::Index => Self::Invalid(
                "constraint node index has duplicate keys or an inconsistent count".into(),
            ),
        }
    }
}
pub(super) fn check_cancelled() -> Result<(), ConstraintPlanError> {
    storage::check_cancelled().map_err(Into::into)
}
pub(super) fn reserve<T>(count: usize) -> Result<Vec<T>, ConstraintPlanError> {
    storage::reserve(count).map_err(Into::into)
}
pub(super) fn slots<T>(count: usize) -> Result<u64, ConstraintPlanError> {
    storage::slots::<T>(count).map_err(Into::into)
}
pub(super) fn multiply(a: usize, b: usize) -> Result<usize, ConstraintPlanError> {
    storage::multiply(a, b).map_err(Into::into)
}
pub(super) fn count_add(a: usize, b: usize) -> Result<usize, ConstraintPlanError> {
    storage::count_add(a, b).map_err(Into::into)
}
pub(super) fn add(a: u64, b: u64) -> Result<u64, ConstraintPlanError> {
    storage::add(a, b).map_err(Into::into)
}
pub(super) fn overflow() -> ConstraintPlanError {
    ConstraintPlanError::Invalid("constraint requested storage exceeds addressable slots".into())
}
mod ordering;
pub(super) use ordering::{ordering_workspace_bytes, sort_stable_by};
