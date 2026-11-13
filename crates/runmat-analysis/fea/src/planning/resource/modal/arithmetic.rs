//! Checked scalar resource arithmetic shared by modal sizing and planning.
use super::ResourcePlanningError;

pub(super) fn count(value: usize, label: &'static str) -> Result<u64, ResourcePlanningError> {
    u64::try_from(value).map_err(|_| ResourcePlanningError::Overflow(label))
}

pub(super) fn checked_bytes(
    count: u64,
    width: u64,
    label: &'static str,
) -> Result<u64, ResourcePlanningError> {
    checked_mul(count, width, label)
}

pub(super) fn checked_mul(
    left: u64,
    right: u64,
    label: &'static str,
) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}

pub(super) fn checked_add(
    left: u64,
    right: u64,
    label: &'static str,
) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}
