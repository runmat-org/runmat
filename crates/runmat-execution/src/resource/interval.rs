//! Shared scalar arithmetic for requested and actually assigned intervals.
use super::ResourceRequest;
use crate::ContractError;

pub(super) struct ScalarIntervalCharges {
    pub cpu_core_millis: u64,
    pub memory_byte_millis: u128,
    pub retained_byte_millis: u128,
}

pub(super) fn scalar_interval_charges(
    allocation: &ResourceRequest,
    wall_millis: u64,
    artifact_bytes: u64,
) -> Result<ScalarIntervalCharges, ContractError> {
    let overflow = |field| ContractError::invalid("resource accounting", field);
    Ok(ScalarIntervalCharges {
        cpu_core_millis: u64::from(allocation.cpu_millicores)
            .checked_mul(wall_millis)
            .ok_or_else(|| overflow("CPU time overflowed"))?
            .div_ceil(1_000),
        memory_byte_millis: u128::from(allocation.memory_bytes)
            .checked_mul(u128::from(wall_millis))
            .ok_or_else(|| overflow("memory time overflowed"))?,
        retained_byte_millis: u128::from(artifact_bytes)
            .checked_mul(u128::from(wall_millis))
            .ok_or_else(|| overflow("artifact time overflowed"))?,
    })
}
