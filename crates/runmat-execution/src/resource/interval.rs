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
        // Millicore-milliseconds can exceed u64 for a long wall time (an unbounded meshing
        // wall ceiling is the largest wire integer) while the core-milliseconds they round up
        // to still fit; the product is formed in u128 and only the rounded result must fit.
        cpu_core_millis: u64::try_from(
            u128::from(allocation.cpu_millicores)
                .checked_mul(u128::from(wall_millis))
                .ok_or_else(|| overflow("CPU time overflowed"))?
                .div_ceil(1_000),
        )
        .map_err(|_| overflow("CPU time overflowed"))?,
        memory_byte_millis: u128::from(allocation.memory_bytes)
            .checked_mul(u128::from(wall_millis))
            .ok_or_else(|| overflow("memory time overflowed"))?,
        retained_byte_millis: u128::from(artifact_bytes)
            .checked_mul(u128::from(wall_millis))
            .ok_or_else(|| overflow("artifact time overflowed"))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(cpu_millicores: u32) -> ResourceRequest {
        ResourceRequest {
            cpu_millicores,
            memory_bytes: 23_173_607_424,
            scratch_bytes: 23_173_607_424,
            max_wall_millis: 1,
            max_artifact_bytes: 1,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: Vec::new(),
            required_capabilities: Default::default(),
        }
    }

    #[test]
    fn cpu_time_of_an_unbounded_wall_rounds_before_it_must_fit() {
        // The largest wire integer: the wall ceiling of a run admitted no wall-time budget.
        let wall = (1_u64 << 53) - 1;
        let charges = scalar_interval_charges(&request(14_000), wall, 1).unwrap();
        assert_eq!(
            u128::from(charges.cpu_core_millis),
            (14_000_u128 * u128::from(wall)).div_ceil(1_000)
        );
        assert_eq!(charges.cpu_core_millis, 126_100_789_566_373_874);
        // Rounding up is unchanged at small walls.
        assert_eq!(
            scalar_interval_charges(&request(1_500), 3, 0)
                .unwrap()
                .cpu_core_millis,
            5
        );
        // A rounded result beyond u64 is still refused, not wrapped or saturated.
        assert!(scalar_interval_charges(&request(u32::MAX), u64::MAX, 0).is_err());
    }
}
