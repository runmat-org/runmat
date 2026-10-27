//! Reservation projections retain requested constraints, never invented devices.
use super::{interval::scalar_interval_charges, AcceleratorRequest, ResourceRequest};
use crate::ContractError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestedAcceleratorUsage {
    pub requirement: AcceleratorRequest,
    /// Requested device count multiplied by the maximum admitted duration.
    pub device_millis: u64,
}

/// Upper reservation projection, not measured execution or a device lease.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestedResourceUsage {
    pub cpu_core_millis: u64,
    #[serde(with = "super::usage::wide_counter")]
    pub memory_byte_millis: u128,
    pub accelerator_millis: Vec<RequestedAcceleratorUsage>,
    pub wall_millis: u64,
    #[serde(with = "super::usage::wide_counter")]
    pub retained_byte_millis: u128,
    pub egress_bytes: u64,
    pub relay_bytes: u64,
}

impl RequestedResourceUsage {
    /// Round each interval's CPU charge before multiplying. Accelerator
    /// requirements remain unassigned; only the scheduler can issue leases.
    pub fn for_maximum_intervals(
        request: &ResourceRequest,
        intervals: u16,
    ) -> Result<Self, ContractError> {
        request.validate()?;
        if intervals == 0 {
            return Err(ContractError::invalid(
                "resource reservation",
                "interval count must be non-zero",
            ));
        }
        let overflow =
            || ContractError::invalid("resource reservation", "cumulative reservation overflowed");
        let times = |value: u64| value.checked_mul(u64::from(intervals)).ok_or_else(overflow);
        let times128 = |value: u128| {
            value
                .checked_mul(u128::from(intervals))
                .ok_or_else(overflow)
        };
        let scalar =
            scalar_interval_charges(request, request.max_wall_millis, request.max_artifact_bytes)?;
        let accelerator_millis = request
            .accelerators
            .iter()
            .map(|requirement| {
                let device_millis = request
                    .max_wall_millis
                    .checked_mul(u64::from(requirement.count))
                    .ok_or_else(overflow)?;
                Ok(RequestedAcceleratorUsage {
                    requirement: requirement.clone(),
                    device_millis: times(device_millis)?,
                })
            })
            .collect::<Result<_, ContractError>>()?;
        Ok(Self {
            cpu_core_millis: times(scalar.cpu_core_millis)?,
            memory_byte_millis: times128(scalar.memory_byte_millis)?,
            accelerator_millis,
            wall_millis: times(request.max_wall_millis)?,
            retained_byte_millis: times128(scalar.retained_byte_millis)?,
            egress_bytes: times(request.max_egress_bytes)?,
            relay_bytes: times(request.max_relay_bytes)?,
        })
    }
}
