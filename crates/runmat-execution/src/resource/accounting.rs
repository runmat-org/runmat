//! Placement-neutral accounting for measured execution intervals.
//!
//! CPU, memory, and accelerator time represent the resources reserved for the
//! interval rather than sampled process utilization. Accelerator time is
//! attributed to each exactly leased device. Attempt and durable-job identities
//! belong to their orchestration layers.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{
    AcceleratorClass, AcceleratorDeviceId, AcceleratorDeviceLease, AcceleratorProviderId,
    AcceleratorUsage, ResourceAssignment, ResourceRequest, ResourceUsage,
};
use crate::ContractError;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceAccounting {
    pub allocation: ResourceRequest,
    pub usage: ResourceUsage,
    pub artifact_bytes: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceAccountingSummary {
    pub usage: ResourceUsage,
    pub peak_cpu_millicores: u32,
    pub peak_memory_bytes: u64,
    pub peak_scratch_bytes: u64,
    pub artifact_bytes: u64,
}

impl ResourceAccounting {
    /// Account a reserved interval without accelerator leases. Allocations that
    /// request accelerators must use [`Self::for_assigned_interval`].
    pub fn for_reserved_interval(
        allocation: ResourceRequest,
        wall_millis: u64,
        artifact_bytes: u64,
        egress_bytes: u64,
        relay_bytes: u64,
    ) -> Result<Self, ContractError> {
        Self::for_assigned_interval(
            allocation,
            &ResourceAssignment::default(),
            wall_millis,
            artifact_bytes,
            egress_bytes,
            relay_bytes,
        )
    }

    /// Account a reserved interval whose accelerators were exactly leased.
    pub fn for_assigned_interval(
        allocation: ResourceRequest,
        assignment: &ResourceAssignment,
        wall_millis: u64,
        artifact_bytes: u64,
        egress_bytes: u64,
        relay_bytes: u64,
    ) -> Result<Self, ContractError> {
        assignment.validate_for_request(&allocation)?;
        let accelerator_millis = device_usage(&assignment.accelerator_leases, wall_millis);
        Self::from_parts(
            allocation,
            accelerator_millis,
            wall_millis,
            artifact_bytes,
            egress_bytes,
            relay_bytes,
        )
    }

    fn from_parts(
        allocation: ResourceRequest,
        mut accelerator_millis: Vec<AcceleratorUsage>,
        wall_millis: u64,
        artifact_bytes: u64,
        egress_bytes: u64,
        relay_bytes: u64,
    ) -> Result<Self, ContractError> {
        allocation.validate()?;
        let scalar =
            super::interval::scalar_interval_charges(&allocation, wall_millis, artifact_bytes)?;
        accelerator_millis.sort_by(|left, right| usage_key(left).cmp(&usage_key(right)));
        Ok(Self {
            allocation,
            usage: ResourceUsage {
                cpu_core_millis: scalar.cpu_core_millis,
                memory_byte_millis: scalar.memory_byte_millis,
                accelerator_millis,
                wall_millis,
                retained_byte_millis: scalar.retained_byte_millis,
                egress_bytes,
                relay_bytes,
            },
            artifact_bytes,
        })
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        let mut per_class = BTreeMap::<&AcceleratorClass, usize>::new();
        for usage in &self.usage.accelerator_millis {
            if usage.device_millis != self.usage.wall_millis {
                return Err(ContractError::invalid(
                    "resource accounting",
                    "leased accelerator time differs from the measured interval",
                ));
            }
            *per_class.entry(&usage.class).or_default() += 1;
        }
        let mut requested = BTreeMap::<&AcceleratorClass, usize>::new();
        for request in &self.allocation.accelerators {
            *requested.entry(&request.class).or_default() += usize::from(request.count);
        }
        if per_class != requested {
            return Err(ContractError::invalid(
                "resource accounting",
                "leased accelerator usage differs from the reserved allocation",
            ));
        }
        let expected = Self::from_parts(
            self.allocation.clone(),
            self.usage.accelerator_millis.clone(),
            self.usage.wall_millis,
            self.artifact_bytes,
            self.usage.egress_bytes,
            self.usage.relay_bytes,
        )?;
        if self != &expected {
            return Err(ContractError::invalid(
                "resource accounting",
                "reserved allocation, usage, and artifact bytes are inconsistent",
            ));
        }
        Ok(())
    }
}

fn device_usage(leases: &[AcceleratorDeviceLease], wall_millis: u64) -> Vec<AcceleratorUsage> {
    leases
        .iter()
        .map(|lease| AcceleratorUsage {
            class: lease.class.clone(),
            provider: lease.provider.id.clone(),
            device_id: lease.device_id.clone(),
            device_millis: wall_millis,
        })
        .collect()
}

type UsageKey<'a> = (
    &'a AcceleratorClass,
    &'a AcceleratorProviderId,
    &'a AcceleratorDeviceId,
);

fn usage_key(usage: &AcceleratorUsage) -> UsageKey<'_> {
    (&usage.class, &usage.provider, &usage.device_id)
}

impl ResourceAccountingSummary {
    pub fn aggregate(
        intervals: &[ResourceAccounting],
        graph_wall_millis: u64,
    ) -> Result<Self, ContractError> {
        let summary_error = |reason| ContractError::invalid("resource accounting summary", reason);
        if intervals.is_empty() {
            return Err(summary_error("at least one measured interval is required"));
        }
        let mut summary = Self {
            usage: ResourceUsage {
                wall_millis: graph_wall_millis,
                ..ResourceUsage::default()
            },
            ..Self::default()
        };
        let mut accelerator_millis = BTreeMap::<UsageKey<'_>, u64>::new();
        for interval in intervals {
            interval.validate()?;
            if graph_wall_millis < interval.usage.wall_millis {
                return Err(ContractError::invalid(
                    "resource accounting summary",
                    format!(
                        "graph wall time is shorter than a measured interval: graph={} ms, interval={} ms",
                        graph_wall_millis, interval.usage.wall_millis,
                    ),
                ));
            }
            let usage = &mut summary.usage;
            usage.cpu_core_millis = usage
                .cpu_core_millis
                .checked_add(interval.usage.cpu_core_millis)
                .ok_or_else(|| summary_error("CPU time overflowed"))?;
            usage.memory_byte_millis = usage
                .memory_byte_millis
                .checked_add(interval.usage.memory_byte_millis)
                .ok_or_else(|| summary_error("memory time overflowed"))?;
            usage.retained_byte_millis = usage
                .retained_byte_millis
                .checked_add(interval.usage.retained_byte_millis)
                .ok_or_else(|| summary_error("retained time overflowed"))?;
            usage.egress_bytes = usage
                .egress_bytes
                .checked_add(interval.usage.egress_bytes)
                .ok_or_else(|| summary_error("egress overflowed"))?;
            usage.relay_bytes = usage
                .relay_bytes
                .checked_add(interval.usage.relay_bytes)
                .ok_or_else(|| summary_error("relay overflowed"))?;
            for accelerator in &interval.usage.accelerator_millis {
                let total = accelerator_millis
                    .entry(usage_key(accelerator))
                    .or_default();
                *total = total
                    .checked_add(accelerator.device_millis)
                    .ok_or_else(|| summary_error("accelerator time overflowed"))?;
            }
            summary.peak_cpu_millicores = summary
                .peak_cpu_millicores
                .max(interval.allocation.cpu_millicores);
            summary.peak_memory_bytes = summary
                .peak_memory_bytes
                .max(interval.allocation.memory_bytes);
            summary.peak_scratch_bytes = summary
                .peak_scratch_bytes
                .max(interval.allocation.scratch_bytes);
            summary.artifact_bytes = summary
                .artifact_bytes
                .checked_add(interval.artifact_bytes)
                .ok_or_else(|| summary_error("artifact bytes overflowed"))?;
        }
        summary.usage.accelerator_millis = accelerator_millis
            .into_iter()
            .map(
                |((class, provider, device_id), device_millis)| AcceleratorUsage {
                    class: class.clone(),
                    provider: provider.clone(),
                    device_id: device_id.clone(),
                    device_millis,
                },
            )
            .collect();
        Ok(summary)
    }
}

#[cfg(test)]
#[path = "accounting/tests.rs"]
mod tests;
