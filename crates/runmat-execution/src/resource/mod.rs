use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::ContractError;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    ProcessIsolation,
    BrowserWorker,
    NetworkDenied,
    Accelerator(String),
    Custom(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorRequest {
    pub class: String,
    pub count: u16,
    pub memory_bytes_each: u64,
}

impl AcceleratorRequest {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.class.is_empty()
            || self.class.len() > 96
            || !self.class.is_ascii()
            || self.class.chars().any(char::is_control)
            || self.count == 0
        {
            return Err(ContractError::invalid(
                "accelerator request",
                "class must be 1..=96 printable ASCII bytes and count must be non-zero",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub max_wall_millis: u64,
    pub max_artifact_bytes: u64,
    pub max_egress_bytes: u64,
    pub max_relay_bytes: u64,
    pub accelerators: Vec<AcceleratorRequest>,
    pub required_capabilities: BTreeSet<Capability>,
}

impl ResourceRequest {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.cpu_millicores == 0 {
            return Err(ContractError::invalid(
                "cpu_millicores",
                "must be greater than zero",
            ));
        }
        if self.memory_bytes == 0 || self.max_wall_millis == 0 {
            return Err(ContractError::invalid(
                "resource request",
                "memory and maximum duration must be greater than zero",
            ));
        }
        if self.accelerators.len() > 16 {
            return Err(ContractError::Limit {
                field: "accelerators",
                limit: 16,
            });
        }
        for accelerator in &self.accelerators {
            accelerator.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceInventory {
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub accelerators: Vec<AcceleratorRequest>,
    pub capabilities: BTreeSet<Capability>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_core_millis: u64,
    #[serde(with = "u128_decimal")]
    pub memory_byte_millis: u128,
    pub accelerator_millis: Vec<AcceleratorUsage>,
    pub wall_millis: u64,
    #[serde(with = "u128_decimal")]
    pub retained_byte_millis: u128,
    pub egress_bytes: u64,
    pub relay_bytes: u64,
}

/// Placement-neutral accounting for one measured execution interval. CPU, memory, and
/// accelerator time represent the resources reserved for the interval rather than sampled
/// process utilization. Attempt and durable-job identities belong to their orchestration layers.
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
    pub fn for_reserved_interval(
        allocation: ResourceRequest,
        wall_millis: u64,
        artifact_bytes: u64,
        egress_bytes: u64,
        relay_bytes: u64,
    ) -> Result<Self, ContractError> {
        allocation.validate()?;
        let cpu_core_millis = u64::from(allocation.cpu_millicores)
            .checked_mul(wall_millis)
            .ok_or_else(|| ContractError::invalid("resource accounting", "CPU time overflowed"))?
            .div_ceil(1_000);
        let memory_byte_millis = u128::from(allocation.memory_bytes)
            .checked_mul(u128::from(wall_millis))
            .ok_or_else(|| {
                ContractError::invalid("resource accounting", "memory time overflowed")
            })?;
        let retained_byte_millis = u128::from(artifact_bytes)
            .checked_mul(u128::from(wall_millis))
            .ok_or_else(|| {
                ContractError::invalid("resource accounting", "artifact time overflowed")
            })?;
        let accelerator_millis = allocation
            .accelerators
            .iter()
            .map(|accelerator| {
                u64::from(accelerator.count)
                    .checked_mul(wall_millis)
                    .map(|device_millis| AcceleratorUsage {
                        class: accelerator.class.clone(),
                        device_millis,
                    })
                    .ok_or_else(|| {
                        ContractError::invalid("resource accounting", "accelerator time overflowed")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            allocation,
            usage: ResourceUsage {
                cpu_core_millis,
                memory_byte_millis,
                accelerator_millis,
                wall_millis,
                retained_byte_millis,
                egress_bytes,
                relay_bytes,
            },
            artifact_bytes,
        })
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        let expected = Self::for_reserved_interval(
            self.allocation.clone(),
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

impl ResourceAccountingSummary {
    pub fn aggregate(
        intervals: &[ResourceAccounting],
        graph_wall_millis: u64,
    ) -> Result<Self, ContractError> {
        if intervals.is_empty() {
            return Err(ContractError::invalid(
                "resource accounting summary",
                "at least one measured interval is required",
            ));
        }
        let mut summary = Self {
            usage: ResourceUsage {
                wall_millis: graph_wall_millis,
                ..ResourceUsage::default()
            },
            ..Self::default()
        };
        let mut accelerator_millis = BTreeMap::<String, u64>::new();
        for interval in intervals {
            interval.validate()?;
            if graph_wall_millis < interval.usage.wall_millis {
                return Err(ContractError::invalid(
                    "resource accounting summary",
                    "graph wall time is shorter than a measured interval",
                ));
            }
            summary.usage.cpu_core_millis = summary
                .usage
                .cpu_core_millis
                .checked_add(interval.usage.cpu_core_millis)
                .ok_or_else(|| {
                    ContractError::invalid("resource accounting summary", "CPU time overflowed")
                })?;
            summary.usage.memory_byte_millis = summary
                .usage
                .memory_byte_millis
                .checked_add(interval.usage.memory_byte_millis)
                .ok_or_else(|| {
                    ContractError::invalid("resource accounting summary", "memory time overflowed")
                })?;
            summary.usage.retained_byte_millis = summary
                .usage
                .retained_byte_millis
                .checked_add(interval.usage.retained_byte_millis)
                .ok_or_else(|| {
                    ContractError::invalid(
                        "resource accounting summary",
                        "retained time overflowed",
                    )
                })?;
            summary.usage.egress_bytes = summary
                .usage
                .egress_bytes
                .checked_add(interval.usage.egress_bytes)
                .ok_or_else(|| {
                    ContractError::invalid("resource accounting summary", "egress overflowed")
                })?;
            summary.usage.relay_bytes = summary
                .usage
                .relay_bytes
                .checked_add(interval.usage.relay_bytes)
                .ok_or_else(|| {
                    ContractError::invalid("resource accounting summary", "relay overflowed")
                })?;
            for accelerator in &interval.usage.accelerator_millis {
                let total = accelerator_millis
                    .entry(accelerator.class.clone())
                    .or_default();
                *total = total
                    .checked_add(accelerator.device_millis)
                    .ok_or_else(|| {
                        ContractError::invalid(
                            "resource accounting summary",
                            "accelerator time overflowed",
                        )
                    })?;
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
                .ok_or_else(|| {
                    ContractError::invalid(
                        "resource accounting summary",
                        "artifact bytes overflowed",
                    )
                })?;
        }
        summary.usage.accelerator_millis = accelerator_millis
            .into_iter()
            .map(|(class, device_millis)| AcceleratorUsage {
                class,
                device_millis,
            })
            .collect();
        Ok(summary)
    }
}

mod u128_decimal {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &u128, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u128, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        encoded.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorUsage {
    pub class: String,
    pub device_millis: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ResourceRequest {
        ResourceRequest {
            cpu_millicores: 1_500,
            memory_bytes: 2_048,
            scratch_bytes: 1_024,
            max_wall_millis: 1_000,
            max_artifact_bytes: 4_096,
            max_egress_bytes: 512,
            max_relay_bytes: 256,
            accelerators: vec![AcceleratorRequest {
                class: "gpu.test".into(),
                count: 2,
                memory_bytes_each: 8_192,
            }],
            required_capabilities: BTreeSet::new(),
        }
    }

    #[test]
    fn reserved_interval_accounting_is_exact_and_self_validating() {
        let accounting =
            ResourceAccounting::for_reserved_interval(request(), 20, 300, 40, 10).unwrap();
        assert_eq!(accounting.usage.cpu_core_millis, 30);
        assert_eq!(accounting.usage.memory_byte_millis, 40_960);
        assert_eq!(accounting.usage.retained_byte_millis, 6_000);
        assert_eq!(accounting.usage.accelerator_millis[0].device_millis, 40);
        accounting.validate().unwrap();

        let mut substituted = accounting;
        substituted.usage.cpu_core_millis += 1;
        assert!(substituted.validate().is_err());
    }

    #[test]
    fn accelerator_classes_are_portable_printable_identifiers() {
        for class in ["", "gpu\nclass", "gpu-é"] {
            let mut invalid = request();
            invalid.accelerators[0].class = class.into();
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn reserved_interval_accounting_rejects_invalid_allocations() {
        let mut invalid = request();
        invalid.cpu_millicores = 0;
        assert!(ResourceAccounting::for_reserved_interval(invalid, 1, 0, 0, 0).is_err());
    }

    #[test]
    fn accounting_summary_aggregates_heterogeneous_intervals_canonically() {
        let first = ResourceAccounting::for_reserved_interval(request(), 20, 300, 40, 10).unwrap();
        let mut smaller = request();
        smaller.cpu_millicores = 500;
        smaller.memory_bytes = 1_024;
        smaller.scratch_bytes = 512;
        smaller.accelerators.clear();
        let second = ResourceAccounting::for_reserved_interval(smaller, 10, 100, 20, 5).unwrap();
        let summary = ResourceAccountingSummary::aggregate(&[first, second], 25).unwrap();
        assert_eq!(summary.usage.cpu_core_millis, 35);
        assert_eq!(summary.usage.memory_byte_millis, 51_200);
        assert_eq!(summary.usage.wall_millis, 25);
        assert_eq!(summary.usage.egress_bytes, 60);
        assert_eq!(summary.usage.relay_bytes, 15);
        assert_eq!(summary.peak_cpu_millicores, 1_500);
        assert_eq!(summary.peak_memory_bytes, 2_048);
        assert_eq!(summary.peak_scratch_bytes, 1_024);
        assert_eq!(summary.artifact_bytes, 400);
        assert_eq!(summary.usage.accelerator_millis[0].class, "gpu.test");
        assert_eq!(summary.usage.accelerator_millis[0].device_millis, 40);
    }

    #[test]
    fn accounting_summary_rejects_empty_or_impossible_graph_intervals() {
        assert!(ResourceAccountingSummary::aggregate(&[], 0).is_err());
        let interval = ResourceAccounting::for_reserved_interval(request(), 20, 0, 0, 0).unwrap();
        assert!(ResourceAccountingSummary::aggregate(&[interval], 19).is_err());
    }
}
