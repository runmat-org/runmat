use std::collections::BTreeSet;

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
            if accelerator.class.is_empty()
                || accelerator.class.len() > 96
                || accelerator.count == 0
            {
                return Err(ContractError::invalid(
                    "accelerator request",
                    "class must be 1..=96 bytes and count must be non-zero",
                ));
            }
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
    pub memory_byte_millis: u128,
    pub accelerator_millis: Vec<AcceleratorUsage>,
    pub wall_millis: u64,
    pub retained_byte_millis: u128,
    pub egress_bytes: u64,
    pub relay_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorUsage {
    pub class: String,
    pub device_millis: u64,
}
