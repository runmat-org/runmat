use serde::{Deserialize, Serialize};

use super::{AcceleratorClass, AcceleratorDeviceId, AcceleratorProviderId};

pub(super) mod wide_counter;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceUsage {
    pub cpu_core_millis: u64,
    #[serde(with = "wide_counter")]
    pub memory_byte_millis: u128,
    pub accelerator_millis: Vec<AcceleratorUsage>,
    pub wall_millis: u64,
    #[serde(with = "wide_counter")]
    pub retained_byte_millis: u128,
    pub egress_bytes: u64,
    pub relay_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorUsage {
    pub class: AcceleratorClass,
    pub provider: AcceleratorProviderId,
    pub device_id: AcceleratorDeviceId,
    pub device_millis: u64,
}
