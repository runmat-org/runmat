use serde::{Deserialize, Serialize};

use super::{AcceleratorClass, AcceleratorDeviceId, AcceleratorProviderId};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceUsage {
    pub cpu_core_millis: u64,
    #[serde(deserialize_with = "deserialize_counter")]
    pub memory_byte_millis: u128,
    pub accelerator_millis: Vec<AcceleratorUsage>,
    pub wall_millis: u64,
    #[serde(deserialize_with = "deserialize_counter")]
    pub retained_byte_millis: u128,
    pub egress_bytes: u64,
    pub relay_bytes: u64,
}

// Arithmetic retains u128 headroom, while canonical human-readable transport
// admits integer counters through u64. Decode that existing numeric wire before
// widening: Serde's buffer for internally tagged result enums cannot deserialize
// u128 directly, even when the original JSON integer fits in u64. Do not accept
// floating-point or string coercions, or broaden the canonical integer range.
fn deserialize_counter<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<u128, D::Error> {
    if deserializer.is_human_readable() {
        u64::deserialize(deserializer).map(u128::from)
    } else {
        u128::deserialize(deserializer)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorUsage {
    pub class: AcceleratorClass,
    pub provider: AcceleratorProviderId,
    pub device_id: AcceleratorDeviceId,
    pub device_millis: u64,
}
