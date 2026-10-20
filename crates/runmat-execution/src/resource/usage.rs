use serde::{Deserialize, Serialize};

use super::{AcceleratorClass, AcceleratorDeviceId, AcceleratorProviderId};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceUsage {
    pub cpu_core_millis: u64,
    #[serde(with = "wide_count")]
    pub memory_byte_millis: u128,
    pub accelerator_millis: Vec<AcceleratorUsage>,
    pub wall_millis: u64,
    #[serde(with = "wide_count")]
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

/// Byte-millisecond counts are u128, but serde's buffered deserializer (used by
/// internally tagged enums such as attempt results) rejects 128-bit integers.
/// Counts that fit in u64 keep their plain integer encoding; larger counts are
/// written as decimal strings. Both forms are read on every path.
mod wide_count {
    use serde::{de, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(value: &u128, serializer: S) -> Result<S::Ok, S::Error> {
        match u64::try_from(*value) {
            Ok(narrow) => serializer.serialize_u64(narrow),
            Err(_) => serializer.serialize_str(&value.to_string()),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<u128, D::Error> {
        struct Count;
        impl de::Visitor<'_> for Count {
            type Value = u128;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a non-negative integer count")
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<u128, E> {
                Ok(u128::from(value))
            }
            fn visit_u128<E: de::Error>(self, value: u128) -> Result<u128, E> {
                Ok(value)
            }
            fn visit_i64<E: de::Error>(self, value: i64) -> Result<u128, E> {
                u128::try_from(value).map_err(|_| E::custom("count must be non-negative"))
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<u128, E> {
                if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err(E::custom("count string must be decimal digits"));
                }
                value.parse().map_err(E::custom)
            }
        }
        deserializer.deserialize_any(Count)
    }
}
