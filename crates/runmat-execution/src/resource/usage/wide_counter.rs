//! Exact counters across JSON, buffered tagged enums, and binary serializers.
//! JSON integers within u64 stay integers; wider counters use decimal text.
use serde::{de, Deserializer, Serializer};
use std::fmt;

pub(in crate::resource) fn serialize<S: Serializer>(
    value: &u128,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    if !serializer.is_human_readable() {
        return serializer.serialize_u128(*value);
    }
    match u64::try_from(*value) {
        Ok(value) => serializer.serialize_u64(value),
        Err(_) => serializer.collect_str(value),
    }
}

pub(in crate::resource) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<u128, D::Error> {
    struct Counter;
    impl de::Visitor<'_> for Counter {
        type Value = u128;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an unsigned integer or exact unsigned decimal counter")
        }
        fn visit_u64<E: de::Error>(self, value: u64) -> Result<u128, E> {
            Ok(u128::from(value))
        }
        fn visit_u128<E: de::Error>(self, value: u128) -> Result<u128, E> {
            Ok(value)
        }
        fn visit_str<E: de::Error>(self, value: &str) -> Result<u128, E> {
            if value.is_empty()
                || !value.bytes().all(|byte| byte.is_ascii_digit())
                || (value.len() > 1 && value.starts_with('0'))
            {
                return Err(E::custom(
                    "counter must contain canonical unsigned decimal digits",
                ));
            }
            value.parse().map_err(E::custom)
        }
    }
    // Serde's buffered tagged-enum deserializer does not implement
    // deserialize_u128, even when the original JSON number fits in u64.
    if deserializer.is_human_readable() {
        deserializer.deserialize_any(Counter)
    } else {
        deserializer.deserialize_u128(Counter)
    }
}

#[cfg(test)]
mod tests {
    use crate::resource::ResourceUsage;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "kind")]
    enum Result {
        Metered { usage: ResourceUsage },
    }

    #[test]
    fn tagged_results_and_value_trees_preserve_full_counter_range() {
        for count in [0, u128::from(u64::MAX), u128::from(u64::MAX) + 1, u128::MAX] {
            let result = Result::Metered {
                usage: ResourceUsage {
                    memory_byte_millis: count,
                    retained_byte_millis: count,
                    ..ResourceUsage::default()
                },
            };
            let tree = serde_json::to_value(&result).unwrap();
            assert_eq!(
                tree["usage"]["memory_byte_millis"].is_string(),
                count > u128::from(u64::MAX)
            );
            assert_eq!(serde_json::from_value::<Result>(tree).unwrap(), result);
            let bytes = serde_json::to_vec(&result).unwrap();
            assert_eq!(serde_json::from_slice::<Result>(&bytes).unwrap(), result);
        }
        for invalid in [
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("-1"),
            serde_json::json!("01"),
            serde_json::json!("340282366920938463463374607431768211456"),
        ] {
            let mut usage = serde_json::to_value(ResourceUsage::default()).unwrap();
            usage["memory_byte_millis"] = invalid;
            assert!(serde_json::from_value::<ResourceUsage>(usage).is_err());
        }
    }
}
