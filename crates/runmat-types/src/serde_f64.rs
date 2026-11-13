//! Lossless floating literals in human-readable artifact metadata.
//!
//! Finite numbers keep Serde's existing numeric representation. Nonfinite
//! numbers use one canonical IEEE-754 bit tag, avoiding JSON's lossy `null`
//! encoding and JavaScript's inability to represent all 64-bit integers.
//! Binary formats retain their native f64 encoding.
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    if value.is_finite() || !serializer.is_human_readable() {
        return serializer.serialize_f64(*value);
    }
    let mut map = serializer.serialize_map(Some(1))?;
    map.serialize_entry("f64_bits", &format!("{:016x}", value.to_bits()))?;
    map.end()
}

pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    if !deserializer.is_human_readable() {
        return f64::deserialize(deserializer);
    }
    deserializer.deserialize_any(FloatVisitor)
}

struct FloatVisitor;

/// Lossless vectors for scene and artifact fields without changing finite JSON arrays.
pub mod vec {
    use super::*;
    use serde::ser::SerializeSeq;

    pub fn serialize<S: Serializer>(values: &[f64], serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(values.len()))?;
        for value in values {
            sequence.serialize_element(&FloatRef(value))?;
        }
        sequence.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f64>, D::Error> {
        deserializer.deserialize_seq(VecVisitor)
    }

    struct FloatRef<'a>(&'a f64);

    impl Serialize for FloatRef<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            super::serialize(self.0, serializer)
        }
    }

    struct FloatValue(f64);

    impl<'de> Deserialize<'de> for FloatValue {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            super::deserialize(deserializer).map(Self)
        }
    }

    struct VecVisitor;

    impl<'de> Visitor<'de> for VecVisitor {
        type Value = Vec<f64>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence of lossless f64 values")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
            let mut values = Vec::new();
            while let Some(FloatValue(value)) = sequence.next_element()? {
                values.push(value);
            }
            Ok(values)
        }
    }
}

impl<'de> Visitor<'de> for FloatVisitor {
    type Value = f64;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(
            "a finite number or a single f64_bits tag containing nonfinite IEEE-754 bits",
        )
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<f64, E> {
        if value.is_finite() {
            Ok(value)
        } else {
            Err(E::custom(
                "nonfinite human-readable floats require f64_bits",
            ))
        }
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<f64, E> {
        Ok(value as f64)
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<f64, E> {
        Ok(value as f64)
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<f64, M::Error> {
        let key = map
            .next_key::<String>()?
            .ok_or_else(|| de::Error::missing_field("f64_bits"))?;
        if key != "f64_bits" {
            return Err(de::Error::unknown_field(&key, &["f64_bits"]));
        }
        let encoded = map.next_value::<String>()?;
        if let Some(extra) = map.next_key::<String>()? {
            return Err(if extra == "f64_bits" {
                de::Error::duplicate_field("f64_bits")
            } else {
                de::Error::unknown_field(&extra, &["f64_bits"])
            });
        }
        if encoded.len() != 16
            || !encoded
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(de::Error::custom(
                "f64_bits must contain exactly 16 lowercase hexadecimal digits",
            ));
        }
        let bits = u64::from_str_radix(&encoded, 16).map_err(de::Error::custom)?;
        let value = f64::from_bits(bits);
        if value.is_finite() {
            return Err(de::Error::custom(
                "finite floats must use ordinary numbers, not f64_bits",
            ));
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests;
