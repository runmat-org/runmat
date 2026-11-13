//! Lossless single-precision floats in human-readable artifacts.
//!
//! Finite values keep ordinary JSON numbers. Nonfinite values use an exact
//! IEEE-754 bit tag; binary serializers retain native f32 encoding.
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

pub fn serialize<S: Serializer>(value: &f32, serializer: S) -> Result<S::Ok, S::Error> {
    if value.is_finite() || !serializer.is_human_readable() {
        return serializer.serialize_f32(*value);
    }
    let mut map = serializer.serialize_map(Some(1))?;
    map.serialize_entry("f32_bits", &format!("{:08x}", value.to_bits()))?;
    map.end()
}

pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    if !deserializer.is_human_readable() {
        return f32::deserialize(deserializer);
    }
    deserializer.deserialize_any(FloatVisitor)
}

struct FloatVisitor;

impl<'de> Visitor<'de> for FloatVisitor {
    type Value = f32;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(
            "a finite number or a single f32_bits tag containing nonfinite IEEE-754 bits",
        )
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<f32, E> {
        let narrowed = value as f32;
        if value.is_finite() && narrowed.is_finite() {
            Ok(narrowed)
        } else {
            Err(E::custom(
                "nonfinite human-readable floats require f32_bits",
            ))
        }
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<f32, E> {
        self.visit_f64(value as f64)
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<f32, E> {
        self.visit_f64(value as f64)
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<f32, M::Error> {
        let key = map
            .next_key::<String>()?
            .ok_or_else(|| de::Error::missing_field("f32_bits"))?;
        if key != "f32_bits" {
            return Err(de::Error::unknown_field(&key, &["f32_bits"]));
        }
        let encoded = map.next_value::<String>()?;
        if let Some(extra) = map.next_key::<String>()? {
            return Err(if extra == "f32_bits" {
                de::Error::duplicate_field("f32_bits")
            } else {
                de::Error::unknown_field(&extra, &["f32_bits"])
            });
        }
        if encoded.len() != 8
            || !encoded
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(de::Error::custom(
                "f32_bits must contain exactly 8 lowercase hexadecimal digits",
            ));
        }
        let bits = u32::from_str_radix(&encoded, 16).map_err(de::Error::custom)?;
        let value = f32::from_bits(bits);
        if value.is_finite() {
            return Err(de::Error::custom(
                "finite floats must use ordinary numbers, not f32_bits",
            ));
        }
        Ok(value)
    }
}

/// Lossless vectors for scene and artifact fields without changing finite JSON arrays.
pub mod vec {
    use super::*;
    use serde::ser::SerializeSeq;

    pub fn serialize<S: Serializer>(values: &[f32], serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(values.len()))?;
        for value in values {
            sequence.serialize_element(&FloatRef(value))?;
        }
        sequence.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f32>, D::Error> {
        deserializer.deserialize_seq(VecVisitor)
    }

    struct FloatRef<'a>(&'a f32);

    impl Serialize for FloatRef<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            super::serialize(self.0, serializer)
        }
    }

    struct FloatValue(f32);

    impl<'de> Deserialize<'de> for FloatValue {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            super::deserialize(deserializer).map(Self)
        }
    }

    struct VecVisitor;

    impl<'de> Visitor<'de> for VecVisitor {
        type Value = Vec<f32>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence of lossless f32 values")
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

#[cfg(test)]
mod tests;
