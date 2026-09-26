//! Lossless floating literals in human-readable artifact metadata.
//!
//! Finite numbers keep Serde's existing numeric representation. Nonfinite
//! numbers use one canonical IEEE-754 bit tag, avoiding JSON's lossy `null`
//! encoding and JavaScript's inability to represent all 64-bit integers.
//! Binary formats retain their native f64 encoding.
use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serializer};
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
