//! Integers on the FEA wire contracts.
//!
//! The FEA contracts cross a JSON boundary into JavaScript, which represents
//! integers exactly only up to 2^53 - 1. Every 64-bit or pointer-width integer
//! field on those contracts therefore carries this adapter, which rejects values
//! above [`MAX_WIRE_INTEGER`] in magnitude on both encode and decode. Narrower
//! integer types are already exact and need no adapter. Lower bounds stay with
//! each field's own type and validation (unsigned fields reject negatives at
//! decode; nonzero requirements stay where they are validated today).
//!
//! The TypeScript contract generator requires this adapter on every such field,
//! so the Rust attribute is the single source of the bound for both languages.
use serde::de::{DeserializeOwned, Error as _};
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::Display;

/// Largest integer magnitude JavaScript represents exactly (`Number.MAX_SAFE_INTEGER`).
pub const MAX_WIRE_INTEGER: u64 = (1 << 53) - 1;

/// An integer type that may appear on an FEA wire contract.
pub trait WireInteger: Copy + Display + Serialize + DeserializeOwned {
    fn within_wire_range(self) -> bool;
}

impl WireInteger for u64 {
    fn within_wire_range(self) -> bool {
        self <= MAX_WIRE_INTEGER
    }
}

impl WireInteger for usize {
    fn within_wire_range(self) -> bool {
        u64::try_from(self).is_ok_and(WireInteger::within_wire_range)
    }
}

impl WireInteger for i64 {
    fn within_wire_range(self) -> bool {
        self.unsigned_abs() <= MAX_WIRE_INTEGER
    }
}

impl WireInteger for isize {
    fn within_wire_range(self) -> bool {
        i64::try_from(self).is_ok_and(WireInteger::within_wire_range)
    }
}

/// The value when it lies within the shared wire range. Authoring paths use this
/// so they reject exactly what decoding rejects.
pub fn bounded<T: WireInteger>(value: T) -> Option<T> {
    value.within_wire_range().then_some(value)
}

/// Wire range for a value already held as a finite whole `f64` (authoring literals).
pub fn bounded_whole_f64(value: f64) -> Option<u64> {
    (value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= MAX_WIRE_INTEGER as f64)
        .then_some(value as u64)
}

fn out_of_range(value: impl Display) -> String {
    format!("integer {value} exceeds the FEA wire-contract bound of {MAX_WIRE_INTEGER} (2^53 - 1)")
}

fn check<T: WireInteger>(value: T) -> Result<T, String> {
    bounded(value).ok_or_else(|| out_of_range(value))
}

pub fn serialize<T: WireInteger, S: Serializer>(
    value: &T,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    check(*value)
        .map_err(S::Error::custom)?
        .serialize(serializer)
}

pub fn deserialize<'de, T: WireInteger, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    check(T::deserialize(deserializer)?).map_err(D::Error::custom)
}

/// `Option<T>` fields: `None` is `null`. Because the field has a deserializer
/// adapter, an omitted field is rejected unless the field also declares
/// `#[serde(default)]`; each field keeps its own presence rule that way (an
/// optional field adds `default`, a field whose omission must fail does not).
pub mod option {
    use super::*;

    pub fn serialize<T: WireInteger, S: Serializer>(
        value: &Option<T>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        if let Some(value) = value {
            check(*value).map_err(S::Error::custom)?;
        }
        value.serialize(serializer)
    }

    pub fn deserialize<'de, T: WireInteger, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<T>, D::Error> {
        Option::<T>::deserialize(deserializer)?
            .map(check)
            .transpose()
            .map_err(D::Error::custom)
    }
}

/// `Vec<T>` fields.
pub mod vec {
    use super::*;

    pub fn serialize<T: WireInteger, S: Serializer>(
        values: &[T],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        if let Some(value) = values.iter().find(|value| !value.within_wire_range()) {
            return Err(S::Error::custom(out_of_range(value)));
        }
        values.serialize(serializer)
    }

    pub fn deserialize<'de, T: WireInteger, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<T>, D::Error> {
        let values = Vec::<T>::deserialize(deserializer)?;
        match values.iter().find(|value| !value.within_wire_range()) {
            Some(value) => Err(D::Error::custom(out_of_range(value))),
            None => Ok(values),
        }
    }
}

#[cfg(test)]
mod tests;
