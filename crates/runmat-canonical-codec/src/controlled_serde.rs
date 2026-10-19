//! Cooperative adapters around Serde's existing value representation, without an intermediate wire buffer.
#[cfg(test)]
mod deserialize;
mod serialize;

use crate::CanonicalCodecControl;
use serde::{Serialize, Serializer};

pub(crate) fn to_value<T: Serialize>(
    value: &T,
    control: &dyn CanonicalCodecControl,
) -> Result<serde_json::Value, serde_json::Error> {
    serde_json::to_value(TrackedValue { value, control })
}

pub(super) struct TrackedValue<'a, T: ?Sized> {
    value: &'a T,
    control: &'a dyn CanonicalCodecControl,
}
impl<T: Serialize + ?Sized> Serialize for TrackedValue<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        crate::checkpoint(self.control).map_err(serde::ser::Error::custom)?;
        self.value.serialize(serialize::TrackedSerializer {
            inner: serializer,
            control: self.control,
        })
    }
}

#[cfg(test)]
pub(crate) fn from_value<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
    control: &dyn CanonicalCodecControl,
) -> Result<T, serde_json::Error> {
    T::deserialize(deserialize::TrackedDeserializer {
        inner: value,
        control,
    })
}

#[cfg(test)]
mod tests;

/// Borrow the original value while probing every recursive Serde boundary.
pub(crate) fn tracked<'a, T: Serialize + ?Sized>(
    value: &'a T,
    control: &'a dyn CanonicalCodecControl,
) -> impl Serialize + 'a {
    TrackedValue { value, control }
}
