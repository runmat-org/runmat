//! Adapters for established canonical numeric wire fields. The enclosing field supplies
//! the physical kind; deserialization validates the scalar instead of accepting non-finite data.
//! This is a compatibility wire adapter, not a second quantity representation.
use super::{Quantity, QuantityMarker};
use serde::{Deserialize, Deserializer, Serializer};
pub fn serialize<K: QuantityMarker, S: Serializer>(
    value: &Quantity<K>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    if !value.canonical_value().is_finite() {
        return Err(serde::ser::Error::custom(super::UnitError::NonFiniteValue));
    }
    serializer.serialize_f64(value.canonical_value())
}
pub fn deserialize<'de, K: QuantityMarker, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Quantity<K>, D::Error> {
    Quantity::try_new(f64::deserialize(deserializer)?).map_err(serde::de::Error::custom)
}

/// Existing optional canonical numeric fields retain number/null on the wire.
pub mod optional {
    use super::*;
    use serde::Serialize;
    pub fn serialize<K: QuantityMarker, S: Serializer>(
        value: &Option<Quantity<K>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        if value.is_some_and(|quantity| !quantity.canonical_value().is_finite()) {
            return Err(serde::ser::Error::custom(
                super::super::UnitError::NonFiniteValue,
            ));
        }
        value
            .map(|quantity| quantity.canonical_value())
            .serialize(serializer)
    }
    pub fn deserialize<'de, K: QuantityMarker, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Quantity<K>>, D::Error> {
        Option::<f64>::deserialize(deserializer)?
            .map(Quantity::try_new)
            .transpose()
            .map_err(serde::de::Error::custom)
    }
}
