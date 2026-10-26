//! Established canonical array wires keep their shape. The enclosing typed field
//! supplies quantity and coordinate role; external numeric content is validated.
use super::{Quantity, QuantityMarker, QuantityStorage, UnitError};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
pub fn serialize<K: QuantityMarker, V: QuantityStorage + Serialize, S: Serializer>(
    value: &Quantity<K, V>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    if !value.canonical_values().all_finite() {
        return Err(serde::ser::Error::custom(UnitError::NonFiniteValue));
    }
    value.canonical_values().serialize(serializer)
}
pub fn deserialize<
    'de,
    K: QuantityMarker,
    V: QuantityStorage + Deserialize<'de>,
    D: Deserializer<'de>,
>(
    deserializer: D,
) -> Result<Quantity<K, V>, D::Error> {
    Quantity::try_new(V::deserialize(deserializer)?).map_err(serde::de::Error::custom)
}
