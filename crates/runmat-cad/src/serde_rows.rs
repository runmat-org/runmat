//! Serialise a `BTreeMap` with structured keys as a list of `[key, value]` rows, in key order.
//! JSON maps need string keys; rows keep the order stable and the keys typed.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

pub fn serialize<K: Serialize, V: Serialize, S: Serializer>(
    m: &BTreeMap<K, V>,
    s: S,
) -> Result<S::Ok, S::Error> {
    let rows: Vec<(&K, &V)> = m.iter().collect();
    rows.serialize(s)
}

pub fn deserialize<'de, K: Deserialize<'de> + Ord, V: Deserialize<'de>, D: Deserializer<'de>>(
    d: D,
) -> Result<BTreeMap<K, V>, D::Error> {
    let rows = Vec::<(K, V)>::deserialize(d)?;
    Ok(rows.into_iter().collect())
}
