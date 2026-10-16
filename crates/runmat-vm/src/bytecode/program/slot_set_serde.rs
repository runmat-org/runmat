//! Canonical wire order for activation-presence slot sets.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub(super) fn serialize<S>(slots: &HashSet<usize>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mut sorted = slots.iter().copied().collect::<Vec<_>>();
    sorted.sort_unstable();
    sorted.serialize(serializer)
}

pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<HashSet<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Vec::<usize>::deserialize(deserializer).map(|slots| slots.into_iter().collect())
}
