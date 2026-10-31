//! Bounded typed-key JSON maps inside encrypted remote pool checkpoints.
//!
//! JSON object keys cannot represent nonce or wave identities. Ordered pairs
//! preserve their typed identity and reject duplicate authority on restore.

use super::MAX_CHECKPOINT_METADATA_BYTES;
use serde::de::{Error as _, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;

pub(super) fn serialize<S, K, V>(map: &BTreeMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    K: Ord + Serialize,
    V: Serialize,
{
    serializer.collect_seq(map.iter())
}

pub(super) fn deserialize_clock<'de, D, K, V>(deserializer: D) -> Result<BTreeMap<K, V>, D::Error>
where
    D: Deserializer<'de>,
    K: Ord + Deserialize<'de>,
    V: Deserialize<'de>,
{
    deserializer.deserialize_seq(MapVisitor::<
        K,
        V,
        { runmat_execution::security::MAX_CLOCK_OBSERVATIONS_PER_NODE },
    >(PhantomData))
}

pub(super) fn deserialize_metadata<'de, D, K, V>(
    deserializer: D,
) -> Result<BTreeMap<K, V>, D::Error>
where
    D: Deserializer<'de>,
    K: Ord + Deserialize<'de>,
    V: Deserialize<'de>,
{
    // Certified waves survive release for the lifetime of a bounded checkpoint.
    // Unlike active clocks they may exceed the per-node active-slot quota.
    deserializer.deserialize_seq(MapVisitor::<K, V, MAX_CHECKPOINT_METADATA_BYTES>(
        PhantomData,
    ))
}

struct MapVisitor<K, V, const LIMIT: usize>(PhantomData<(K, V)>);

impl<'de, K, V, const LIMIT: usize> Visitor<'de> for MapVisitor<K, V, LIMIT>
where
    K: Ord + Deserialize<'de>,
    V: Deserialize<'de>,
{
    type Value = BTreeMap<K, V>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a bounded sequence of unique typed checkpoint entries")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut map = BTreeMap::new();
        while let Some((key, value)) = sequence.next_element::<(K, V)>()? {
            if map.len() >= LIMIT {
                return Err(A::Error::custom("typed checkpoint entry limit exceeded"));
            }
            if map.insert(key, value).is_some() {
                return Err(A::Error::custom("duplicate typed checkpoint key"));
            }
        }
        Ok(map)
    }
}
