//! Structured identities remain typed sequence entries across the JSON boundary.
use std::collections::{btree_map::Entry, BTreeMap};
use std::fmt;
use std::marker::PhantomData;

use serde::de::{Error, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub(super) fn serialize<K: Serialize, V: Serialize, S: Serializer>(
    values: &BTreeMap<K, V>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(values.iter())
}

pub(super) fn deserialize<
    'de,
    K: Deserialize<'de> + Ord,
    V: Deserialize<'de>,
    D: Deserializer<'de>,
>(
    deserializer: D,
) -> Result<BTreeMap<K, V>, D::Error> {
    struct Entries<K, V>(PhantomData<fn() -> BTreeMap<K, V>>);

    impl<'de, K: Deserialize<'de> + Ord, V: Deserialize<'de>> Visitor<'de> for Entries<K, V> {
        type Value = BTreeMap<K, V>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a sequence of unique exact-import identity entries")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
            let mut values = BTreeMap::new();
            while let Some((key, value)) = sequence.next_element::<(K, V)>()? {
                match values.entry(key) {
                    Entry::Vacant(entry) => {
                        entry.insert(value);
                    }
                    Entry::Occupied(_) => {
                        return Err(A::Error::custom("duplicate exact-import identity"))
                    }
                }
            }
            Ok(values)
        }
    }

    deserializer.deserialize_seq(Entries(PhantomData))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    struct Identity {
        path: Vec<u64>,
        entity: u64,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Names(#[serde(with = "super")] BTreeMap<Identity, Vec<String>>);

    #[test]
    fn structured_identity_entries_preserve_distinct_paths_and_reject_duplicates() {
        let values = Names(BTreeMap::from([
            (
                Identity {
                    path: vec![1],
                    entity: u64::MAX,
                },
                vec!["first".into(), "shared".into()],
            ),
            (
                Identity {
                    path: vec![2],
                    entity: u64::MAX,
                },
                vec!["second".into()],
            ),
        ]));
        let wire = serde_json::to_string(&values).unwrap();
        assert_eq!(serde_json::from_str::<Names>(&wire).unwrap(), values);
        let mut duplicate = serde_json::from_str::<serde_json::Value>(&wire).unwrap();
        let rows = duplicate.as_array_mut().unwrap();
        rows.push(rows[0].clone());
        assert!(serde_json::from_value::<Names>(duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate exact-import identity"));
        assert_eq!(
            serde_json::from_str::<Names>("[]").unwrap(),
            Names(BTreeMap::new())
        );
    }
}
