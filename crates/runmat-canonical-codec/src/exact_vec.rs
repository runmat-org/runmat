//! Predictable owned collection capacities for definite canonical sequences.
use serde::de::{Error, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ExactVec<T>(Vec<T>);
impl<T> ExactVec<T> {
    pub fn into_vec(self) -> Vec<T> {
        self.0
    }
}
impl<T> std::ops::Deref for ExactVec<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> IntoIterator for ExactVec<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl<'a, T> IntoIterator for &'a ExactVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
struct Sequence<T> {
    require_count: bool,
    marker: std::marker::PhantomData<T>,
}
impl<T> Sequence<T> {
    fn new(require_count: bool) -> Self {
        Self {
            require_count,
            marker: std::marker::PhantomData,
        }
    }
}
impl<'de, T: Deserialize<'de>> Visitor<'de> for Sequence<T> {
    type Value = ExactVec<T>;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a sequence with its declared item count")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let Some(count) = sequence.size_hint() else {
            if self.require_count {
                return Err(A::Error::custom("exact sequence count required"));
            }
            // Streaming JSON does not expose a count. Preserve its ordinary
            // Serde sequence semantics; canonical input always takes the exact branch.
            return Vec::<T>::deserialize(serde::de::value::SeqAccessDeserializer::new(sequence))
                .map(ExactVec);
        };
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(A::Error::custom)?;
        for _ in 0..count {
            values.push(
                sequence
                    .next_element()?
                    .ok_or_else(|| A::Error::custom("sequence ended before its admitted count"))?,
            );
        }
        if sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {
            return Err(A::Error::custom("sequence exceeds its admitted count"));
        }
        Ok(ExactVec(values))
    }
}
impl<'de, T: Deserialize<'de>> Deserialize<'de> for ExactVec<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(Sequence::new(true))
    }
}

/// Reserve exact capacity for definite sources. Streaming formats retain their
/// standard Serde behavior; only canonical definite input has an exact inventory.
pub fn deserialize_exact_vec<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    deserializer
        .deserialize_seq(Sequence::new(false))
        .map(ExactVec::into_vec)
}
