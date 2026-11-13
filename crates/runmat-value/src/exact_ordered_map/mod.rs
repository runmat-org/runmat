//! Immutable ordered map generations built in their final exact backing.
//!
//! Entries retain insertion order; the fixed lookup index contains only entry
//! ordinals. Keys and child values keep independent physical/GC owners. No map
//! operation implicitly clones a child or obtains allocation authority.

use std::collections::hash_map::RandomState;

use crate::{ExactArrayStorage, ExactHostNumericAllocationError, Utf8Text};

mod builder;
mod index;
mod unbounded;
mod owned;
pub use builder::{ExactOrderedMapBuilder, MapInsertError, MapInsertErrorKind};
use index::{index_len, lookup, IndexSlot, Lookup};
pub use owned::OwnedOrderedMapEntries;
pub use unbounded::UnboundedOrderedMapBuilder;

#[derive(Clone, Debug)]
pub struct OrderedMapEntry<V> {
    key: Utf8Text,
    value: V,
}

impl<V> OrderedMapEntry<V> {
    pub fn key(&self) -> &Utf8Text {
        &self.key
    }
    pub fn value(&self) -> &V {
        &self.value
    }
}

/// Raw physical bytes, excluding the caller's retained-claim control metadata
/// and independently owned key text/child values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactOrderedMapQuote {
    pub requested_bytes: usize,
    pub retained_claims: usize,
    pub entry_capacity: usize,
    pub index_slots: usize,
}

#[derive(Debug)]
pub struct ExactOrderedMap<V> {
    entries: ExactArrayStorage<OrderedMapEntry<V>>,
    index: ExactArrayStorage<IndexSlot>,
    hasher: RandomState,
}

impl<V> ExactOrderedMap<V> {
    pub fn quote(capacity: usize) -> Result<ExactOrderedMapQuote, ExactHostNumericAllocationError> {
        let index_slots = index_len(capacity)?;
        let requested_bytes = ExactArrayStorage::<OrderedMapEntry<V>>::requested_bytes(capacity)?
            .checked_add(ExactArrayStorage::<IndexSlot>::requested_bytes(
                index_slots,
            )?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        Ok(ExactOrderedMapQuote {
            requested_bytes,
            retained_claims: if capacity == 0 { 0 } else { 2 },
            entry_capacity: capacity,
            index_slots,
        })
    }

    /// Exact key-copy demand. Already-admitted key owners may instead move
    /// into the builder without allocating or requesting another claim.
    pub fn key_quote(text: &str) -> Result<(usize, usize), ExactHostNumericAllocationError> {
        Ok((
            Utf8Text::requested_bytes_for_capacity(text.len())?,
            usize::from(!text.is_empty()),
        ))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn entries(&self) -> &[OrderedMapEntry<V>] {
        &self.entries
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &V)> + DoubleEndedIterator + Clone {
        self.entries
            .iter()
            .map(|entry| (entry.key.as_str(), &entry.value))
    }
    pub fn get(&self, name: &str) -> Option<&V> {
        match lookup(&self.entries, &self.index, &self.hasher, name) {
            Some(Lookup::Found { entry, .. }) => Some(&self.entries[entry].value),
            _ => None,
        }
    }
    pub fn contains_key(&self, name: &str) -> bool {
        self.get(name).is_some()
    }
}

// Only shared or allocation-free empty states can enter a published map. This bound comes from
// ExactArrayStorage's general Clone API; it never invokes V::clone here.
impl<V: Clone> Clone for ExactOrderedMap<V> {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            index: self.index.clone(),
            hasher: self.hasher.clone(),
        }
    }
}

// Match map equality, not sequence equality: iteration order is observable
// separately, and two insertion orders can describe the same mapping.
impl<V: PartialEq> PartialEq for ExactOrderedMap<V> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(name, value)| other.get(name) == Some(value))
    }
}
impl<V: Eq> Eq for ExactOrderedMap<V> {}

impl<V> ExactOrderedMap<V> {
    pub fn empty() -> Self {
        Self {
            entries: ExactArrayStorage::empty(),
            index: ExactArrayStorage::empty(),
            hasher: RandomState::new(),
        }
    }
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &str> + DoubleEndedIterator + Clone {
        self.entries.iter().map(|entry| entry.key.as_str())
    }
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> + DoubleEndedIterator + Clone {
        self.entries.iter().map(|entry| &entry.value)
    }
    pub fn get_index(&self, index: usize) -> Option<(&str, &V)> {
        self.entries
            .get(index)
            .map(|entry| (entry.key.as_str(), &entry.value))
    }
    pub fn is_unique(&self) -> bool {
        self.entries.is_unique() && self.index.is_unique()
    }
    /// No-allocation update of existing values only. Keys/index structure
    /// remain immutable, and aliases never trigger implicit copy-on-write.
    pub fn try_values_mut_unique(&mut self) -> Option<impl ExactSizeIterator<Item = &mut V>> {
        Some(
            self.entries
                .try_as_mut_slice_unique()?
                .iter_mut()
                .map(|entry| &mut entry.value),
        )
    }
    pub fn try_get_mut_unique(&mut self, name: &str) -> Option<&mut V> {
        let Lookup::Found { entry } = lookup(&self.entries, &self.index, &self.hasher, name)?
        else {
            return None;
        };
        Some(&mut self.entries.try_as_mut_slice_unique()?[entry].value)
    }
    /// Existing-value replacement only: no structural growth or implicit COW.
    /// A failed lookup or shared generation returns the unchanged input.
    pub fn try_replace_unique(&mut self, name: &str, value: V) -> Result<V, V> {
        let Some(Lookup::Found { entry }) = lookup(&self.entries, &self.index, &self.hasher, name)
        else {
            return Err(value);
        };
        let Some(entries) = self.entries.try_as_mut_slice_unique() else {
            return Err(value);
        };
        Ok(std::mem::replace(&mut entries[entry].value, value))
    }
}
impl<V, K: AsRef<str> + ?Sized> std::ops::Index<&K> for ExactOrderedMap<V> {
    type Output = V;
    fn index(&self, key: &K) -> &V {
        self.get(key.as_ref()).expect("map key is present")
    }
}

pub struct OrderedMapIter<'a, V>(std::slice::Iter<'a, OrderedMapEntry<V>>);
impl<'a, V> Iterator for OrderedMapIter<'a, V> {
    type Item = (&'a str, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        self.0
            .next()
            .map(|entry| (entry.key.as_str(), &entry.value))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<V> ExactSizeIterator for OrderedMapIter<'_, V> {}
impl<'a, V> IntoIterator for &'a ExactOrderedMap<V> {
    type Item = (&'a str, &'a V);
    type IntoIter = OrderedMapIter<'a, V>;
    fn into_iter(self) -> Self::IntoIter {
        OrderedMapIter(self.entries.iter())
    }
}
