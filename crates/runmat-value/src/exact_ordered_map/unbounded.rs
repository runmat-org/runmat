//! Explicitly unbounded construction of the same immutable ordered map.
//!
//! This is unpublished producer state, never a second published map carrier.
//! No finite admission can be inferred from this constructor. The producer
//! moves keys/children once and publishes the two canonical shared arrays.

use std::collections::hash_map::RandomState;

use super::{
    index::{index_len, lookup, rebuild, IndexSlot, Lookup},
    ExactOrderedMap, OrderedMapEntry,
};
use crate::{ExactArrayStorage, Utf8Text};

#[derive(Debug)]
pub struct UnboundedOrderedMapBuilder<V> {
    entries: Vec<OrderedMapEntry<V>>,
    index: Vec<IndexSlot>,
    hasher: RandomState,
}

impl<V> UnboundedOrderedMapBuilder<V> {
    pub fn with_capacity(capacity: usize) -> Self {
        let entries = Vec::with_capacity(capacity);
        let slots =
            index_len(entries.capacity()).expect("unbounded map capacity exceeds layout limits");
        Self {
            entries,
            index: vec![IndexSlot(0); slots],
            hasher: RandomState::new(),
        }
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
    pub fn into_entries(self) -> impl ExactSizeIterator<Item = (Utf8Text, V)> {
        self.entries
            .into_iter()
            .map(|entry| (entry.key, entry.value))
    }
    pub fn get_index(&self, index: usize) -> Option<(&str, &V)> {
        self.entries
            .get(index)
            .map(|entry| (entry.key.as_str(), &entry.value))
    }

    pub fn get(&self, name: &str) -> Option<&V> {
        match lookup(&self.entries, &self.index, &self.hasher, name) {
            Some(Lookup::Found { entry }) => Some(&self.entries[entry].value),
            _ => None,
        }
    }

    /// Move a key and child into unpublished storage. Replacement preserves
    /// the original key allocation and its position, as in the exact builder.
    pub fn insert_text(&mut self, key: Utf8Text, value: V) -> Option<V> {
        if let Some(Lookup::Found { entry }) =
            lookup(&self.entries, &self.index, &self.hasher, key.as_str())
        {
            return Some(std::mem::replace(&mut self.entries[entry].value, value));
        }
        self.ensure_insert_capacity();
        let Some(Lookup::Vacant { bucket }) =
            lookup(&self.entries, &self.index, &self.hasher, key.as_str())
        else {
            unreachable!("unbounded builder reserves index capacity before insertion");
        };
        let entry = self.entries.len();
        self.entries.push(OrderedMapEntry { key, value });
        self.index[bucket] = IndexSlot(entry + 1);
        None
    }

    pub fn insert(&mut self, key: String, value: V) -> Option<V> {
        self.insert_text(Utf8Text::new(key), value)
    }

    pub fn remove(&mut self, name: &str) -> Option<V> {
        let Lookup::Found { entry } = lookup(&self.entries, &self.index, &self.hasher, name)?
        else {
            return None;
        };
        let removed = self.entries.remove(entry);
        rebuild(&self.entries, &mut self.index, &self.hasher);
        Some(removed.value)
    }

    pub fn finish(self) -> ExactOrderedMap<V> {
        fn publish<T>(values: Vec<T>) -> ExactArrayStorage<T> {
            if values.capacity() == 0 {
                ExactArrayStorage::empty()
            } else {
                ExactArrayStorage::shared_unbounded(values)
            }
        }
        ExactOrderedMap {
            entries: publish(self.entries),
            index: publish(self.index),
            hasher: self.hasher,
        }
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut V> {
        let Lookup::Found { entry } = lookup(&self.entries, &self.index, &self.hasher, name)?
        else {
            return None;
        };
        Some(&mut self.entries[entry].value)
    }
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &str> {
        self.entries.iter().map(|entry| entry.key.as_str())
    }
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> {
        self.entries.iter().map(|entry| &entry.value)
    }
    pub fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut V> {
        self.entries.iter_mut().map(|entry| &mut entry.value)
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &V)> {
        self.entries
            .iter()
            .map(|entry| (entry.key.as_str(), &entry.value))
    }
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = (&str, &mut V)> {
        self.entries
            .iter_mut()
            .map(|entry| (entry.key.as_str(), &mut entry.value))
    }
    pub fn contains_key(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    fn ensure_insert_capacity(&mut self) {
        if self.entries.len() < self.entries.capacity() {
            return;
        }
        self.entries.reserve(1);
        let slots =
            index_len(self.entries.capacity()).expect("unbounded map growth exceeds layout limits");
        self.index = vec![IndexSlot(0); slots];
        rebuild(&self.entries, &mut self.index, &self.hasher);
    }
}

impl<V> FromIterator<(String, V)> for UnboundedOrderedMapBuilder<V> {
    fn from_iter<I: IntoIterator<Item = (String, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Self::with_capacity(iter.size_hint().0);
        for (key, value) in iter {
            builder.insert(key, value);
        }
        builder
    }
}
pub struct UnboundedOrderedMapIntoIter<V>(std::vec::IntoIter<OrderedMapEntry<V>>);
impl<V> Iterator for UnboundedOrderedMapIntoIter<V> {
    type Item = (String, V);
    fn next(&mut self) -> Option<Self::Item> {
        self.0
            .next()
            .map(|entry| (entry.key.into_string_unbounded(), entry.value))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<V> ExactSizeIterator for UnboundedOrderedMapIntoIter<V> {}
impl<V> IntoIterator for UnboundedOrderedMapBuilder<V> {
    type Item = (String, V);
    type IntoIter = UnboundedOrderedMapIntoIter<V>;
    fn into_iter(self) -> Self::IntoIter {
        UnboundedOrderedMapIntoIter(self.entries.into_iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_publication_and_clone_preserve_the_original_children() {
        #[derive(Debug, PartialEq)]
        struct Child(usize);
        impl Clone for Child {
            fn clone(&self) -> Self {
                panic!("publishing or cloning a map must not clone children")
            }
        }
        let mut source = UnboundedOrderedMapBuilder::with_capacity(0);
        for index in 0..9 {
            source.insert(format!("field{index}"), Child(index));
        }
        assert_eq!(source.insert("field0".into(), Child(10)), Some(Child(0)));
        assert_eq!(source.remove("field4"), Some(Child(4)));
        let map = source.finish();
        let alias = map.clone();
        assert!(std::ptr::eq(
            map.entries().as_ptr(),
            alias.entries().as_ptr()
        ));
        assert_eq!(alias.get("field0"), Some(&Child(10)));
        assert_eq!(alias.get("field8"), Some(&Child(8)));
        assert_eq!(
            alias.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            ["field0", "field1", "field2", "field3", "field5", "field6", "field7", "field8"]
        );
    }
}
