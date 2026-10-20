use std::collections::hash_map::RandomState;

use super::{
    index::{lookup, rebuild, IndexSlot, Lookup},
    ExactOrderedMap, OrderedMapEntry,
};
use crate::{
    ExactArrayStorage, ExactArrayStorageBuilder, ExactHostAllocationPermit,
    ExactHostNumericAllocationError, Utf8Text,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MapInsertErrorKind {
    CapacityExceeded,
    KeyNotAdmitted,
    ValueNotAdmitted,
    DuplicateKey,
}
impl std::fmt::Display for MapInsertErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::CapacityExceeded => "ordered map capacity exceeded",
            Self::KeyNotAdmitted => "ordered map key has no retained allocation claim",
            Self::ValueNotAdmitted => "ordered map value has no retained allocation claim",
            Self::DuplicateKey => "ordered map contains a duplicate key",
        })
    }
}
impl std::error::Error for MapInsertErrorKind {}

/// A rejected insertion retains both original owners, including key claims.
#[derive(Debug)]
pub struct MapInsertError<V> {
    pub kind: MapInsertErrorKind,
    key: Utf8Text,
    value: V,
}
impl<V> MapInsertError<V> {
    pub fn into_parts(self) -> (MapInsertErrorKind, Utf8Text, V) {
        (self.kind, self.key, self.value)
    }
    pub(crate) fn value_not_admitted(key: Utf8Text, value: V) -> Self {
        Self {
            kind: MapInsertErrorKind::ValueNotAdmitted,
            key,
            value,
        }
    }
}

/// Unique unpublished construction of the final generation. Entry and index
/// blocks are reserved before this builder can consume any key or child.
#[derive(Debug)]
pub struct ExactOrderedMapBuilder<V> {
    entries: ExactArrayStorageBuilder<OrderedMapEntry<V>>,
    index: ExactArrayStorage<IndexSlot>,
    hasher: RandomState,
}

impl<V> ExactOrderedMapBuilder<V> {
    pub fn try_new(
        capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let quote = ExactOrderedMap::<V>::quote(capacity)?;
        let entries = ExactArrayStorage::try_begin(capacity, permit)?;
        let mut index = ExactArrayStorage::try_begin(quote.index_slots, permit)?;
        for _ in 0..quote.index_slots {
            index
                .try_push(IndexSlot(0))
                .expect("one reserved slot per index entry");
        }
        let index = index
            .try_finish()
            .expect("index initialization is complete");
        Ok(Self {
            entries,
            index,
            hasher: RandomState::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.entries.initialized().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn entries(&self) -> &[OrderedMapEntry<V>] {
        self.entries.initialized()
    }

    pub(crate) fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        let Lookup::Found { entry, .. } =
            lookup(self.entries.initialized(), &self.index, &self.hasher, key)?
        else {
            return None;
        };
        Some(&mut self.entries.initialized_mut()[entry].value)
    }

    /// Replace an existing value without changing its original key or order,
    /// or append a new independently admitted key and owned child.
    pub fn try_insert_prepared(
        &mut self,
        key: Utf8Text,
        value: V,
    ) -> Result<Option<V>, MapInsertError<V>> {
        self.insert(key, value, false)
    }

    /// Reject duplicates without dropping or replacing either input owner.
    pub fn try_insert_new_prepared(
        &mut self,
        key: Utf8Text,
        value: V,
    ) -> Result<(), MapInsertError<V>> {
        self.insert(key, value, true).map(|old| {
            debug_assert!(old.is_none());
        })
    }

    /// Replace one existing child without allocating or constructing a key.
    /// Failure returns the unchanged input child.
    pub fn try_replace_existing(&mut self, key: &str, value: V) -> Result<V, V> {
        let Some(Lookup::Found { entry, .. }) =
            lookup(self.entries.initialized(), &self.index, &self.hasher, key)
        else {
            return Err(value);
        };
        Ok(std::mem::replace(
            &mut self.entries.initialized_mut()[entry].value,
            value,
        ))
    }

    fn insert(
        &mut self,
        key: Utf8Text,
        value: V,
        reject_duplicate: bool,
    ) -> Result<Option<V>, MapInsertError<V>> {
        let found = lookup(
            self.entries.initialized(),
            &self.index,
            &self.hasher,
            key.as_str(),
        );
        if let Some(Lookup::Found { entry, .. }) = found {
            if reject_duplicate {
                return Err(MapInsertError {
                    kind: MapInsertErrorKind::DuplicateKey,
                    key,
                    value,
                });
            }
            let old = std::mem::replace(&mut self.entries.initialized_mut()[entry].value, value);
            return Ok(Some(old));
        }
        if self.len() == self.entries.expected_len() {
            return Err(MapInsertError {
                kind: MapInsertErrorKind::CapacityExceeded,
                key,
                value,
            });
        }
        if key.capacity() != 0 && !key.has_retained_claim() {
            return Err(MapInsertError {
                kind: MapInsertErrorKind::KeyNotAdmitted,
                key,
                value,
            });
        }
        let Some(Lookup::Vacant { bucket }) = found else {
            unreachable!("nonfull entry storage has a vacant fixed-index bucket");
        };
        let entry = self.len();
        if let Err(OrderedMapEntry { key, value }) =
            self.entries.try_push(OrderedMapEntry { key, value })
        {
            return Err(MapInsertError {
                kind: MapInsertErrorKind::CapacityExceeded,
                key,
                value,
            });
        }
        self.index
            .try_as_mut_slice_unique()
            .expect("unpublished index is unique")[bucket] = IndexSlot(entry + 1);
        Ok(None)
    }

    /// Explicit builder edit: shift remaining entries to preserve their order,
    /// then rebuild the already-reserved index without temporary allocation.
    pub fn remove(&mut self, name: &str) -> Option<V> {
        let Lookup::Found { entry, .. } =
            lookup(self.entries.initialized(), &self.index, &self.hasher, name)?
        else {
            return None;
        };
        let removed = self
            .entries
            .remove_initialized(entry)
            .expect("lookup returned an initialized entry");
        rebuild(
            self.entries.initialized(),
            self.index
                .try_as_mut_slice_unique()
                .expect("unpublished index is unique"),
            &self.hasher,
        );
        Some(removed.value)
    }

    /// Publish the initialized prefix, retaining the full original capacity
    /// claim even if replacement/removal left unused slots.
    pub fn finish(self) -> ExactOrderedMap<V> {
        ExactOrderedMap {
            entries: self.entries.finish_initialized(),
            index: self.index,
            hasher: self.hasher,
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
