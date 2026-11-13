//! GC-owned storage for a `containers.Map` handle.
//!
//! Key admission and value conversion belong to the runtime. This module owns
//! only representation, exact-key indexing, and the values traced by the GC.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;

use runmat_gc_api::{Trace, Tracer};

use crate::{NumericDType, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContainersMapKeyType {
    Char,
    Double,
    Single,
    Int32,
    UInt32,
    Int64,
    UInt64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContainersMapValueType {
    Any,
    Char,
    Double,
    Single,
    Logical,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ContainersMapKey {
    String(String),
    Float(u64),
    Int(i64),
    UInt(u64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContainersMapEntry {
    pub normalized: ContainersMapKey,
    pub key_value: Value,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContainersMapValueClass {
    Char,
    String,
    Numeric(NumericDType),
    Logical,
    Cell,
    Struct,
    Object,
    Other(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MapIndexSlot {
    Empty,
    Tombstone,
    Occupied { hash: u64, entry: usize },
}

#[derive(Debug)]
pub struct ContainersMapStorage {
    pub key_type: ContainersMapKeyType,
    pub value_type: ContainersMapValueType,
    pub uniform_values: bool,
    pub uniform_class: Option<ContainersMapValueClass>,
    entries: Vec<ContainersMapEntry>,
    index: Vec<MapIndexSlot>,
    index_len: usize,
    index_tombstones: usize,
    index_hasher: RandomState,
}

impl Clone for ContainersMapStorage {
    fn clone(&self) -> Self {
        let mut clone = Self {
            key_type: self.key_type,
            value_type: self.value_type,
            uniform_values: self.uniform_values,
            uniform_class: self.uniform_class.clone(),
            entries: self.entries.clone(),
            index: Vec::new(),
            index_len: 0,
            index_tombstones: 0,
            index_hasher: RandomState::new(),
        };
        clone.rebuild_index(self.index.len());
        clone
    }
}

impl PartialEq for ContainersMapStorage {
    fn eq(&self, other: &Self) -> bool {
        self.key_type == other.key_type
            && self.value_type == other.value_type
            && self.uniform_values == other.uniform_values
            && self.uniform_class == other.uniform_class
            && self.entries == other.entries
    }
}

impl ContainersMapStorage {
    pub fn new(
        key_type: ContainersMapKeyType,
        value_type: ContainersMapValueType,
        uniform_values: bool,
    ) -> Self {
        Self {
            key_type,
            value_type,
            uniform_values,
            uniform_class: None,
            entries: Vec::new(),
            index: Vec::new(),
            index_len: 0,
            index_tombstones: 0,
            index_hasher: RandomState::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, key: &ContainersMapKey) -> bool {
        self.find_index(key).is_some()
    }

    pub fn get(&self, key: &ContainersMapKey) -> Option<&ContainersMapEntry> {
        self.find_index(key).map(|(_, index)| &self.entries[index])
    }

    /// Caller must validate uniqueness before insertion.
    pub fn insert_unique(&mut self, entry: ContainersMapEntry) {
        debug_assert!(self.find_index(&entry.normalized).is_none());
        self.ensure_index_capacity_for_insert();
        let index = self.entries.len();
        self.entries.push(entry);
        self.insert_index(index);
    }

    /// Replace an entry or append it, returning the displaced value for
    /// destruction after the GC owner's mutable borrow ends.
    pub fn replace_or_insert(&mut self, entry: ContainersMapEntry) -> Option<ContainersMapEntry> {
        if let Some((_, index)) = self.find_index(&entry.normalized) {
            Some(std::mem::replace(&mut self.entries[index], entry))
        } else {
            self.insert_unique(entry);
            None
        }
    }

    pub fn remove_entry(&mut self, key: &ContainersMapKey) -> Option<ContainersMapEntry> {
        let (slot, index) = self.find_index(key)?;
        self.index[slot] = MapIndexSlot::Tombstone;
        self.index_len -= 1;
        self.index_tombstones += 1;
        let moved_slot = (index + 1 < self.entries.len()).then(|| {
            let moved = self
                .entries
                .last()
                .expect("non-last removal has a final map entry");
            self.find_index(&moved.normalized)
                .map(|(slot, _)| slot)
                .expect("final map entry remains indexed before removal")
        });
        let removed = self.entries.swap_remove(index);
        if let Some(moved_slot) = moved_slot {
            let MapIndexSlot::Occupied { hash, .. } = self.index[moved_slot] else {
                unreachable!("lookup returned an occupied slot")
            };
            self.index[moved_slot] = MapIndexSlot::Occupied { hash, entry: index };
        }
        if self.entries.is_empty() {
            self.uniform_class = None;
        }
        if self.index_tombstones > self.index_len {
            self.rebuild_index(self.index.len());
        }
        Some(removed)
    }

    fn hash_key(&self, key: &ContainersMapKey) -> u64 {
        self.index_hasher.hash_one(key)
    }

    fn find_index(&self, key: &ContainersMapKey) -> Option<(usize, usize)> {
        if self.index.is_empty() {
            return None;
        }
        let hash = self.hash_key(key);
        let mask = self.index.len() - 1;
        for step in 0..self.index.len() {
            let slot = (hash as usize).wrapping_add(step) & mask;
            match self.index[slot] {
                MapIndexSlot::Empty => return None,
                MapIndexSlot::Tombstone => {}
                MapIndexSlot::Occupied {
                    hash: candidate,
                    entry,
                } if candidate == hash && self.entries[entry].normalized == *key => {
                    return Some((slot, entry));
                }
                MapIndexSlot::Occupied { .. } => {}
            }
        }
        None
    }

    fn ensure_index_capacity_for_insert(&mut self) {
        let used = self.index_len + self.index_tombstones;
        if self.index.is_empty() || used.saturating_add(1) > self.index.len() / 2 {
            let capacity = self
                .index
                .len()
                .max(4)
                .checked_mul(2)
                .expect("containers.Map index capacity overflow");
            self.rebuild_index(capacity);
        }
    }

    fn rebuild_index(&mut self, requested: usize) {
        let minimum = self
            .entries
            .len()
            .checked_mul(2)
            .expect("containers.Map index capacity overflow")
            .max(8);
        let capacity = requested
            .max(minimum)
            .checked_next_power_of_two()
            .expect("containers.Map index capacity overflow");
        self.index = vec![MapIndexSlot::Empty; capacity];
        self.index_len = 0;
        self.index_tombstones = 0;
        for entry in 0..self.entries.len() {
            self.insert_index(entry);
        }
    }

    fn insert_index(&mut self, entry: usize) {
        let hash = self.hash_key(&self.entries[entry].normalized);
        let mask = self.index.len() - 1;
        let mut first_tombstone = None;
        for step in 0..self.index.len() {
            let slot = (hash as usize).wrapping_add(step) & mask;
            match self.index[slot] {
                MapIndexSlot::Empty => {
                    let slot = first_tombstone.unwrap_or(slot);
                    if first_tombstone.is_some() {
                        self.index_tombstones -= 1;
                    }
                    self.index[slot] = MapIndexSlot::Occupied { hash, entry };
                    self.index_len += 1;
                    return;
                }
                MapIndexSlot::Tombstone if first_tombstone.is_none() => {
                    first_tombstone = Some(slot);
                }
                MapIndexSlot::Tombstone | MapIndexSlot::Occupied { .. } => {}
            }
        }
        unreachable!("map index load bound always leaves an empty slot")
    }

    pub fn sorted_entries(&self) -> Vec<&ContainersMapEntry> {
        let mut entries = self.entries.iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| match (&left.normalized, &right.normalized) {
            (ContainersMapKey::String(left), ContainersMapKey::String(right)) => left.cmp(right),
            (ContainersMapKey::Int(left), ContainersMapKey::Int(right)) => left.cmp(right),
            (ContainersMapKey::UInt(left), ContainersMapKey::UInt(right)) => left.cmp(right),
            (ContainersMapKey::Float(left), ContainersMapKey::Float(right)) => {
                f64::from_bits(*left)
                    .partial_cmp(&f64::from_bits(*right))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            _ => std::cmp::Ordering::Equal,
        });
        entries
    }

    pub fn for_each_value<'a>(&'a self, mut visitor: impl FnMut(&'a Value)) {
        for entry in &self.entries {
            visitor(&entry.key_value);
            visitor(&entry.value);
        }
    }

    pub(crate) fn trace_value_at(&self, index: usize) -> Option<&Value> {
        let entry = self.entries.get(index / 2)?;
        Some(if index.is_multiple_of(2) {
            &entry.key_value
        } else {
            &entry.value
        })
    }

    pub(super) fn retained_physical_bytes(
        &self,
    ) -> Result<usize, crate::ExactHostNumericAllocationError> {
        use crate::ExactHostNumericAllocationError::LayoutOverflow;
        use std::alloc::Layout;

        let entries = Layout::array::<ContainersMapEntry>(self.entries.capacity())
            .map_err(|_| LayoutOverflow)?
            .size();
        let index_entries = Layout::array::<MapIndexSlot>(self.index.capacity())
            .map_err(|_| LayoutOverflow)?
            .size();
        let mut bytes = entries.checked_add(index_entries).ok_or(LayoutOverflow)?;
        for entry in &self.entries {
            if let ContainersMapKey::String(value) = &entry.normalized {
                bytes = bytes.checked_add(value.capacity()).ok_or(LayoutOverflow)?;
            }
        }
        Ok(bytes)
    }
}

impl Trace for ContainersMapStorage {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.for_each_value(|value| value.trace(tracer));
    }
}
