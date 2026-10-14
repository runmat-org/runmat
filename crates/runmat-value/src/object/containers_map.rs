//! GC-owned storage for a `containers.Map` handle.
//!
//! Key admission and value conversion belong to the runtime. This module owns
//! only representation, exact-key indexing, and the values traced by the GC.

use std::collections::HashMap;

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

#[derive(Clone, Debug, PartialEq)]
pub struct ContainersMapStorage {
    pub key_type: ContainersMapKeyType,
    pub value_type: ContainersMapValueType,
    pub uniform_values: bool,
    pub uniform_class: Option<ContainersMapValueClass>,
    entries: Vec<ContainersMapEntry>,
    index: HashMap<ContainersMapKey, usize>,
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
            index: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, key: &ContainersMapKey) -> bool {
        self.index.contains_key(key)
    }

    pub fn get(&self, key: &ContainersMapKey) -> Option<&ContainersMapEntry> {
        self.index.get(key).map(|&index| &self.entries[index])
    }

    /// Caller must validate uniqueness before insertion.
    pub fn insert_unique(&mut self, entry: ContainersMapEntry) {
        debug_assert!(!self.index.contains_key(&entry.normalized));
        let index = self.entries.len();
        self.index.insert(entry.normalized.clone(), index);
        self.entries.push(entry);
    }

    /// Replace an entry or append it, returning the displaced value for
    /// destruction after the GC owner's mutable borrow ends.
    pub fn replace_or_insert(&mut self, entry: ContainersMapEntry) -> Option<ContainersMapEntry> {
        if let Some(&index) = self.index.get(&entry.normalized) {
            Some(std::mem::replace(&mut self.entries[index], entry))
        } else {
            self.insert_unique(entry);
            None
        }
    }

    pub fn remove_entry(&mut self, key: &ContainersMapKey) -> Option<ContainersMapEntry> {
        let index = self.index.remove(key)?;
        let removed = self.entries.swap_remove(index);
        if let Some(moved) = self.entries.get(index) {
            self.index.insert(moved.normalized.clone(), index);
        }
        if self.entries.is_empty() {
            self.uniform_class = None;
        }
        Some(removed)
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
}

impl Trace for ContainersMapStorage {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.for_each_value(|value| value.trace(tracer));
    }
}
