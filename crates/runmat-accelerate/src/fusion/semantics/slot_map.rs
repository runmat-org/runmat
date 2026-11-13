//! Fixed-capacity request-local evidence keyed by admitted source ValueIds.

use anyhow::{anyhow, Result};
use serde::{ser::SerializeMap, Serialize, Serializer};

use crate::graph::ValueId;

use super::fact_index::FactIndex;

#[derive(Debug)]
pub(super) struct SlotMap<T> {
    slots: Vec<(ValueId, Option<T>)>,
    populated: usize,
}

impl<T> SlotMap<T> {
    pub(super) fn new(facts: &FactIndex) -> Self {
        let mut slots = Vec::with_capacity(facts.ids().len());
        slots.extend(facts.ids().map(|id| (id, None)));
        Self {
            slots,
            populated: 0,
        }
    }

    pub(super) fn get(&self, id: &ValueId) -> Option<&T> {
        self.slots
            .binary_search_by_key(id, |(key, _)| *key)
            .ok()
            .and_then(|index| self.slots[index].1.as_ref())
    }

    pub(super) fn contains_key(&self, id: &ValueId) -> bool {
        self.get(id).is_some()
    }

    pub(super) fn insert(&mut self, id: ValueId, value: T) -> Result<()> {
        let index = self
            .slots
            .binary_search_by_key(&id, |(key, _)| *key)
            .map_err(|_| anyhow!("fusion evidence has no source or producer slot {id}"))?;
        if self.slots[index].1.replace(value).is_none() {
            self.populated += 1;
        }
        Ok(())
    }

    pub(super) fn is_empty(&self) -> bool {
        self.populated == 0
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&ValueId, &T)> {
        self.slots
            .iter()
            .filter_map(|(id, value)| value.as_ref().map(|value| (id, value)))
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &T> {
        self.iter().map(|(_, value)| value)
    }
}

impl<T: Serialize> Serialize for SlotMap<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.populated))?;
        for (id, value) in self.iter() {
            map.serialize_entry(id, value)?;
        }
        map.end()
    }
}
