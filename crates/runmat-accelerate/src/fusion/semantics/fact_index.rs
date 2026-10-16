//! Fixed-capacity facts for the exact invocation value-ID graph.
//!
//! Every source leaf and produced output has one sorted slot before semantic
//! inference begins. Admission can replace a fact without growing a map or
//! treating a sparse, large ValueId as an allocation index.

use anyhow::{anyhow, Result};
use runmat_types::ValueFact;
use serde::{ser::SerializeMap, Serialize, Serializer};

use crate::graph::ValueId;

use super::{invocation_leaves::FusionInvocationLeaves, FusionGroupPlan, FusionOp};

#[derive(Debug)]
pub(in crate::fusion) struct FactIndex {
    slots: Vec<(ValueId, Option<ValueFact>)>,
}

impl FactIndex {
    pub(super) fn new(plan: &FusionGroupPlan, leaves: &FusionInvocationLeaves<'_>) -> Self {
        let mut slots = Vec::with_capacity(leaves.values().len() + plan.operations.len());
        slots.extend(leaves.values().iter().map(|(id, _)| (*id, None)));
        slots.extend(
            plan.operations
                .iter()
                .filter_map(|operation| match operation {
                    FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => {
                        *output
                    }
                })
                .map(|id| (id, None)),
        );
        slots.sort_unstable_by_key(|(id, _)| *id);
        slots.dedup_by_key(|(id, _)| *id);
        Self { slots }
    }

    pub(in crate::fusion) fn get(&self, id: &ValueId) -> Option<&ValueFact> {
        self.slots
            .binary_search_by_key(id, |(key, _)| *key)
            .ok()
            .and_then(|index| self.slots[index].1.as_ref())
    }

    pub(super) fn get_mut(&mut self, id: &ValueId) -> Option<&mut ValueFact> {
        self.slots
            .binary_search_by_key(id, |(key, _)| *key)
            .ok()
            .and_then(|index| self.slots[index].1.as_mut())
    }

    pub(super) fn insert(&mut self, id: ValueId, fact: ValueFact) -> Result<()> {
        let index = self
            .slots
            .binary_search_by_key(&id, |(key, _)| *key)
            .map_err(|_| anyhow!("fusion fact has no source or producer slot {id}"))?;
        self.slots[index].1 = Some(fact);
        Ok(())
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&ValueId, &ValueFact)> {
        self.slots
            .iter()
            .filter_map(|(id, fact)| fact.as_ref().map(|fact| (id, fact)))
    }

    pub(super) fn ids(&self) -> impl ExactSizeIterator<Item = ValueId> + '_ {
        self.slots.iter().map(|(id, _)| *id)
    }
}

impl Serialize for FactIndex {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        for (id, fact) in self.iter() {
            map.serialize_entry(id, fact)?;
        }
        map.end()
    }
}
