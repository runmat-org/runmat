//! Source-indexed WGSL expression slots. The selected producer cannot grow a
//! request-local hash table while emitting an already admitted source graph.

use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::graph::ValueId;

use super::{FusionGroupPlan, FusionOp};
use crate::fusion::ExpressionLookup;

#[derive(Clone)]
pub(crate) struct ExpressionSlots {
    slots: Vec<(ValueId, Option<String>)>,
    populated: usize,
}

impl ExpressionSlots {
    pub(crate) fn new(plan: &FusionGroupPlan) -> Result<Self> {
        let capacity = plan
            .inputs
            .len()
            .checked_add(plan.const_values.len())
            .and_then(|count| count.checked_add(plan.operations.len()))
            .ok_or_else(|| anyhow!("fusion expression slot count overflows"))?;
        let mut slots = Vec::new();
        slots.try_reserve_exact(capacity)?;
        slots.extend(plan.inputs.iter().copied().map(|id| (id, None)));
        slots.extend(plan.const_values.keys().copied().map(|id| (id, None)));
        for operation in &plan.operations {
            let output = match operation {
                FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => output,
            };
            if let Some(output) = output {
                slots.push((*output, None));
            }
        }
        slots.sort_unstable_by_key(|(id, _)| *id);
        slots.dedup_by_key(|(id, _)| *id);
        Ok(Self {
            slots,
            populated: 0,
        })
    }

    pub(crate) fn from_sorted_ids(ids: &[ValueId]) -> Self {
        Self {
            slots: ids.iter().copied().map(|id| (id, None)).collect(),
            populated: 0,
        }
    }

    pub(super) fn from_map(
        plan: &FusionGroupPlan,
        expressions: HashMap<ValueId, String>,
    ) -> Result<Self> {
        let mut slots = Self::new(plan)?;
        for (id, expression) in expressions {
            slots.insert(id, expression)?;
        }
        Ok(slots)
    }

    pub(crate) fn get(&self, id: &ValueId) -> Option<&String> {
        let index = self.slots.binary_search_by_key(id, |(key, _)| *key).ok()?;
        self.slots[index].1.as_ref()
    }

    pub(crate) fn contains_key(&self, id: &ValueId) -> bool {
        self.get(id).is_some()
    }

    pub(crate) fn insert(&mut self, id: ValueId, expression: String) -> Result<()> {
        let index = self
            .slots
            .binary_search_by_key(&id, |(key, _)| *key)
            .map_err(|_| anyhow!("fusion expression has no source or producer slot {id}"))?;
        if self.slots[index].1.is_none() {
            self.populated += 1;
        }
        self.slots[index].1 = Some(expression);
        Ok(())
    }

    pub(crate) fn insert_if_missing(&mut self, id: ValueId) -> Result<()> {
        if !self.contains_key(&id) {
            self.insert(id, String::new())?;
        }
        Ok(())
    }

    pub(super) fn keys(&self) -> impl Iterator<Item = &ValueId> {
        self.slots
            .iter()
            .filter_map(|(id, expression)| expression.as_ref().map(|_| id))
    }

    pub(super) fn len(&self) -> usize {
        self.populated
    }
}

impl ExpressionLookup for ExpressionSlots {
    fn get(&self, id: &ValueId) -> Option<&String> {
        ExpressionSlots::get(self, id)
    }
}
