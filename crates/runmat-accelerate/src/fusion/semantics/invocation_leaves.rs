//! One request-local value-ID authority for semantic facts and domain probes.
//! Structural constants remain in the plan fingerprint, but a ValueId bound
//! to a supplied input denotes the value actually consumed this invocation.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};
use runmat_time::Instant;
use runmat_value::Value;

use crate::graph::ValueId;
use crate::placement::{FusionLeafIndexWork, FusionPlacementObserver, PlacementCorrelationId};

use super::{FusionGroupPlan, FusionOp};

#[path = "resident_input.rs"]
mod resident_input;
use resident_input::FusionResidentInput;

pub(crate) struct FusionInvocationLeaves<'a> {
    values: HashMap<ValueId, &'a Value>,
    input_ids: HashSet<ValueId>,
    resident_inputs: HashMap<ValueId, FusionResidentInput<'a>>,
}

struct LeafIndexObservation {
    observer: FusionPlacementObserver,
    started: Instant,
    work: FusionLeafIndexWork,
    completed: bool,
}

impl LeafIndexObservation {
    fn new(correlation: Option<PlacementCorrelationId>) -> Self {
        Self {
            observer: FusionPlacementObserver::new(correlation),
            started: Instant::now(),
            work: FusionLeafIndexWork::default(),
            completed: false,
        }
    }
}

impl Drop for LeafIndexObservation {
    fn drop(&mut self) {
        self.observer
            .leaf_index_preparation(self.started, self.work, self.completed);
    }
}

impl<'a> FusionInvocationLeaves<'a> {
    #[cfg(test)]
    pub(crate) fn new(plan: &'a FusionGroupPlan, inputs: &[&'a Value]) -> Result<Self> {
        Self::new_observed(plan, inputs, None)
    }

    pub(crate) fn new_observed(
        plan: &'a FusionGroupPlan,
        inputs: &[&'a Value],
        correlation: Option<PlacementCorrelationId>,
    ) -> Result<Self> {
        let mut observation = LeafIndexObservation::new(correlation);
        if inputs.len() != plan.inputs.len() {
            return Err(anyhow!("fusion operand count does not match the plan"));
        }
        let mut input_ids = HashSet::with_capacity(plan.inputs.len());
        for &id in &plan.inputs {
            observation.work.input_ids_indexed =
                observation.work.input_ids_indexed.saturating_add(1);
            if !input_ids.insert(id) {
                return Err(anyhow!("fusion input value IDs are duplicated"));
            }
        }
        let mut values = HashMap::with_capacity(input_ids.len() + plan.const_values.len());
        for (&id, value) in &plan.const_values {
            observation.work.source_constants_indexed =
                observation.work.source_constants_indexed.saturating_add(1);
            values.insert(id, value);
        }
        for (&id, &value) in plan.inputs.iter().zip(inputs) {
            values.insert(id, value);
        }
        observation.work.unique_leaves = u64::try_from(values.len()).unwrap_or(u64::MAX);
        let mut produced = HashSet::new();
        for operation in &plan.operations {
            let output = match operation {
                FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => output,
            };
            if let Some(id) = output {
                observation.work.producer_ids_checked =
                    observation.work.producer_ids_checked.saturating_add(1);
                if values.contains_key(id) || !produced.insert(*id) {
                    return Err(anyhow!("fusion value ID has conflicting producers"));
                }
            }
        }
        let resident_inputs = plan
            .inputs
            .iter()
            .zip(inputs)
            .filter_map(|(&id, &value)| {
                if matches!(value, Value::GpuTensor(_)) {
                    observation.work.resident_inputs_checked =
                        observation.work.resident_inputs_checked.saturating_add(1);
                }
                FusionResidentInput::authenticate(id, value).map(|proof| (id, proof))
            })
            .collect();
        observation.completed = true;
        Ok(Self {
            values,
            input_ids,
            resident_inputs,
        })
    }

    pub(crate) fn values(&self) -> &HashMap<ValueId, &'a Value> {
        &self.values
    }

    pub(crate) fn unique_leaf_count(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn input_ids(&self) -> &HashSet<ValueId> {
        &self.input_ids
    }

    /// A carrier facet for this exact live input, not a new global ValueFact.
    pub(crate) fn refine_input_storage(
        &self,
        id: ValueId,
        value: &Value,
        fact: &mut runmat_types::ValueFact,
    ) {
        if let Some(proof) = self.resident_inputs.get(&id) {
            proof.refine_storage(id, value, fact);
        }
    }

    pub(crate) fn require_live_resident_inputs(&self) -> Result<()> {
        for proof in self.resident_inputs.values() {
            proof.require_live()?;
        }
        Ok(())
    }
}
