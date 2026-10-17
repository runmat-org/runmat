//! One request-local value-ID authority for semantic facts and domain probes.
//! Structural constants remain in the plan fingerprint, but a ValueId bound
//! to a supplied input denotes the value actually consumed this invocation.

use anyhow::{anyhow, Result};
use runmat_time::Instant;
use runmat_value::Value;

use crate::graph::ValueId;
use crate::placement::{FusionLeafIndexWork, FusionPlacementObserver, PlacementCorrelationId};

use super::{FusionGroupPlan, FusionOp};

#[path = "resident_input.rs"]
mod resident_input;
use resident_input::FusionResidentInput;
pub(crate) use resident_input::{authenticate_resident_input, FloatingFusionEligibility};

pub(super) fn resident_slot_bytes(
    count: usize,
) -> std::result::Result<u64, super::preparation_resources::PreparationBoundError> {
    count
        .checked_mul(std::mem::size_of::<(ValueId, FusionResidentInput<'static>)>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(super::preparation_resources::PreparationBoundError::Overflow)
}

pub(crate) struct FusionInvocationLeaves<'a> {
    values: Vec<(ValueId, &'a Value)>,
    input_ids: Vec<ValueId>,
    resident_inputs: Vec<(ValueId, FusionResidentInput<'a>)>,
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
        let mut input_ids = plan.inputs.clone();
        input_ids.sort_unstable();
        if input_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(anyhow!("fusion input value IDs are duplicated"));
        }
        observation.work.input_ids_indexed = u64::try_from(plan.inputs.len()).unwrap_or(u64::MAX);
        let mut values = Vec::with_capacity(input_ids.len() + plan.const_values.len());
        for (&id, value) in &plan.const_values {
            observation.work.source_constants_indexed =
                observation.work.source_constants_indexed.saturating_add(1);
            values.push((id, value));
        }
        values.sort_unstable_by_key(|(id, _)| *id);
        let constant_count = values.len();
        for (&id, &value) in plan.inputs.iter().zip(inputs) {
            match values[..constant_count].binary_search_by_key(&id, |(key, _)| *key) {
                Ok(index) => values[index] = (id, value),
                Err(_) => values.push((id, value)),
            }
        }
        values.sort_unstable_by_key(|(id, _)| *id);
        observation.work.unique_leaves = u64::try_from(values.len()).unwrap_or(u64::MAX);
        let mut produced = Vec::with_capacity(plan.operations.len());
        for operation in &plan.operations {
            let output = match operation {
                FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => output,
            };
            if let Some(id) = output {
                observation.work.producer_ids_checked =
                    observation.work.producer_ids_checked.saturating_add(1);
                if values.binary_search_by_key(id, |(key, _)| *key).is_ok() {
                    return Err(anyhow!("fusion value ID has conflicting producers"));
                }
                produced.push(*id);
            }
        }
        produced.sort_unstable();
        if produced.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(anyhow!("fusion value ID has conflicting producers"));
        }
        let mut resident_inputs = Vec::with_capacity(inputs.len());
        for (&id, &value) in plan.inputs.iter().zip(inputs) {
            if matches!(value, Value::GpuTensor(_)) {
                observation.work.resident_inputs_checked =
                    observation.work.resident_inputs_checked.saturating_add(1);
            }
            if let Some(proof) = FusionResidentInput::authenticate(id, value) {
                resident_inputs.push((id, proof));
            }
        }
        resident_inputs.sort_unstable_by_key(|(id, _)| *id);
        observation.completed = true;
        Ok(Self {
            values,
            input_ids,
            resident_inputs,
        })
    }

    pub(crate) fn values(&self) -> &[(ValueId, &'a Value)] {
        &self.values
    }

    pub(crate) fn value(&self, id: ValueId) -> Option<&'a Value> {
        self.values
            .binary_search_by_key(&id, |(key, _)| *key)
            .ok()
            .map(|index| self.values[index].1)
    }

    pub(crate) fn unique_leaf_count(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn input_ids(&self) -> &[ValueId] {
        &self.input_ids
    }

    /// A carrier facet for this exact live input, not a new global ValueFact.
    pub(crate) fn refine_input_storage(
        &self,
        id: ValueId,
        value: &Value,
        fact: &mut runmat_types::ValueFact,
    ) {
        if let Some(proof) = self
            .resident_inputs
            .binary_search_by_key(&id, |(key, _)| *key)
            .ok()
            .map(|index| &self.resident_inputs[index].1)
        {
            proof.refine_storage(id, value, fact);
        }
    }

    pub(crate) fn require_live_resident_inputs(&self) -> Result<()> {
        for (_, proof) in &self.resident_inputs {
            proof.require_live()?;
        }
        Ok(())
    }
}
