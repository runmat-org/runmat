use super::ValueFact;
use crate::{CallContract, CallableIdentity, CapabilitySet, DynamicReason};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallableFact {
    pub identity: Option<CallableIdentity>,
    /// Semantic host capabilities required when this callable executes.
    /// This is part of the value fact so requirements survive assignment,
    /// capture, branch joins, and indirect calls.
    #[serde(default)]
    pub capabilities: CapabilitySet,
    pub parameters: Vec<ValueFact>,
    pub parameters_complete: bool,
    pub outputs: Vec<ValueFact>,
    pub outputs_complete: bool,
    pub variadic_inputs: bool,
    pub variadic_outputs: bool,
    pub captures: Vec<ValueFact>,
    pub captures_complete: bool,
}

impl CallableFact {
    /// Build the call contract carried by this callable value.
    ///
    /// Identity-aware catalogs may refine this contract before inference. This
    /// representation remains authoritative for anonymous, bound, imported,
    /// and otherwise unresolved callables whose output facts travel with the
    /// value itself.
    pub fn call_contract(&self, unresolved: DynamicReason) -> CallContract {
        CallContract {
            outputs: self.outputs.clone(),
            variadic_output: (self.variadic_outputs || !self.outputs_complete)
                .then(|| Box::new(ValueFact::unknown(unresolved.clone()))),
            maximum_outputs: (self.outputs_complete && !self.variadic_outputs)
                .then_some(self.outputs.len()),
            effects: Default::default(),
            capabilities: self.capabilities.clone(),
            dynamic_reason: (!self.outputs_complete || self.variadic_outputs).then_some(unresolved),
        }
    }
}
