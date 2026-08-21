use crate::{
    CapabilitySet, DynamicReason, EffectSet, InferenceDiagnostic, OutputSelection, ValueFact,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallContract {
    /// Ordered output slots with the most precise statically declared fact.
    pub outputs: Vec<ValueFact>,
    /// Fact for slots beyond `outputs`, when varargout or requested-count
    /// semantics allow them.
    pub variadic_output: Option<Box<ValueFact>>,
    /// Maximum supported requested output count; `None` means unbounded.
    pub maximum_outputs: Option<usize>,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
    pub dynamic_reason: Option<DynamicReason>,
}

impl CallContract {
    pub fn fixed(outputs: Vec<ValueFact>) -> Self {
        let maximum_outputs = Some(outputs.len());
        Self {
            outputs,
            variadic_output: None,
            maximum_outputs,
            effects: EffectSet::default(),
            capabilities: CapabilitySet::default(),
            dynamic_reason: None,
        }
    }

    pub fn dynamic(reason: DynamicReason) -> Self {
        Self {
            outputs: Vec::new(),
            variadic_output: Some(Box::new(ValueFact::unknown(reason.clone()))),
            maximum_outputs: None,
            effects: EffectSet::default(),
            capabilities: CapabilitySet::default(),
            dynamic_reason: Some(reason),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallInference {
    /// One entry for every requested output when the request count is known.
    pub outputs: Vec<ValueFact>,
    /// Whether output count itself remains runtime-dependent.
    pub dynamic_outputs: bool,
    pub discarded: Vec<usize>,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
    pub diagnostics: Vec<InferenceDiagnostic>,
}

impl CallInference {
    pub fn materialized_outputs(&self) -> impl Iterator<Item = (usize, &ValueFact)> {
        self.outputs
            .iter()
            .enumerate()
            .filter(|(index, _)| !self.discarded.contains(index))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallRequest {
    pub arguments: Vec<ValueFact>,
    pub literals: crate::LiteralContext,
    pub outputs: OutputSelection,
}
