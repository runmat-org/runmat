use super::ValueFact;
use crate::{CallableIdentity, CapabilitySet};
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
