use super::ValueFact;
use crate::CallableIdentity;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallableFact {
    pub identity: Option<CallableIdentity>,
    pub parameters: Vec<ValueFact>,
    pub parameters_complete: bool,
    pub outputs: Vec<ValueFact>,
    pub outputs_complete: bool,
    pub variadic_inputs: bool,
    pub variadic_outputs: bool,
    pub captures: Vec<ValueFact>,
    pub captures_complete: bool,
}
