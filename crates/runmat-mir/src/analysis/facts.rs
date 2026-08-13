use runmat_types::ValueFact;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirLocalFact {
    pub value: ValueFact,
}
