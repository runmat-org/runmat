use runmat_types::ValueFact;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirLocalFact {
    pub value: ValueFact,
}
