use runmat_hir::{FunctionId, SpawnSafetyFact};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpawnSafetySummary {
    pub function: FunctionId,
    pub safety: SpawnSafetyFact,
}
