use runmat_hir::{BindingId, FunctionId, RequestedOutputCount, SpawnSafetyFact, TypeFact};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::{AccelEligibilityFact, EffectSummary, FusibilityFact, ParallelSafetyFact};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionSummary {
    pub function: FunctionId,
    pub outputs: Vec<TypeFact>,
    pub requested_output_sensitive: Vec<(RequestedOutputCount, Vec<TypeFact>)>,
    pub effects: EffectSummary,
    pub reads_captures: BTreeSet<BindingId>,
    pub writes_captures: BTreeSet<BindingId>,
    pub spawn_safety: SpawnSafetyFact,
    pub fusibility: FusibilityFact,
    pub parallel_safety: ParallelSafetyFact,
    pub accel_eligibility: AccelEligibilityFact,
}
