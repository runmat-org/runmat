use runmat_types::{
    CapabilitySet, EffectSet, ProgramPointId, ProgramSpan, RegionValueId, ValueFact,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssignmentFact {
    Unassigned,
    MaybeAssigned,
    DefinitelyAssigned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramLocalFact {
    pub value: RegionValueId,
    pub assignment: AssignmentFact,
    pub fact: Option<ValueFact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramPointFacts {
    pub point: ProgramPointId,
    pub span: ProgramSpan,
    pub locals: Vec<ProgramLocalFact>,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
}

impl ProgramPointFacts {
    pub fn local(&self, value: RegionValueId) -> Option<&ProgramLocalFact> {
        self.locals
            .binary_search_by_key(&value, |fact| fact.value)
            .ok()
            .map(|index| &self.locals[index])
    }

    pub(crate) fn local_fact_vector(&self, local_count: usize) -> Vec<Option<ValueFact>> {
        let mut values = vec![None; local_count];
        for local in &self.locals {
            if local.value.function != self.point.function {
                continue;
            }
            if let Ok(index) = usize::try_from(local.value.local) {
                if let Some(slot) = values.get_mut(index) {
                    *slot = local.fact.clone();
                }
            }
        }
        values
    }
}
