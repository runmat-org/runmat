use crate::{CapabilitySet, EffectSet};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationEffectFacts {
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
}
