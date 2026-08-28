use crate::{DistributedValueId, DistributionScheme, LabCount, ParallelRegionId, ValueFact};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedFact {
    pub id: DistributedValueId,
    pub owner: ParallelRegionId,
    pub scheme: Option<DistributionScheme>,
    pub value: Box<ValueFact>,
    pub materializable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeFact {
    pub owner: ParallelRegionId,
    pub labs: LabCount,
    pub value: Box<ValueFact>,
}
