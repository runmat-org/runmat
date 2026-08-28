use crate::{
    DistributedValueId, DistributionScheme, LabCount, ParallelRegionId, ProgramFunctionId,
    ValueFact,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum DistributedOwner {
    Client(ProgramFunctionId),
    Region(ParallelRegionId),
}

impl DistributedOwner {
    pub const fn function(self) -> ProgramFunctionId {
        match self {
            Self::Client(function) => function,
            Self::Region(region) => region.0.function,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedFact {
    pub id: DistributedValueId,
    pub owner: DistributedOwner,
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
