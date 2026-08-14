use runmat_types::{ProgramPointId, RegionContract, RegionValueId};
use serde::{Deserialize, Serialize};

/// Analysis-only detail associated with one portable region contract.
///
/// The contract is the cross-product execution boundary. Operation and future
/// consumer points remain analysis evidence and do not grow the stable R06A
/// region schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionAnalysis {
    pub contract: RegionContract,
    pub operations: Vec<ProgramPointId>,
    pub future_consumers: Vec<RegionFutureConsumer>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionFutureConsumer {
    pub value: RegionValueId,
    pub point: ProgramPointId,
}
