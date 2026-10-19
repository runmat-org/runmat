use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OperationDistribution {
    Local,
    HostOnly,
    PartitionLocal,
    DeterministicReduction,
    Collective,
    CheckedByLowering,
}
