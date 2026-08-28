use crate::MirOperand;
use runmat_types::{DistributedValueId, DistributionScheme, ParallelRegionId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirDistributedOp {
    Create {
        id: DistributedValueId,
        owner: ParallelRegionId,
        input: MirOperand,
        scheme: DistributionScheme,
    },
    LocalPart {
        value: DistributedValueId,
    },
    Materialize {
        value: DistributedValueId,
    },
    Redistribute {
        value: DistributedValueId,
        scheme: DistributionScheme,
    },
}
