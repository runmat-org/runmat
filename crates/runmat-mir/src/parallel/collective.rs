use crate::MirOperand;
use runmat_types::{CollectiveId, LabRank, OperatorKind};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirCollectiveOp {
    Barrier {
        id: CollectiveId,
    },
    Broadcast {
        id: CollectiveId,
        input: MirOperand,
        root: LabRank,
    },
    Gather {
        id: CollectiveId,
        input: MirOperand,
        root: LabRank,
    },
    Scatter {
        id: CollectiveId,
        input: MirOperand,
        root: LabRank,
    },
    AllGather {
        id: CollectiveId,
        input: MirOperand,
    },
    Reduce {
        id: CollectiveId,
        input: MirOperand,
        root: LabRank,
        operator: OperatorKind,
    },
    AllReduce {
        id: CollectiveId,
        input: MirOperand,
        operator: OperatorKind,
    },
    Send {
        id: CollectiveId,
        input: MirOperand,
        peer: LabRank,
    },
    Receive {
        id: CollectiveId,
        peer: LabRank,
    },
}
