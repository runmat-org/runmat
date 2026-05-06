use crate::{MirCall, MirIndexing, MirOperand};
use runmat_hir::{AggregateKind, FunctionId, OperatorKind};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirRvalue {
    Use(MirOperand),
    Unary(OperatorKind, MirOperand),
    Binary(MirOperand, OperatorKind, MirOperand),
    Call(MirCall),
    Aggregate {
        kind: AggregateKind,
        elements: Vec<MirOperand>,
    },
    Index {
        base: MirOperand,
        indexing: MirIndexing,
    },
    Future(FunctionId),
    Spawn(MirOperand),
}
