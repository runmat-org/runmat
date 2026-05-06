use crate::{MirCall, MirIndexing, MirOperand};
use runmat_hir::{ClassId, FunctionId, OperatorKind};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirRvalue {
    Use(MirOperand),
    Unary(OperatorKind, MirOperand),
    Binary(MirOperand, OperatorKind, MirOperand),
    Range {
        start: MirOperand,
        step: Option<MirOperand>,
        end: MirOperand,
    },
    Call(MirCall),
    Aggregate {
        kind: MirAggregateKind,
        elements: Vec<MirOperand>,
    },
    Index {
        base: MirOperand,
        indexing: MirIndexing,
    },
    Future(FunctionId),
    Spawn(MirOperand),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirAggregateKind {
    Tensor,
    Cell,
    Struct,
    ObjectArray(ClassId),
}
