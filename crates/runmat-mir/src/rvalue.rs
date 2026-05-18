use crate::{MirCallArg, MirIndexing, MirOperand};
use runmat_hir::{
    CallSyntax, FunctionId, MemberName, OperatorKind, QualifiedName, RequestedOutputCount,
};
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
    Call(crate::MirCall),
    Aggregate {
        kind: MirAggregateKind,
        rows: usize,
        cols: usize,
        elements: Vec<MirOperand>,
    },
    Index {
        base: MirOperand,
        indexing: MirIndexing,
    },
    Member {
        base: MirOperand,
        member: MemberName,
    },
    DynamicMember {
        base: MirOperand,
        member: MirOperand,
    },
    MetaClass(QualifiedName),
    Colon,
    End,
    Future {
        function: FunctionId,
        args: Vec<MirCallArg>,
        syntax: CallSyntax,
        requested_outputs: RequestedOutputCount,
    },
    Spawn(MirOperand),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirAggregateKind {
    Tensor,
    Cell,
}
