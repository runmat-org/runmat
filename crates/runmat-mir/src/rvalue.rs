use crate::{MirCallArg, MirIndexing, MirOperand, MirStmt};
use runmat_hir::{
    CallSyntax, FunctionId, MemberName, OperatorKind, QualifiedName, RequestedOutputCount,
    SymbolName,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirRvalue {
    Use(MirOperand),
    Unary(OperatorKind, MirOperand),
    Binary(MirOperand, OperatorKind, MirOperand),
    ShortCircuit {
        left: MirOperand,
        op: MirShortCircuitOp,
        right_temps: Vec<MirStmt>,
        right: MirOperand,
    },
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
    StructLiteral {
        fields: Vec<(MemberName, MirOperand)>,
    },
    ObjectLiteral {
        class_name: QualifiedName,
        fields: Vec<(MemberName, MirOperand)>,
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
    WorkspaceFirstStaticProperty {
        workspace_name: SymbolName,
        class_name: String,
        property: MemberName,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirShortCircuitOp {
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirAggregateKind {
    Tensor,
    Cell,
}
