use crate::{BasicBlockId, MirOperand};
use runmat_hir::Span;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirTerminator {
    pub kind: MirTerminatorKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirTerminatorKind {
    Goto(BasicBlockId),
    Branch {
        cond: MirOperand,
        then_block: BasicBlockId,
        else_block: BasicBlockId,
    },
    Return(Vec<MirOperand>),
    Await {
        future: MirOperand,
        resume: BasicBlockId,
        cleanup: Option<BasicBlockId>,
    },
    Unreachable,
}
