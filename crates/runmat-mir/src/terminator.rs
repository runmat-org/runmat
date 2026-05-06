use crate::{BasicBlockId, MirLocalId, MirOperand, MirRvalue};
use runmat_hir::{LoopIterationSemantics, Span};
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
    For {
        binding: MirLocalId,
        iterable: MirRvalue,
        semantics: LoopIterationSemantics,
        body_block: BasicBlockId,
        exit_block: BasicBlockId,
    },
    Return(Vec<MirOperand>),
    Await {
        future: MirOperand,
        resume: BasicBlockId,
        cleanup: Option<BasicBlockId>,
    },
    Unreachable,
}
