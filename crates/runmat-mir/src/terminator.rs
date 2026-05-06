use crate::{BasicBlockId, MirLocalId, MirOperand, MirPlace, MirRvalue};
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
    Switch {
        discr: MirOperand,
        cases: Vec<(MirOperand, BasicBlockId)>,
        otherwise: BasicBlockId,
    },
    For {
        binding: MirLocalId,
        iterable: MirRvalue,
        semantics: LoopIterationSemantics,
        body_block: BasicBlockId,
        exit_block: BasicBlockId,
    },
    TryCatch {
        try_block: BasicBlockId,
        catch_block: BasicBlockId,
    },
    Return(Vec<MirOperand>),
    Await {
        future: MirOperand,
        result: Option<MirPlace>,
        resume: BasicBlockId,
        cleanup: Option<BasicBlockId>,
    },
    Unreachable,
}
