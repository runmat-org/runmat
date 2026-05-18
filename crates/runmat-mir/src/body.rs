use crate::{BasicBlock, MirLocalId};
use runmat_hir::{BindingId, FunctionAbi, FunctionId, Span};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirBody {
    pub function: FunctionId,
    pub abi: FunctionAbi,
    pub locals: Vec<MirLocal>,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirLocal {
    pub id: MirLocalId,
    pub binding: Option<BindingId>,
    pub kind: MirLocalKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirLocalKind {
    Parameter,
    Output,
    Binding,
    Temporary,
    Capture,
}
