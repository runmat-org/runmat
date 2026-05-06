use crate::{BasicBlock, MirLocalId, MirSourceMap};
use runmat_hir::{BindingId, FunctionId, Span};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirBody {
    pub function: FunctionId,
    pub locals: Vec<MirLocal>,
    pub blocks: Vec<BasicBlock>,
    pub source_map: MirSourceMap,
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
    Binding,
    Temporary,
    Capture,
}
