use crate::{BasicBlockId, MirLocalId};
use runmat_hir::{BindingId, ClassId, ExprId, FunctionId, ModuleId, Span, StmtId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MirSourceMap {
    pub function: Option<FunctionId>,
    pub module: Option<ModuleId>,
    pub enclosing_class: Option<ClassId>,
    pub statements: Vec<MirSourceRecord>,
    pub locals: Vec<MirLocalSource>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirSourceRecord {
    pub block: BasicBlockId,
    pub stmt: Option<StmtId>,
    pub expr: Option<ExprId>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirLocalSource {
    pub local: MirLocalId,
    pub binding: Option<BindingId>,
    pub expr: Option<ExprId>,
    pub span: Span,
}
