use crate::{BasicBlock, MirLocalId};
use runmat_hir::{BindingId, FunctionAbi, FunctionId, Span};
use serde::{Deserialize, Serialize};

#[path = "body/expression_region_validation/mod.rs"]
mod expression_region_validation;

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

impl MirBody {
    /// Validates contextual expression regions against their owning body's
    /// local namespace. Region temporaries are private straight-line values:
    /// they may not be defined or observed by the surrounding control-flow
    /// graph.
    pub fn validate_expression_regions(&self) -> Result<(), String> {
        expression_region_validation::validate(self)
    }
}
