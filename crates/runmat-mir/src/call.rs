use crate::MirOperand;
use runmat_hir::{CallSyntax, HirCallableRef, RequestedOutputCount};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirCall {
    pub callee: HirCallableRef,
    pub args: Vec<MirOperand>,
    pub syntax: CallSyntax,
    pub requested_outputs: RequestedOutputCount,
}
