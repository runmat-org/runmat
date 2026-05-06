use crate::MirOperand;
use runmat_hir::{CallSyntax, HirCallableRef, RequestedOutputCount};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirCall {
    pub callee: HirCallableRef,
    pub args: Vec<MirCallArg>,
    pub syntax: CallSyntax,
    pub requested_outputs: RequestedOutputCount,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirCallArg {
    Single(MirOperand),
    Expansion(MirOperand),
}

impl MirCallArg {
    pub fn operand(&self) -> &MirOperand {
        match self {
            MirCallArg::Single(operand) | MirCallArg::Expansion(operand) => operand,
        }
    }
}
