use crate::MirLocalId;
use runmat_hir::{CallableIdentity, StringLiteral, SymbolName};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirOperand {
    Local(MirLocalId),
    Constant(MirConstant),
    FunctionHandle(CallableIdentity),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirConstant {
    Number(String),
    String(StringLiteral),
    Symbol(SymbolName),
    Bool(bool),
    EmptyArray,
}
