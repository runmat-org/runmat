use crate::{MirLocalId, MirTempId};
use runmat_hir::{FunctionHandleTarget, StringLiteral, SymbolName};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirOperand {
    Local(MirLocalId),
    Temp(MirTempId),
    Constant(MirConstant),
    FunctionHandle(FunctionHandleTarget),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirConstant {
    Number(String),
    String(StringLiteral),
    Symbol(SymbolName),
    Bool(bool),
    EmptyArray,
}
