use crate::{MirLocalId, MirTempId};
use runmat_hir::{FunctionHandleTarget, StringLiteral};
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
    Bool(bool),
    EmptyArray,
}
