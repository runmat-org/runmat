use super::OperationTypePattern;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OperationParameterArity {
    Required,
    Optional,
    Variadic,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OperationDefaultValue {
    Logical(bool),
    Integer(i64),
    FloatBits(u64),
    String(String),
    EnumCase(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationParameter {
    pub name: String,
    pub value: OperationTypePattern,
    pub arity: OperationParameterArity,
    pub default: Option<OperationDefaultValue>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationSignature {
    pub inputs: Vec<OperationParameter>,
    pub outputs: Vec<OperationParameter>,
}
