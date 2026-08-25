use serde::{Deserialize, Serialize};

use super::{CallingConvention, NativeType, ParameterDirection, PointerOwnership};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameter {
    pub name: String,
    pub ty: NativeType,
    #[serde(default = "input_direction")]
    pub direction: ParameterDirection,
    #[serde(default = "borrowed_ownership")]
    pub ownership: PointerOwnership,
    #[serde(default)]
    pub nullable: bool,
}

fn input_direction() -> ParameterDirection {
    ParameterDirection::Input
}

fn borrowed_ownership() -> PointerOwnership {
    PointerOwnership::Borrowed
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SymbolPrototype {
    pub name: String,
    pub exported_name: String,
    #[serde(default = "c_calling_convention")]
    pub calling_convention: CallingConvention,
    pub return_type: NativeType,
    #[serde(default)]
    pub return_ownership: Option<PointerOwnership>,
    #[serde(default)]
    pub return_nullable: bool,
    #[serde(default)]
    pub parameters: Vec<Parameter>,
    #[serde(default)]
    pub variadic: bool,
}

fn c_calling_convention() -> CallingConvention {
    CallingConvention::C
}
