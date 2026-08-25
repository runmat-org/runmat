use serde::{Deserialize, Serialize};

use super::{CallingConvention, Parameter, PointerMutability};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeScalar {
    Bool,
    Char,
    SignedChar,
    UnsignedChar,
    Short,
    UnsignedShort,
    Int,
    UnsignedInt,
    Long,
    UnsignedLong,
    LongLong,
    UnsignedLongLong,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    Isize,
    Usize,
    F32,
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeType {
    Void,
    Scalar {
        scalar: NativeScalar,
    },
    Pointer {
        pointee: Box<NativeType>,
        mutability: PointerMutability,
    },
    Array {
        element: Box<NativeType>,
        length: usize,
    },
    Structure {
        name: String,
    },
    Enumeration {
        name: String,
        storage: NativeScalar,
    },
    Callback {
        calling_convention: CallingConvention,
        return_type: Box<NativeType>,
        parameters: Vec<Parameter>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructureField {
    pub name: String,
    pub ty: NativeType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructureDefinition {
    pub name: String,
    pub fields: Vec<StructureField>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnumerationVariant {
    pub name: String,
    pub value: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnumerationDefinition {
    pub name: String,
    pub storage: NativeScalar,
    pub variants: Vec<EnumerationVariant>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypeAliasDefinition {
    pub name: String,
    pub target: NativeType,
}
