use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumericClass {
    Double,
    Single,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
}

impl NumericClass {
    /// Resolve the canonical source-language name of a built-in numeric class.
    pub fn from_class_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "double" => Some(Self::Double),
            "single" => Some(Self::Single),
            "int8" => Some(Self::Int8),
            "uint8" => Some(Self::UInt8),
            "int16" => Some(Self::Int16),
            "uint16" => Some(Self::UInt16),
            "int32" => Some(Self::Int32),
            "uint32" => Some(Self::UInt32),
            "int64" => Some(Self::Int64),
            "uint64" => Some(Self::UInt64),
            _ => None,
        }
    }

    pub const fn class_name(self) -> &'static str {
        match self {
            Self::Double => "double",
            Self::Single => "single",
            Self::Int8 => "int8",
            Self::UInt8 => "uint8",
            Self::Int16 => "int16",
            Self::UInt16 => "uint16",
            Self::Int32 => "int32",
            Self::UInt32 => "uint32",
            Self::Int64 => "int64",
            Self::UInt64 => "uint64",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumericDomain {
    Real,
    Complex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericFact {
    pub class: NumericClass,
    pub domain: NumericDomain,
}
