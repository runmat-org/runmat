use serde::{Deserialize, Serialize};

use crate::{IntegerElementType, ProviderPrecision};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderElementType {
    Logical,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    ComplexF32,
    ComplexF64,
}

impl From<ProviderPrecision> for ProviderElementType {
    fn from(value: ProviderPrecision) -> Self {
        match value {
            ProviderPrecision::F32 => Self::F32,
            ProviderPrecision::F64 => Self::F64,
        }
    }
}

impl From<IntegerElementType> for ProviderElementType {
    fn from(value: IntegerElementType) -> Self {
        match value {
            IntegerElementType::I8 => Self::I8,
            IntegerElementType::I16 => Self::I16,
            IntegerElementType::I32 => Self::I32,
            IntegerElementType::I64 => Self::I64,
            IntegerElementType::U8 => Self::U8,
            IntegerElementType::U16 => Self::U16,
            IntegerElementType::U32 => Self::U32,
            IntegerElementType::U64 => Self::U64,
        }
    }
}

impl ProviderElementType {
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::Logical | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
            Self::ComplexF32 => 8,
            Self::ComplexF64 => 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderStorage {
    DenseReal,
    DenseComplexInterleaved,
    Sparse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderLayout {
    ColumnMajorContiguous,
    Strided,
    Opaque,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderResidency {
    Host,
    Device,
    Mirrored,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderRepresentation {
    pub element_type: ProviderElementType,
    pub storage: ProviderStorage,
    pub layout: ProviderLayout,
    pub shape: Vec<u64>,
    pub residency: ProviderResidency,
}

impl ProviderRepresentation {
    pub fn checked_element_count(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
    }

    pub fn checked_byte_len(&self) -> Option<u64> {
        self.checked_element_count()?
            .checked_mul(self.element_type.byte_width())
    }
}
