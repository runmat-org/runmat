use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumericDType {
    F64,
    F32,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl NumericDType {
    pub fn class_identity(self) -> runmat_types::ClassIdentity {
        use runmat_types::standard;

        match self {
            NumericDType::F64 => standard::DOUBLE.owned(),
            NumericDType::F32 => standard::SINGLE.owned(),
            NumericDType::I8 => standard::INT8.owned(),
            NumericDType::I16 => standard::INT16.owned(),
            NumericDType::I32 => standard::INT32.owned(),
            NumericDType::I64 => standard::INT64.owned(),
            NumericDType::U8 => standard::UINT8.owned(),
            NumericDType::U16 => standard::UINT16.owned(),
            NumericDType::U32 => standard::UINT32.owned(),
            NumericDType::U64 => standard::UINT64.owned(),
        }
    }

    pub fn class_name(self) -> &'static str {
        match self {
            NumericDType::F64 => "double",
            NumericDType::F32 => "single",
            NumericDType::I8 => "int8",
            NumericDType::I16 => "int16",
            NumericDType::I32 => "int32",
            NumericDType::I64 => "int64",
            NumericDType::U8 => "uint8",
            NumericDType::U16 => "uint16",
            NumericDType::U32 => "uint32",
            NumericDType::U64 => "uint64",
        }
    }

    pub fn byte_size(self) -> usize {
        match self {
            NumericDType::F64 => 8,
            NumericDType::F32 => 4,
            NumericDType::I8 => 1,
            NumericDType::I16 => 2,
            NumericDType::I32 => 4,
            NumericDType::I64 => 8,
            NumericDType::U8 => 1,
            NumericDType::U16 => 2,
            NumericDType::U32 => 4,
            NumericDType::U64 => 8,
        }
    }
}
