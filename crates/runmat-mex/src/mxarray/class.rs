use runmat_value::NumericDType;

/// Public `mxClassID` values used by the C Matrix API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum MxClassId {
    Unknown = 0,
    Cell = 1,
    Struct = 2,
    Logical = 3,
    Char = 4,
    Void = 5,
    Double = 6,
    Single = 7,
    Int8 = 8,
    Uint8 = 9,
    Int16 = 10,
    Uint16 = 11,
    Int32 = 12,
    Uint32 = 13,
    Int64 = 14,
    Uint64 = 15,
    Function = 16,
    Opaque = 17,
    Object = 18,
}

impl MxClassId {
    pub const fn from_numeric_dtype(dtype: NumericDType) -> Self {
        match dtype {
            NumericDType::F64 => Self::Double,
            NumericDType::F32 => Self::Single,
            NumericDType::I8 => Self::Int8,
            NumericDType::I16 => Self::Int16,
            NumericDType::I32 => Self::Int32,
            NumericDType::I64 => Self::Int64,
            NumericDType::U8 => Self::Uint8,
            NumericDType::U16 => Self::Uint16,
            NumericDType::U32 => Self::Uint32,
            NumericDType::U64 => Self::Uint64,
        }
    }

    pub const fn numeric_dtype(self) -> Option<NumericDType> {
        match self {
            Self::Double => Some(NumericDType::F64),
            Self::Single => Some(NumericDType::F32),
            Self::Int8 => Some(NumericDType::I8),
            Self::Int16 => Some(NumericDType::I16),
            Self::Int32 => Some(NumericDType::I32),
            Self::Int64 => Some(NumericDType::I64),
            Self::Uint8 => Some(NumericDType::U8),
            Self::Uint16 => Some(NumericDType::U16),
            Self::Uint32 => Some(NumericDType::U32),
            Self::Uint64 => Some(NumericDType::U64),
            _ => None,
        }
    }

    pub const fn is_numeric(self) -> bool {
        self.numeric_dtype().is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum MxComplexity {
    Real = 0,
    Complex = 1,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_classes_round_trip_without_widening() {
        for dtype in [
            NumericDType::F64,
            NumericDType::F32,
            NumericDType::I8,
            NumericDType::I16,
            NumericDType::I32,
            NumericDType::I64,
            NumericDType::U8,
            NumericDType::U16,
            NumericDType::U32,
            NumericDType::U64,
        ] {
            assert_eq!(
                MxClassId::from_numeric_dtype(dtype).numeric_dtype(),
                Some(dtype)
            );
        }
    }
}
