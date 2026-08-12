use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

impl IntValue {
    pub fn to_i64(&self) -> i64 {
        match self {
            IntValue::I8(v) => *v as i64,
            IntValue::I16(v) => *v as i64,
            IntValue::I32(v) => *v as i64,
            IntValue::I64(v) => *v,
            IntValue::U8(v) => *v as i64,
            IntValue::U16(v) => *v as i64,
            IntValue::U32(v) => *v as i64,
            IntValue::U64(v) => {
                if *v > i64::MAX as u64 {
                    i64::MAX
                } else {
                    *v as i64
                }
            }
        }
    }

    /// Returns the signed representation when it is exactly representable.
    ///
    /// Unlike [`Self::to_i64`], this never saturates an out-of-range `uint64`.
    pub fn try_to_i64(&self) -> Option<i64> {
        match self {
            IntValue::I8(v) => Some(*v as i64),
            IntValue::I16(v) => Some(*v as i64),
            IntValue::I32(v) => Some(*v as i64),
            IntValue::I64(v) => Some(*v),
            IntValue::U8(v) => Some(*v as i64),
            IntValue::U16(v) => Some(*v as i64),
            IntValue::U32(v) => Some(*v as i64),
            IntValue::U64(v) => i64::try_from(*v).ok(),
        }
    }

    /// Returns the `int32` representation when it is exactly representable.
    pub fn try_to_i32(&self) -> Option<i32> {
        self.try_to_i64()
            .and_then(|value| i32::try_from(value).ok())
    }

    /// Returns the platform signed representation when it is exactly
    /// representable.
    ///
    /// This is intended for signed offsets and shifts. In particular, it
    /// rejects `uint64` values above `int64::MAX` instead of saturating them.
    pub fn try_to_isize(&self) -> Option<isize> {
        self.try_to_i64()
            .and_then(|value| isize::try_from(value).ok())
    }

    /// Returns the unsigned representation when it is exactly representable.
    pub fn try_to_u64(&self) -> Option<u64> {
        match self {
            IntValue::I8(v) => u64::try_from(*v).ok(),
            IntValue::I16(v) => u64::try_from(*v).ok(),
            IntValue::I32(v) => u64::try_from(*v).ok(),
            IntValue::I64(v) => u64::try_from(*v).ok(),
            IntValue::U8(v) => Some(*v as u64),
            IntValue::U16(v) => Some(*v as u64),
            IntValue::U32(v) => Some(*v as u64),
            IntValue::U64(v) => Some(*v),
        }
    }

    /// Returns the platform dimension representation when it is exactly
    /// representable and non-negative.
    pub fn try_to_usize(&self) -> Option<usize> {
        self.try_to_u64()
            .and_then(|value| usize::try_from(value).ok())
    }
    pub fn to_f64(&self) -> f64 {
        match self {
            // `uint64` has a wider positive range than `int64`. Converting it
            // through `to_i64` incorrectly clamps every value above i64::MAX
            // before normal IEEE-754 rounding can occur.
            IntValue::U64(value) => *value as f64,
            _ => self.to_i64() as f64,
        }
    }
    pub fn is_zero(&self) -> bool {
        match self {
            IntValue::I8(value) => *value == 0,
            IntValue::I16(value) => *value == 0,
            IntValue::I32(value) => *value == 0,
            IntValue::I64(value) => *value == 0,
            IntValue::U8(value) => *value == 0,
            IntValue::U16(value) => *value == 0,
            IntValue::U32(value) => *value == 0,
            IntValue::U64(value) => *value == 0,
        }
    }
    pub fn class_name(&self) -> &'static str {
        match self {
            IntValue::I8(_) => "int8",
            IntValue::I16(_) => "int16",
            IntValue::I32(_) => "int32",
            IntValue::I64(_) => "int64",
            IntValue::U8(_) => "uint8",
            IntValue::U16(_) => "uint16",
            IntValue::U32(_) => "uint32",
            IntValue::U64(_) => "uint64",
        }
    }

    /// Returns the exact base-10 representation without narrowing through a
    /// signed integer or floating-point compatibility path.
    pub fn decimal_string(&self) -> String {
        match self {
            IntValue::I8(value) => value.to_string(),
            IntValue::I16(value) => value.to_string(),
            IntValue::I32(value) => value.to_string(),
            IntValue::I64(value) => value.to_string(),
            IntValue::U8(value) => value.to_string(),
            IntValue::U16(value) => value.to_string(),
            IntValue::U32(value) => value.to_string(),
            IntValue::U64(value) => value.to_string(),
        }
    }

    /// Add two values of the same MATLAB integer class with saturating
    /// semantics. Sparse triplet construction uses this for duplicate entries.
    pub fn saturating_add(&self, rhs: &Self) -> Result<Self, String> {
        match (self, rhs) {
            (Self::I8(lhs), Self::I8(rhs)) => Ok(Self::I8(lhs.saturating_add(*rhs))),
            (Self::I16(lhs), Self::I16(rhs)) => Ok(Self::I16(lhs.saturating_add(*rhs))),
            (Self::I32(lhs), Self::I32(rhs)) => Ok(Self::I32(lhs.saturating_add(*rhs))),
            (Self::I64(lhs), Self::I64(rhs)) => Ok(Self::I64(lhs.saturating_add(*rhs))),
            (Self::U8(lhs), Self::U8(rhs)) => Ok(Self::U8(lhs.saturating_add(*rhs))),
            (Self::U16(lhs), Self::U16(rhs)) => Ok(Self::U16(lhs.saturating_add(*rhs))),
            (Self::U32(lhs), Self::U32(rhs)) => Ok(Self::U32(lhs.saturating_add(*rhs))),
            (Self::U64(lhs), Self::U64(rhs)) => Ok(Self::U64(lhs.saturating_add(*rhs))),
            (lhs, rhs) => Err(format!(
                "cannot add {} and {} integer values",
                lhs.class_name(),
                rhs.class_name()
            )),
        }
    }
}

/// One exact scalar read from or written to [`NumericStorage`].
///
/// The variant is part of the value: extracting an integer does not first
/// route it through a floating-point representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumericScalar {
    F64(f64),
    F32(f32),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

impl From<IntValue> for NumericScalar {
    fn from(value: IntValue) -> Self {
        match value {
            IntValue::I8(value) => Self::I8(value),
            IntValue::I16(value) => Self::I16(value),
            IntValue::I32(value) => Self::I32(value),
            IntValue::I64(value) => Self::I64(value),
            IntValue::U8(value) => Self::U8(value),
            IntValue::U16(value) => Self::U16(value),
            IntValue::U32(value) => Self::U32(value),
            IntValue::U64(value) => Self::U64(value),
        }
    }
}

impl NumericScalar {
    pub fn into_int_value(self) -> Option<IntValue> {
        match self {
            Self::I8(value) => Some(IntValue::I8(value)),
            Self::I16(value) => Some(IntValue::I16(value)),
            Self::I32(value) => Some(IntValue::I32(value)),
            Self::I64(value) => Some(IntValue::I64(value)),
            Self::U8(value) => Some(IntValue::U8(value)),
            Self::U16(value) => Some(IntValue::U16(value)),
            Self::U32(value) => Some(IntValue::U32(value)),
            Self::U64(value) => Some(IntValue::U64(value)),
            Self::F64(_) | Self::F32(_) => None,
        }
    }
}

impl NumericScalar {
    pub fn numeric_dtype(self) -> NumericDType {
        match self {
            Self::F64(_) => NumericDType::F64,
            Self::F32(_) => NumericDType::F32,
            Self::I8(_) => NumericDType::I8,
            Self::I16(_) => NumericDType::I16,
            Self::I32(_) => NumericDType::I32,
            Self::I64(_) => NumericDType::I64,
            Self::U8(_) => NumericDType::U8,
            Self::U16(_) => NumericDType::U16,
            Self::U32(_) => NumericDType::U32,
            Self::U64(_) => NumericDType::U64,
        }
    }

    pub fn class_name(self) -> &'static str {
        self.numeric_dtype().class_name()
    }

    pub fn is_zero(self) -> bool {
        match self {
            Self::F64(value) => value == 0.0,
            Self::F32(value) => value == 0.0,
            Self::I8(value) => value == 0,
            Self::I16(value) => value == 0,
            Self::I32(value) => value == 0,
            Self::I64(value) => value == 0,
            Self::U8(value) => value == 0,
            Self::U16(value) => value == 0,
            Self::U32(value) => value == 0,
            Self::U64(value) => value == 0,
        }
    }

    pub fn is_finite(self) -> bool {
        match self {
            Self::F64(value) => value.is_finite(),
            Self::F32(value) => value.is_finite(),
            Self::I8(_)
            | Self::I16(_)
            | Self::I32(_)
            | Self::I64(_)
            | Self::U8(_)
            | Self::U16(_)
            | Self::U32(_)
            | Self::U64(_) => true,
        }
    }

    /// Explicitly materializes this scalar in the `f64` computation domain.
    ///
    /// Integer values outside the exact binary64 range may lose precision.
    pub fn materialize_f64(self) -> f64 {
        match self {
            Self::F64(value) => value,
            Self::F32(value) => f64::from(value),
            Self::I8(value) => f64::from(value),
            Self::I16(value) => f64::from(value),
            Self::I32(value) => f64::from(value),
            Self::I64(value) => value as f64,
            Self::U8(value) => f64::from(value),
            Self::U16(value) => f64::from(value),
            Self::U32(value) => f64::from(value),
            Self::U64(value) => value as f64,
        }
    }
}
