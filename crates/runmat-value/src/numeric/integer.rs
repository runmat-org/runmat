use super::view::{cast_exact_unsigned, cast_f64_signed, cast_f64_unsigned, set_integer_element};
use super::*;

/// Exact homogeneous backing storage for MATLAB integer arrays.
///
/// This deliberately stores each class in its native Rust representation so
/// `int64` and `uint64` values never round through `f64` before an
/// integer-aware runtime path consumes them.
#[derive(Debug, Clone, PartialEq)]
pub enum IntegerStorage {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl IntegerStorage {
    /// Returns the logical MATLAB class represented by this exact buffer.
    pub fn numeric_dtype(&self) -> NumericDType {
        match self {
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

    /// Construct a one-element buffer preserving the scalar's MATLAB integer
    /// class.
    pub fn from_scalar(value: IntValue) -> Self {
        match value {
            IntValue::I8(value) => Self::I8(vec![value]),
            IntValue::I16(value) => Self::I16(vec![value]),
            IntValue::I32(value) => Self::I32(vec![value]),
            IntValue::I64(value) => Self::I64(vec![value]),
            IntValue::U8(value) => Self::U8(vec![value]),
            IntValue::U16(value) => Self::U16(vec![value]),
            IntValue::U32(value) => Self::U32(vec![value]),
            IntValue::U64(value) => Self::U64(vec![value]),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::I8(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn class_name(&self) -> &'static str {
        match self {
            Self::I8(_) => "int8",
            Self::I16(_) => "int16",
            Self::I32(_) => "int32",
            Self::I64(_) => "int64",
            Self::U8(_) => "uint8",
            Self::U16(_) => "uint16",
            Self::U32(_) => "uint32",
            Self::U64(_) => "uint64",
        }
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Self::I8(values) => values.iter().map(|&value| value as f64).collect(),
            Self::I16(values) => values.iter().map(|&value| value as f64).collect(),
            Self::I32(values) => values.iter().map(|&value| value as f64).collect(),
            Self::I64(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U8(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U16(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U32(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U64(values) => values.iter().map(|&value| value as f64).collect(),
        }
    }

    /// Returns an exact scalar from this homogeneous buffer.
    pub fn value_at(&self, index: usize) -> Option<IntValue> {
        match self {
            Self::I8(values) => values.get(index).copied().map(IntValue::I8),
            Self::I16(values) => values.get(index).copied().map(IntValue::I16),
            Self::I32(values) => values.get(index).copied().map(IntValue::I32),
            Self::I64(values) => values.get(index).copied().map(IntValue::I64),
            Self::U8(values) => values.get(index).copied().map(IntValue::U8),
            Self::U16(values) => values.get(index).copied().map(IntValue::U16),
            Self::U32(values) => values.get(index).copied().map(IntValue::U32),
            Self::U64(values) => values.get(index).copied().map(IntValue::U64),
        }
    }

    /// Returns exact values in storage order.
    pub fn exact_values(&self) -> Vec<IntValue> {
        (0..self.len())
            .map(|index| {
                self.value_at(index)
                    .expect("integer storage index is valid")
            })
            .collect()
    }

    /// Converts an exact integer scalar to this storage class using the
    /// round-and-saturate assignment semantics used by MATLAB integer arrays.
    pub fn cast_exact_assignment(&self, value: &IntValue) -> IntValue {
        match self {
            Self::I8(_) => IntValue::I8(value.to_i64().clamp(i8::MIN as i64, i8::MAX as i64) as i8),
            Self::I16(_) => {
                IntValue::I16(value.to_i64().clamp(i16::MIN as i64, i16::MAX as i64) as i16)
            }
            Self::I32(_) => {
                IntValue::I32(value.to_i64().clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            }
            Self::I64(_) => IntValue::I64(value.to_i64()),
            Self::U8(_) => IntValue::U8(cast_exact_unsigned(value, u8::MAX as u64) as u8),
            Self::U16(_) => IntValue::U16(cast_exact_unsigned(value, u16::MAX as u64) as u16),
            Self::U32(_) => IntValue::U32(cast_exact_unsigned(value, u32::MAX as u64) as u32),
            Self::U64(_) => IntValue::U64(cast_exact_unsigned(value, u64::MAX)),
        }
    }

    /// Converts a floating scalar to this storage class using the
    /// round-and-saturate assignment semantics used by MATLAB integer arrays.
    pub fn cast_f64_assignment(&self, value: f64) -> IntValue {
        match self {
            Self::I8(_) => {
                IntValue::I8(cast_f64_signed(value, i8::MIN as i64, i8::MAX as i64) as i8)
            }
            Self::I16(_) => {
                IntValue::I16(cast_f64_signed(value, i16::MIN as i64, i16::MAX as i64) as i16)
            }
            Self::I32(_) => {
                IntValue::I32(cast_f64_signed(value, i32::MIN as i64, i32::MAX as i64) as i32)
            }
            Self::I64(_) => IntValue::I64(cast_f64_signed(value, i64::MIN, i64::MAX)),
            Self::U8(_) => IntValue::U8(cast_f64_unsigned(value, u8::MAX as u64) as u8),
            Self::U16(_) => IntValue::U16(cast_f64_unsigned(value, u16::MAX as u64) as u16),
            Self::U32(_) => IntValue::U32(cast_f64_unsigned(value, u32::MAX as u64) as u32),
            Self::U64(_) => IntValue::U64(cast_f64_unsigned(value, u64::MAX)),
        }
    }

    /// Rebuilds this homogeneous storage class from exact values.
    pub fn from_exact_values_like(&self, values: Vec<IntValue>) -> Result<Self, String> {
        macro_rules! rebuild {
            ($variant:ident, $value_variant:ident) => {{
                let mut output = Vec::with_capacity(values.len());
                for value in values {
                    let IntValue::$value_variant(value) = value else {
                        return Err("integer storage class mismatch".into());
                    };
                    output.push(value);
                }
                Ok(Self::$variant(output))
            }};
        }
        match self {
            Self::I8(_) => rebuild!(I8, I8),
            Self::I16(_) => rebuild!(I16, I16),
            Self::I32(_) => rebuild!(I32, I32),
            Self::I64(_) => rebuild!(I64, I64),
            Self::U8(_) => rebuild!(U8, U8),
            Self::U16(_) => rebuild!(U16, U16),
            Self::U32(_) => rebuild!(U32, U32),
            Self::U64(_) => rebuild!(U64, U64),
        }
    }

    /// Applies a structural reorder while preserving this exact storage class.
    pub fn reorder(
        &self,
        reorder: impl Fn(&[IntValue]) -> Result<Vec<IntValue>, String>,
    ) -> Result<Self, String> {
        self.from_exact_values_like(reorder(&self.exact_values())?)
    }

    /// Allocates zeros while preserving this integer class.
    pub fn zeros_like(&self, len: usize) -> Self {
        match self {
            Self::I8(_) => Self::I8(vec![0; len]),
            Self::I16(_) => Self::I16(vec![0; len]),
            Self::I32(_) => Self::I32(vec![0; len]),
            Self::I64(_) => Self::I64(vec![0; len]),
            Self::U8(_) => Self::U8(vec![0; len]),
            Self::U16(_) => Self::U16(vec![0; len]),
            Self::U32(_) => Self::U32(vec![0; len]),
            Self::U64(_) => Self::U64(vec![0; len]),
        }
    }

    /// Allocates ones while preserving this integer class.
    pub fn ones_like(&self, len: usize) -> Self {
        match self {
            Self::I8(_) => Self::I8(vec![1; len]),
            Self::I16(_) => Self::I16(vec![1; len]),
            Self::I32(_) => Self::I32(vec![1; len]),
            Self::I64(_) => Self::I64(vec![1; len]),
            Self::U8(_) => Self::U8(vec![1; len]),
            Self::U16(_) => Self::U16(vec![1; len]),
            Self::U32(_) => Self::U32(vec![1; len]),
            Self::U64(_) => Self::U64(vec![1; len]),
        }
    }

    /// Stores a same-class exact scalar without floating-point conversion.
    pub fn set_value(&mut self, index: usize, value: IntValue) -> Result<(), String> {
        match (self, value) {
            (Self::I8(values), IntValue::I8(value)) => set_integer_element(values, index, value),
            (Self::I16(values), IntValue::I16(value)) => set_integer_element(values, index, value),
            (Self::I32(values), IntValue::I32(value)) => set_integer_element(values, index, value),
            (Self::I64(values), IntValue::I64(value)) => set_integer_element(values, index, value),
            (Self::U8(values), IntValue::U8(value)) => set_integer_element(values, index, value),
            (Self::U16(values), IntValue::U16(value)) => set_integer_element(values, index, value),
            (Self::U32(values), IntValue::U32(value)) => set_integer_element(values, index, value),
            (Self::U64(values), IntValue::U64(value)) => set_integer_element(values, index, value),
            (storage, value) => Err(format!(
                "cannot store {} in {} integer storage",
                value.class_name(),
                storage.class_name()
            )),
        }
    }

    /// Converts and stores an exact scalar without materializing a floating
    /// compatibility value or rebuilding the backing buffer.
    pub fn set_exact_assignment(&mut self, index: usize, value: &IntValue) -> Result<(), String> {
        let value = self.cast_exact_assignment(value);
        self.set_value(index, value)
    }

    /// Converts and stores a floating scalar using integer assignment
    /// semantics without rebuilding the backing buffer.
    pub fn set_f64_assignment(&mut self, index: usize, value: f64) -> Result<(), String> {
        let value = self.cast_f64_assignment(value);
        self.set_value(index, value)
    }

    /// Builds storage with this buffer's class from same-class exact values.
    pub fn from_same_class_values(&self, values: Vec<IntValue>) -> Result<Self, String> {
        macro_rules! collect_values {
            ($variant:ident, $type:ty) => {
                values
                    .into_iter()
                    .map(|value| match value {
                        IntValue::$variant(value) => Ok(value),
                        value => Err(format!(
                            "cannot store {} in {} integer storage",
                            value.class_name(),
                            self.class_name()
                        )),
                    })
                    .collect::<Result<Vec<$type>, String>>()
                    .map(Self::$variant)
            };
        }
        match self {
            Self::I8(_) => collect_values!(I8, i8),
            Self::I16(_) => collect_values!(I16, i16),
            Self::I32(_) => collect_values!(I32, i32),
            Self::I64(_) => collect_values!(I64, i64),
            Self::U8(_) => collect_values!(U8, u8),
            Self::U16(_) => collect_values!(U16, u16),
            Self::U32(_) => collect_values!(U32, u32),
            Self::U64(_) => collect_values!(U64, u64),
        }
    }
}
