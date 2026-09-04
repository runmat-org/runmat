use runmat_value::{CellArray, IntValue, LogicalArray, NumericStorage, StringArray, Tensor, Value};

use crate::BuiltinResult;

use super::{error, plan::Repetition};

pub(super) enum CombinationColumn {
    Numeric(NumericStorage),
    Logical(Vec<u8>),
    String(Vec<String>),
    Cell(Vec<Value>),
}

impl CombinationColumn {
    pub(super) fn from_value(value: Value) -> BuiltinResult<Self> {
        Ok(match value {
            Value::Tensor(tensor) => {
                Self::Numeric(tensor.into_numeric_storage().map_err(error::internal)?)
            }
            Value::Int(value) => Self::Numeric(integer_scalar(value)),
            Value::Num(value) => Self::Numeric(NumericStorage::F64(vec![value])),
            Value::Bool(value) => Self::Logical(vec![u8::from(value)]),
            Value::LogicalArray(array) => Self::Logical(array.data.to_vec()),
            Value::String(value) => Self::String(vec![value]),
            Value::StringArray(array) => Self::String(array.data),
            Value::CharArray(array) if array.rows == 1 => Self::String(
                array
                    .data
                    .into_iter()
                    .map(|value| value.to_string())
                    .collect(),
            ),
            Value::Cell(array) => Self::Cell(array.data),
            other => Self::Cell(vec![other]),
        })
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Numeric(values) => values.len(),
            Self::Logical(values) => values.len(),
            Self::String(values) => values.len(),
            Self::Cell(values) => values.len(),
        }
    }

    pub(super) fn materialize(self, rows: usize, repetition: Repetition) -> BuiltinResult<Value> {
        match self {
            Self::Numeric(storage) => materialize_numeric(storage, rows, repetition),
            Self::Logical(values) => {
                LogicalArray::new(repeat_values(&values, repetition, rows), vec![rows, 1])
                    .map(Value::LogicalArray)
                    .map_err(error::internal)
            }
            Self::String(values) => {
                StringArray::new(repeat_values(&values, repetition, rows), vec![rows, 1])
                    .map(Value::StringArray)
                    .map_err(error::internal)
            }
            Self::Cell(values) => CellArray::new(repeat_values(&values, repetition, rows), rows, 1)
                .map(Value::Cell)
                .map_err(error::internal),
        }
    }
}

fn materialize_numeric(
    storage: NumericStorage,
    rows: usize,
    repetition: Repetition,
) -> BuiltinResult<Value> {
    macro_rules! repeated {
        ($values:expr, $variant:ident) => {
            NumericStorage::$variant(repeat_values(&$values, repetition, rows))
        };
    }
    let output = match storage {
        NumericStorage::F64(values) => repeated!(values, F64),
        NumericStorage::F32(values) => repeated!(values, F32),
        NumericStorage::I8(values) => repeated!(values, I8),
        NumericStorage::I16(values) => repeated!(values, I16),
        NumericStorage::I32(values) => repeated!(values, I32),
        NumericStorage::I64(values) => repeated!(values, I64),
        NumericStorage::U8(values) => repeated!(values, U8),
        NumericStorage::U16(values) => repeated!(values, U16),
        NumericStorage::U32(values) => repeated!(values, U32),
        NumericStorage::U64(values) => repeated!(values, U64),
    };
    Tensor::from_numeric_storage(output, vec![rows, 1])
        .map(Value::Tensor)
        .map_err(error::internal)
}

fn repeat_values<T: Clone>(values: &[T], repetition: Repetition, rows: usize) -> Vec<T> {
    let mut output = Vec::with_capacity(rows);
    for _ in 0..repetition.outer {
        for value in values {
            output.extend(std::iter::repeat_n(value.clone(), repetition.inner));
        }
    }
    output
}

fn integer_scalar(value: IntValue) -> NumericStorage {
    match value {
        IntValue::I8(value) => NumericStorage::I8(vec![value]),
        IntValue::I16(value) => NumericStorage::I16(vec![value]),
        IntValue::I32(value) => NumericStorage::I32(vec![value]),
        IntValue::I64(value) => NumericStorage::I64(vec![value]),
        IntValue::U8(value) => NumericStorage::U8(vec![value]),
        IntValue::U16(value) => NumericStorage::U16(vec![value]),
        IntValue::U32(value) => NumericStorage::U32(vec![value]),
        IntValue::U64(value) => NumericStorage::U64(vec![value]),
    }
}
