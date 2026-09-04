use runmat_value::{IntValue, NumericStorage, StringArray, Tensor, Value};

use crate::BuiltinResult;

use super::error;

pub(super) enum Labels {
    Text(Vec<String>),
    Numeric(NumericStorage),
}

impl Labels {
    pub(super) fn from_value(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(value) => Ok(Self::Numeric(NumericStorage::F64(vec![*value]))),
            Value::Int(value) => Ok(Self::Numeric(integer_scalar(value))),
            Value::Tensor(tensor) => tensor
                .clone()
                .into_numeric_storage()
                .map(Self::Numeric)
                .map_err(error::internal),
            _ => text_values(value).map(Self::Text),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Text(values) => values.len(),
            Self::Numeric(values) => values.len(),
        }
    }

    pub(super) fn materialize(
        self,
        bins: &[Option<usize>],
        shape: Vec<usize>,
    ) -> BuiltinResult<Value> {
        match self {
            Self::Text(labels) => {
                let data = bins
                    .iter()
                    .map(|bin| {
                        bin.and_then(|index| labels.get(index - 1).cloned())
                            .unwrap_or_default()
                    })
                    .collect();
                StringArray::new(data, shape)
                    .map(Value::StringArray)
                    .map_err(error::internal)
            }
            Self::Numeric(labels) => {
                let mut output = missing_numeric(&labels, bins.len());
                for (position, bin) in bins.iter().enumerate() {
                    if let Some(index) = bin {
                        let label = labels.value_at(index - 1).ok_or_else(|| {
                            error::internal("replacement value index is out of bounds")
                        })?;
                        output.set_value(position, label).map_err(error::internal)?;
                    }
                }
                Tensor::from_numeric_storage(output, shape)
                    .map(Value::Tensor)
                    .map_err(error::internal)
            }
        }
    }
}

fn integer_scalar(value: &IntValue) -> NumericStorage {
    match value {
        IntValue::I8(value) => NumericStorage::I8(vec![*value]),
        IntValue::I16(value) => NumericStorage::I16(vec![*value]),
        IntValue::I32(value) => NumericStorage::I32(vec![*value]),
        IntValue::I64(value) => NumericStorage::I64(vec![*value]),
        IntValue::U8(value) => NumericStorage::U8(vec![*value]),
        IntValue::U16(value) => NumericStorage::U16(vec![*value]),
        IntValue::U32(value) => NumericStorage::U32(vec![*value]),
        IntValue::U64(value) => NumericStorage::U64(vec![*value]),
    }
}

fn missing_numeric(labels: &NumericStorage, len: usize) -> NumericStorage {
    match labels {
        NumericStorage::F64(_) => NumericStorage::F64(vec![f64::NAN; len]),
        NumericStorage::F32(_) => NumericStorage::F32(vec![f32::NAN; len]),
        _ => labels.zeros_like(len),
    }
}

fn text_values(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(value) => Ok(vec![value.clone()]),
        Value::StringArray(value) => Ok(value.data.clone()),
        Value::CharArray(value) if value.rows <= 1 => Ok(vec![value.data.iter().collect()]),
        Value::Cell(value) => value
            .data
            .iter()
            .map(|item| match item {
                Value::String(text) => Ok(text.clone()),
                Value::CharArray(chars) if chars.rows <= 1 => Ok(chars.data.iter().collect()),
                other => Err(error::invalid(format!(
                    "discretize: text replacement cell contains {other:?}"
                ))),
            })
            .collect(),
        other => Err(error::invalid(format!(
            "discretize: replacement values must be real numeric or text, got {other:?}"
        ))),
    }
}
