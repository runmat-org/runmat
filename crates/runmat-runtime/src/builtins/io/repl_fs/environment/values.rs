use runmat_value::{IntValue, NumericScalar, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum EnvironmentValue {
    Set(String),
    Remove,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct EnvironmentValues {
    values: Vec<EnvironmentValue>,
    shape: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ValueError {
    InvalidType,
    InvalidCellElement,
}

impl EnvironmentValues {
    pub(super) fn empty_scalar() -> Self {
        Self {
            values: vec![EnvironmentValue::Set(String::new())],
            shape: None,
        }
    }

    pub(super) fn decode(value: &Value) -> Result<Self, ValueError> {
        match value {
            Value::String(text) => Ok(Self::scalar(string_value(text))),
            Value::CharArray(array) if array.rows == 1 => {
                Ok(Self::scalar(EnvironmentValue::Set(character_row(array))))
            }
            Value::StringArray(array) => Ok(Self {
                values: array.data.iter().map(|text| string_value(text)).collect(),
                shape: Some(array.shape.clone()),
            }),
            Value::Cell(array) => {
                let mut values = Vec::with_capacity(array.data.len());
                for value in &array.data {
                    values.push(Self::decode_scalar(value).ok_or(ValueError::InvalidCellElement)?);
                }
                Ok(Self {
                    values,
                    shape: Some(vec![array.rows, array.cols]),
                })
            }
            value => Self::decode_scalar(value)
                .map(Self::scalar)
                .ok_or(ValueError::InvalidType),
        }
    }

    fn scalar(value: EnvironmentValue) -> Self {
        Self {
            values: vec![value],
            shape: None,
        }
    }

    fn decode_scalar(value: &Value) -> Option<EnvironmentValue> {
        match value {
            Value::String(text) => Some(string_value(text)),
            Value::CharArray(array) if array.rows == 1 => {
                Some(EnvironmentValue::Set(character_row(array)))
            }
            Value::Num(value) => Some(EnvironmentValue::Set(format_float(*value))),
            Value::Int(value) => Some(EnvironmentValue::Set(format_integer(value))),
            Value::Tensor(tensor) if tensor.len() == 1 => tensor
                .host_buffer()
                .value_at(0)
                .map(format_numeric)
                .map(EnvironmentValue::Set),
            _ => None,
        }
    }

    pub(super) fn align(
        &self,
        count: usize,
        name_shape: Option<&[usize]>,
    ) -> Result<Vec<EnvironmentValue>, ()> {
        if self.values.len() == 1 {
            return Ok(vec![self.values[0].clone(); count]);
        }
        if self.values.len() != count || self.shape.as_deref() != name_shape {
            return Err(());
        }
        Ok(self.values.clone())
    }

    pub(super) fn scalar_value(&self) -> Option<EnvironmentValue> {
        (self.values.len() == 1).then(|| self.values[0].clone())
    }
}

fn string_value(text: &str) -> EnvironmentValue {
    if crate::builtins::strings::common::is_missing_string(text) {
        EnvironmentValue::Remove
    } else {
        EnvironmentValue::Set(text.to_string())
    }
}

fn character_row(array: &runmat_value::CharArray) -> String {
    let mut text: String = array.data.iter().collect();
    while text.ends_with(' ') {
        text.pop();
    }
    text
}

fn format_numeric(value: NumericScalar) -> String {
    match value {
        NumericScalar::F64(value) => format_float(value),
        NumericScalar::F32(value) => format_float(f64::from(value)),
        integer => format_integer(
            &integer
                .into_int_value()
                .expect("non-floating numeric scalar is integer"),
        ),
    }
}

fn format_float(value: f64) -> String {
    value.to_string()
}

fn format_integer(value: &IntValue) -> String {
    match value {
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
