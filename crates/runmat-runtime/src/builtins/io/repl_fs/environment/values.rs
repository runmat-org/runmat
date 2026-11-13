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
    Encoding(runmat_value::Utf16DecodeError),
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
            Value::CharArray(array)
                if array.rows == 1
                    && crate::builtins::common::shape::effective_rank(&array.shape) <= 2 =>
            {
                Ok(Self::scalar(EnvironmentValue::Set(character_row(array)?)))
            }
            Value::StringArray(array) => Ok(Self {
                values: array.data().iter().map(|text| string_value(text)).collect(),
                shape: Some(array.shape.as_slice().to_vec()),
            }),
            Value::Cell(array) => {
                let mut values = Vec::with_capacity(array.data().len());
                for value in array.iter_column_major() {
                    values.push(Self::decode_scalar(value)?.ok_or(ValueError::InvalidCellElement)?);
                }
                Ok(Self {
                    values,
                    shape: Some(array.shape.as_slice().to_vec()),
                })
            }
            value => Self::decode_scalar(value)?
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

    fn decode_scalar(value: &Value) -> Result<Option<EnvironmentValue>, ValueError> {
        Ok(match value {
            Value::String(text) => Some(string_value(text)),
            Value::CharArray(array)
                if array.rows == 1
                    && crate::builtins::common::shape::effective_rank(&array.shape) <= 2 =>
            {
                Some(EnvironmentValue::Set(character_row(array)?))
            }
            Value::Num(value) => Some(EnvironmentValue::Set(format_float(*value))),
            Value::Int(value) => Some(EnvironmentValue::Set(format_integer(value))),
            Value::Tensor(tensor) if tensor.len() == 1 => tensor
                .host_buffer()
                .value_at(0)
                .map(format_numeric)
                .map(EnvironmentValue::Set),
            _ => None,
        })
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

fn string_value(text: &runmat_value::StringElement) -> EnvironmentValue {
    match text.as_str() {
        Some(text) => EnvironmentValue::Set(text.to_owned()),
        None => EnvironmentValue::Remove,
    }
}

fn character_row(array: &runmat_value::CharArray) -> Result<String, ValueError> {
    let mut text =
        runmat_value::decode_utf16_code_units(array.data()).map_err(ValueError::Encoding)?;
    while text.ends_with(' ') {
        text.pop();
    }
    Ok(text)
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

#[cfg(test)]
mod typed_missing_policy {
    use super::*;

    #[test]
    fn only_typed_missing_requests_environment_removal() {
        for text in ["<missing>", "<MISSING>", ""] {
            assert_eq!(
                EnvironmentValues::decode(&Value::String(text.into()))
                    .unwrap()
                    .scalar_value(),
                Some(EnvironmentValue::Set(text.into()))
            );
        }
        assert_eq!(
            EnvironmentValues::decode(&Value::String(runmat_value::StringElement::Missing))
                .unwrap()
                .scalar_value(),
            Some(EnvironmentValue::Remove)
        );
        let array = runmat_value::StringArray::from_elements_unbounded(
            vec!["<missing>".into(), runmat_value::StringElement::Missing],
            vec![1, 2],
        )
        .unwrap();
        assert_eq!(
            EnvironmentValues::decode(&Value::StringArray(array))
                .unwrap()
                .values,
            vec![
                EnvironmentValue::Set("<missing>".into()),
                EnvironmentValue::Remove
            ]
        );
    }
}
