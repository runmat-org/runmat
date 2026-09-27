use runmat_value::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextScalarError {
    NotTextScalar,
}

pub(crate) fn text_scalar(value: &Value) -> Result<String, TextScalarError> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array)
            if array.rows == 1
                && crate::builtins::common::shape::effective_rank(&array.shape) <= 2 =>
        {
            Ok(array.data.iter().collect())
        }
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(TextScalarError::NotTextScalar),
    }
}
