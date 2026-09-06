use runmat_value::Value;

use crate::{gather_if_needed_async, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextScalarError {
    NotTextScalar,
}

pub(crate) async fn gather(args: Vec<Value>, builtin: &str) -> Result<Vec<Value>, RuntimeError> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(gather_if_needed_async(&value).await.map_err(|error| {
            crate::builtins::common::map_control_flow_with_builtin(error, builtin)
        })?);
    }
    Ok(gathered)
}

pub(crate) fn text_scalar(value: &Value) -> Result<String, TextScalarError> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(TextScalarError::NotTextScalar),
    }
}
