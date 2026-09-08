use crate::builtins::common::shape::dims_to_row_tensor;
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::CELLFUN_ERROR_FUNCTION_ERROR;
use runmat_value::{StructValue, Value};

use super::error;

pub(super) fn value(
    raw_error: &RuntimeError,
    linear_index: usize,
    shape: &[usize],
) -> BuiltinResult<Value> {
    let (identifier, message) = identifier_and_message(raw_error);
    let mut context = StructValue::new();
    context
        .fields
        .insert("identifier".into(), Value::String(identifier));
    context
        .fields
        .insert("message".into(), Value::String(message));
    context
        .fields
        .insert("index".into(), Value::Num((linear_index + 1) as f64));
    let indices = dims_to_row_tensor(&linear_indices(linear_index, shape))
        .map_err(|reason| error::internal(format!("cellfun: {reason}")))?;
    context
        .fields
        .insert("indices".into(), Value::Tensor(indices));
    Ok(Value::Struct(context))
}

fn identifier_and_message(error: &RuntimeError) -> (String, String) {
    if let Some(identifier) = error.identifier() {
        return (identifier.to_string(), error.message().to_string());
    }
    split_message(error.message())
}

fn split_message(raw: &str) -> (String, String) {
    let trimmed = raw.trim();
    let mut separators = trimmed.match_indices(':');
    separators.next();
    if let Some((second, _)) = separators.next() {
        let identifier = trimmed[..second].trim();
        let message = trimmed[second + 1..].trim();
        if !identifier.is_empty() && identifier.contains(':') {
            return (
                identifier.to_string(),
                if message.is_empty() { trimmed } else { message }.to_string(),
            );
        }
    } else if trimmed.len() >= 7
        && (trimmed[..7].eq_ignore_ascii_case("matlab:")
            || trimmed[..7].eq_ignore_ascii_case("runmat:"))
    {
        return (trimmed.to_string(), String::new());
    }
    (
        CELLFUN_ERROR_FUNCTION_ERROR
            .identifier
            .expect("cellfun function error descriptor must define an identifier")
            .to_string(),
        trimmed.to_string(),
    )
}

fn linear_indices(mut index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![1];
    }
    shape
        .iter()
        .map(|dimension| {
            if *dimension == 0 {
                1
            } else {
                let coordinate = index % dimension + 1;
                index /= dimension;
                coordinate
            }
        })
        .collect()
}
