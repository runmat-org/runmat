//! Host text joining, with an explicit compatibility-gated row-wise extension.
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, STRJOIN_DESCRIPTOR, STRJOIN_ERROR_ARGUMENT_COUNT,
    STRJOIN_ERROR_INVALID_DELIMITER, STRJOIN_ERROR_INVALID_INPUT, STRJOIN_ERROR_TOO_MANY_OUTPUTS,
    STRJOIN_ROWWISE_EXTENSION,
};
use runmat_value::{CharArray, StringArray, Value};

#[runmat_macros::runtime_builtin(
    name = "strjoin",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::strjoin"
)]
pub async fn strjoin_builtin(input: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &STRJOIN_DESCRIPTOR,
        &STRJOIN_ERROR_TOO_MANY_OUTPUTS,
        "strjoin",
    )?;
    if rest.len() > 1 {
        return Err(error(&STRJOIN_ERROR_ARGUMENT_COUNT));
    }
    let explicit = !rest.is_empty();
    let delimiter = rest
        .into_iter()
        .next()
        .unwrap_or_else(|| Value::String(" ".into()));
    // Preserve the established RunMat two-argument behavior, including column
    // vectors and char rows. Strict mode uses MATLAB vector semantics instead.
    if explicit
        && crate::compatibility::runmat_extensions_enabled()
        && matches!(
            &input,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        )
    {
        let legacy_delimiter = match &delimiter {
            Value::Int(integer) => Some(integer.decimal_string()),
            Value::Num(number) => Some(number.to_string()),
            Value::Bool(logical) => Some(logical.to_string()),
            _ => text_scalar(&delimiter),
        };
        if let Some(delimiter) = legacy_delimiter {
            crate::compatibility::ensure_builtin_extension_enabled(
                &STRJOIN_ROWWISE_EXTENSION,
                "strjoin",
            )?;
            return rowwise(input, &delimiter);
        }
    }
    let (items, character_output) = match input {
        Value::String(text) => (vec![text], false),
        Value::StringArray(array) if vector(&array.shape) => (array.data, false),
        Value::Cell(cell) if vector(&cell.shape) => {
            let items = cell
                .into_column_major()
                .map_err(|_| error(&STRJOIN_ERROR_INVALID_INPUT))?
                .into_iter()
                .map(|item| match item {
                    Value::CharArray(chars) => {
                        char_text(&chars).ok_or_else(|| error(&STRJOIN_ERROR_INVALID_INPUT))
                    }
                    _ => Err(error(&STRJOIN_ERROR_INVALID_INPUT)),
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            (items, true)
        }
        _ => return Err(error(&STRJOIN_ERROR_INVALID_INPUT)),
    };
    let delimiters = delimiters(delimiter, items.len().saturating_sub(1))?;
    let mut joined = String::new();
    for (index, item) in items.into_iter().enumerate() {
        if index > 0 {
            joined.push_str(&delimiters[index - 1]);
        }
        joined.push_str(&item);
    }
    Ok(if character_output {
        Value::CharArray(CharArray::new_row(&joined))
    } else {
        Value::String(joined)
    })
}

fn error(descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    crate::build_runtime_error(descriptor.message)
        .with_builtin("strjoin")
        .with_identifier(descriptor.identifier.expect("strjoin error identifier"))
        .build()
}

fn matrix(shape: &[usize]) -> bool {
    shape.iter().skip(2).all(|dimension| *dimension == 1)
}
fn vector(shape: &[usize]) -> bool {
    matrix(shape)
        && (shape.contains(&0) || shape.iter().filter(|dimension| **dimension > 1).count() <= 1)
}
fn char_text(chars: &CharArray) -> Option<String> {
    (chars.rows <= 1 && matrix(&chars.shape)).then(|| chars.data.iter().collect())
}
fn text_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(chars) => char_text(chars),
        _ => None,
    }
}
fn delimiters(value: Value, count: usize) -> BuiltinResult<Vec<String>> {
    if let Some(text) = text_scalar(&value) {
        return Ok(vec![text; count]);
    }
    let items = match value {
        Value::StringArray(array) if vector(&array.shape) => array.data,
        Value::Cell(cell) if vector(&cell.shape) => cell
            .into_column_major()
            .map_err(|_| error(&STRJOIN_ERROR_INVALID_DELIMITER))?
            .into_iter()
            .map(|item| text_scalar(&item).ok_or_else(|| error(&STRJOIN_ERROR_INVALID_DELIMITER)))
            .collect::<BuiltinResult<Vec<_>>>()?,
        _ => return Err(error(&STRJOIN_ERROR_INVALID_DELIMITER)),
    };
    if items.len() != count {
        return Err(error(&STRJOIN_ERROR_INVALID_DELIMITER));
    }
    Ok(items)
}
fn rowwise(value: Value, delimiter: &str) -> BuiltinResult<Value> {
    let array = match value {
        Value::String(text) => StringArray::new(vec![text], vec![1, 1]),
        Value::StringArray(array) if matrix(&array.shape) => Ok(array),
        Value::CharArray(chars) if matrix(&chars.shape) => {
            let rows = (0..chars.rows)
                .map(|row| {
                    chars.data[row * chars.cols..(row + 1) * chars.cols]
                        .iter()
                        .collect()
                })
                .collect();
            StringArray::new(rows, vec![chars.rows, 1])
        }
        _ => return Err(error(&STRJOIN_ERROR_INVALID_INPUT)),
    }
    .map_err(|_| error(&STRJOIN_ERROR_INVALID_INPUT))?;
    let rows = array.shape.first().copied().unwrap_or(array.data.len());
    let cols = array.shape.get(1).copied().unwrap_or(1);
    let (joined, shape) = if rows == 0 || cols == 0 {
        (Vec::new(), vec![0, 0])
    } else {
        let joined = (0..rows)
            .map(|row| {
                (0..cols)
                    .map(|col| array.data[row + col * rows].as_str())
                    .collect::<Vec<_>>()
                    .join(delimiter)
            })
            .collect();
        (joined, vec![rows, 1])
    };
    Ok(Value::StringArray(
        StringArray::new(joined, shape).map_err(|_| error(&STRJOIN_ERROR_INVALID_INPUT))?,
    ))
}

#[cfg(test)]
mod tests;
