use runmat_builtins::BuiltinExtensionDescriptor;
use runmat_value::{CharArray, StringArray, Tensor, Value};

#[derive(Clone, Copy)]
pub(in crate::builtins::io::repl_fs) enum NumericPolicy {
    Reject,
    RunMatExtension(&'static BuiltinExtensionDescriptor),
}

pub(in crate::builtins::io::repl_fs) enum DecodeError {
    InvalidInput,
    Provider(crate::RuntimeError),
    Compatibility(crate::RuntimeError),
}

pub(in crate::builtins::io::repl_fs) async fn decode(
    values: Vec<Value>,
    builtin: &'static str,
    policy: NumericPolicy,
) -> Result<Vec<String>, DecodeError> {
    for value in &values {
        preflight(value, builtin, policy)?;
    }
    let mut output = Vec::new();
    for value in values {
        collect(value, &mut output).await?;
    }
    Ok(output)
}

fn preflight(
    value: &Value,
    builtin: &'static str,
    policy: NumericPolicy,
) -> Result<(), DecodeError> {
    match value {
        Value::Tensor(_) | Value::GpuTensor(_) => match policy {
            NumericPolicy::Reject => Err(DecodeError::InvalidInput),
            NumericPolicy::RunMatExtension(extension) => {
                crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)
                    .map_err(DecodeError::Compatibility)
            }
        },
        Value::Cell(cell) => cell
            .data
            .iter()
            .try_for_each(|value| preflight(value, builtin, policy)),
        Value::String(_) | Value::StringArray(_) => Ok(()),
        Value::CharArray(chars) if chars.shape.len() <= 2 => Ok(()),
        Value::CharArray(_) => Err(DecodeError::InvalidInput),
        _ => Err(DecodeError::InvalidInput),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn collect(value: Value, output: &mut Vec<String>) -> Result<(), DecodeError> {
    match value {
        Value::String(text) => output.push(text),
        Value::StringArray(StringArray { data, .. }) => output.extend(data),
        Value::CharArray(chars) => collect_char_rows(chars, output),
        Value::Tensor(tensor) => output.push(decode_numeric_row(&tensor)?),
        value @ Value::GpuTensor(_) => {
            let gathered = crate::gather_if_needed_async(&value)
                .await
                .map_err(DecodeError::Provider)?;
            let Value::Tensor(tensor) = gathered else {
                return Err(DecodeError::InvalidInput);
            };
            output.push(decode_numeric_row(&tensor)?);
        }
        Value::Cell(cell) => {
            for value in &cell.data {
                collect(value.clone(), output).await?;
            }
        }
        _ => return Err(DecodeError::InvalidInput),
    }
    Ok(())
}

fn collect_char_rows(chars: CharArray, output: &mut Vec<String>) {
    for row in 0..chars.rows {
        let start = row * chars.cols;
        let text: String = chars.data[start..start + chars.cols].iter().collect();
        output.push(text.trim_end().to_owned());
    }
}

fn decode_numeric_row(tensor: &Tensor) -> Result<String, DecodeError> {
    if tensor.shape.len() > 2 || tensor.rows() > 1 {
        return Err(DecodeError::InvalidInput);
    }
    super::super::tensor_char_codes_to_string(tensor).ok_or(DecodeError::InvalidInput)
}
