use runmat_builtins::BuiltinExtensionDescriptor;
use runmat_value::{CharArray, Tensor, Value};

use crate::builtins::io::repl_fs::text_conversion::tensor_char_codes_to_string;

#[derive(Clone, Copy)]
pub(in crate::builtins::io::repl_fs) enum NumericPolicy {
    Reject,
    RunMatExtension(&'static BuiltinExtensionDescriptor),
}

pub(in crate::builtins::io::repl_fs) enum DecodeError {
    InvalidInput,
    Encoding(runmat_value::Utf16DecodeError),
    Provider(crate::RuntimeError),
    Compatibility(crate::RuntimeError),
}

pub(in crate::builtins::io::repl_fs) async fn decode(
    values: crate::call::BuiltinArguments,
    builtin: &'static str,
    policy: NumericPolicy,
) -> Result<Vec<String>, DecodeError> {
    let mut admitted = Vec::new();
    for value in &values {
        admit(value, builtin, policy, &mut admitted)?;
    }
    let mut output = Vec::new();
    for value in admitted {
        match value {
            AdmittedToken::Text(text) => output.push(text),
            AdmittedToken::Resident(value) => {
                let gathered = crate::gather_if_needed_async(&value)
                    .await
                    .map_err(DecodeError::Provider)?;
                let Value::Tensor(tensor) = gathered else {
                    return Err(DecodeError::InvalidInput);
                };
                output.push(decode_numeric_row(&tensor)?);
            }
        }
    }
    Ok(output)
}

enum AdmittedToken {
    Text(String),
    Resident(Value),
}

fn admit(
    value: &Value,
    builtin: &'static str,
    policy: NumericPolicy,
    output: &mut Vec<AdmittedToken>,
) -> Result<(), DecodeError> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            output.push(AdmittedToken::Text(text.as_str().to_owned()))
        }
        Value::StringArray(array) if array.data().iter().all(|text| !text.is_missing()) => output
            .extend(array.data().iter().map(|text| {
                AdmittedToken::Text(text.as_str().expect("checked nonmissing string").to_owned())
            })),
        Value::CharArray(chars)
            if crate::builtins::common::shape::effective_rank(&chars.shape) <= 2 =>
        {
            collect_char_rows(chars, output)?;
        }
        Value::Tensor(tensor) => {
            admit_numeric(builtin, policy)?;
            output.push(AdmittedToken::Text(decode_numeric_row(tensor)?));
        }
        Value::GpuTensor(ref handle) => {
            admit_numeric(builtin, policy)?;
            if !numeric_row(handle.shape()) {
                return Err(DecodeError::InvalidInput);
            }
            output.push(AdmittedToken::Resident((*value).clone()));
        }
        Value::Cell(cell) => {
            for value in cell.iter_column_major() {
                admit(value, builtin, policy, output)?;
            }
        }
        _ => return Err(DecodeError::InvalidInput),
    }
    Ok(())
}

fn admit_numeric(builtin: &'static str, policy: NumericPolicy) -> Result<(), DecodeError> {
    match policy {
        NumericPolicy::Reject => Err(DecodeError::InvalidInput),
        NumericPolicy::RunMatExtension(extension) => {
            crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)
                .map_err(DecodeError::Compatibility)
        }
    }
}

fn collect_char_rows(
    chars: &CharArray,
    output: &mut Vec<AdmittedToken>,
) -> Result<(), DecodeError> {
    for row in 0..chars.rows {
        let start = row * chars.cols;
        let text = runmat_value::decode_utf16_code_units(&chars.data()[start..start + chars.cols])
            .map_err(DecodeError::Encoding)?;
        output.push(AdmittedToken::Text(text.trim_end().to_owned()));
    }
    Ok(())
}

fn decode_numeric_row(tensor: &Tensor) -> Result<String, DecodeError> {
    if !numeric_row(tensor.shape()) {
        return Err(DecodeError::InvalidInput);
    }
    tensor_char_codes_to_string(tensor).ok_or(DecodeError::InvalidInput)
}

fn numeric_row(shape: &[usize]) -> bool {
    let rows = match shape {
        [] => 0,
        [_] => 1,
        [rows, ..] => *rows,
    };
    crate::builtins::common::shape::effective_rank(shape) <= 2 && rows <= 1
}
