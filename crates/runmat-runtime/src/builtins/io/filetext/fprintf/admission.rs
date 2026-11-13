use super::*;
use crate::builtins::common::tensor;
use runmat_builtins::{
    FPRINTF_ERROR_FORMAT, FPRINTF_INTEGER_FORMAT_EXTENSION, FPRINTF_INTEGER_ID_EXTENSION,
    FPRINTF_NUMERIC_FORMAT_EXTENSION, FPRINTF_RESIDENT_FORMAT_EXTENSION,
    FPRINTF_RESIDENT_ID_EXTENSION, FPRINTF_SINGLE_ID_EXTENSION,
};
use runmat_value::{IntValue, NumericDType};
fn try_tensor_char_row_as_string(value: &Value) -> Option<Result<String, String>> {
    match value {
        Value::Tensor(t) => {
            let len = tensor::tensor_element_len(t);
            let is_row = numeric_format_row_width(t.shape()) == Some(len);
            if is_row {
                let mut out = String::with_capacity(len);
                for index in 0..len {
                    let code = t
                        .numeric_value_at(index)
                        .expect("index within authoritative numeric storage");
                    if let Some(code) = code.into_int_value() {
                        if let Some(ch) = char_from_int_value(&code) {
                            out.push(ch);
                        } else {
                            return Some(Err(
                                "fprintf: formatSpec contains invalid character code".to_string(),
                            ));
                        }
                    } else {
                        let code = code.materialize_f64();
                        if !code.is_finite() || code.fract().abs() > f64::EPSILON || code < 0.0 {
                            return Some(Err(
                                "fprintf: formatSpec must be a character row vector or string scalar"
                                    .to_string(),
                            ));
                        }
                        let v = code as u32;
                        // Allow full Unicode range; MATLAB chars are UTF-16 but format strings are ASCII-compatible typically
                        if let Some(ch) = char::from_u32(v) {
                            out.push(ch);
                        } else {
                            return Some(Err(
                                "fprintf: formatSpec contains invalid character code".to_string(),
                            ));
                        }
                    }
                }
                return Some(Ok(out));
            }
            None
        }
        _ => None,
    }
}

fn numeric_format_row_width(shape: &[usize]) -> Option<usize> {
    match shape {
        [columns] => Some(*columns),
        [1, columns, tail @ ..] if tail.iter().all(|dimension| *dimension == 1) => Some(*columns),
        _ => None,
    }
}

fn char_format_row_shape(value: &runmat_value::CharArray) -> bool {
    value.rows == 1
        && value.data().len() == value.cols
        && value
            .shape
            .get(2..)
            .is_none_or(|tail| tail.iter().all(|dimension| *dimension == 1))
}

pub(super) fn is_fid_candidate(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.len() == 1)
        || matches!(value, Value::GpuTensor(handle) if handle.shape().iter().all(|extent| *extent == 1))
}

pub(super) fn is_format_value(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_)
    ) || matches!(value, Value::Tensor(tensor) if tensor.len() >= 2 && try_tensor_char_row_as_string(value).is_some())
        || matches!(value, Value::GpuTensor(handle)
            if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle)
                && numeric_format_row_width(handle.shape()).is_some_and(|width| width >= 2))
}

pub(super) fn preflight_fid(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FPRINTF_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FPRINTF_INTEGER_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FPRINTF_SINGLE_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FPRINTF_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FPRINTF_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FPRINTF_SINGLE_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

pub(super) fn preflight_format(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FPRINTF_INTEGER_FORMAT_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FPRINTF_INTEGER_FORMAT_EXTENSION,
                BUILTIN_NAME,
            )?;
            crate::compatibility::ensure_builtin_extension_enabled(
                &FPRINTF_NUMERIC_FORMAT_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FPRINTF_NUMERIC_FORMAT_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FPRINTF_RESIDENT_FORMAT_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_none() {
                return Ok(());
            }
            crate::compatibility::ensure_builtin_extension_enabled(
                &FPRINTF_INTEGER_FORMAT_EXTENSION,
                BUILTIN_NAME,
            )
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
pub(super) fn coerce_single_integer_string_argument(
    format: crate::builtins::common::format::FormatSummary,
    mut args: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    if args.len() == 1 && format.sole_string_conversion_without_star() {
        let value = args.pop().expect("single formatting input");
        args.push(coerce_single_integer_string_value(value)?);
    }
    Ok(args)
}

pub(super) fn coerce_single_integer_string_value(value: Value) -> BuiltinResult<Value> {
    Ok(match value {
        Value::Int(value) => Value::string(integer_codes_to_string([value])?),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let storage = tensor.integer_storage().expect("checked integer storage");
            let values = (0..storage.len()).map(|index| {
                storage
                    .value_at(index)
                    .expect("index within integer format storage")
            });
            Value::string(integer_codes_to_string(values)?)
        }
        other => other,
    })
}

fn integer_codes_to_string(values: impl IntoIterator<Item = IntValue>) -> BuiltinResult<String> {
    let mut output = String::new();
    for value in values {
        let ch = char_from_int_value(&value).ok_or_else(|| {
            fprintf_error_with_detail(
                &FPRINTF_ERROR_FORMAT,
                "integer %s data contains an invalid character code",
            )
        })?;
        output.push(ch);
    }
    Ok(output)
}

fn char_from_int_value(value: &IntValue) -> Option<char> {
    value
        .try_to_u64()
        .and_then(|code| u32::try_from(code).ok())
        .and_then(char::from_u32)
}

pub(super) fn coerce_to_format_string(value: &Value) -> BuiltinResult<Option<Value>> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(Some(Value::String(
            runmat_value::StringElement::Text(s.clone()),
        ))),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => {
            Ok(Some(Value::String(sa.data()[0].clone())))
        }
        Value::CharArray(ca) if char_format_row_shape(ca) => {
            let s = crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                fprintf_error_with_detail(
                    &FPRINTF_ERROR_FORMAT,
                    "fprintf: formatSpec contains malformed UTF-16 text",
                )
            })?;
            Ok(Some(Value::string(s)))
        }
        Value::CharArray(_) => Err(fprintf_error_with_detail(
            &FPRINTF_ERROR_FORMAT,
            "fprintf: formatSpec must be a character row vector or string scalar",
        )),
        Value::Tensor(t) => {
            // Only accept numeric codepoint vectors of length >= 2 as formatSpec.
            // This avoids misinterpreting stray 1x1 numerics (e.g., accidental stack values)
            // as a valid format string.
            if tensor::tensor_element_len(t) >= 2 {
                match try_tensor_char_row_as_string(value) {
                    Some(Ok(s)) => Ok(Some(Value::string(s))),
                    Some(Err(e)) => Err(fprintf_error_with_detail(&FPRINTF_ERROR_FORMAT, e)),
                    None => Ok(None),
                }
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}
