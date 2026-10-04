use super::target::{match_stream_label, parse_fid, target_from_fid, OutputTarget, SpecialStream};
use super::*;
use crate::builtins::common::tensor;
use runmat_builtins::{
    FPRINTF_ERROR_FORMAT, FPRINTF_ERROR_INVALID_INPUT, FPRINTF_INTEGER_FORMAT_EXTENSION,
    FPRINTF_INTEGER_ID_EXTENSION, FPRINTF_NUMERIC_FORMAT_EXTENSION,
    FPRINTF_RESIDENT_FORMAT_EXTENSION, FPRINTF_RESIDENT_ID_EXTENSION, FPRINTF_SINGLE_ID_EXTENSION,
    FPRINTF_STREAM_LABEL_EXTENSION,
};
use runmat_value::{IntValue, NumericDType};
fn try_tensor_char_row_as_string(value: &Value) -> Option<Result<String, String>> {
    match value {
        Value::Tensor(t) => {
            let len = tensor::tensor_element_len(t);
            let is_row = (t.shape.len() == 2 && t.shape[0] == 1 && len == t.shape[1])
                || (t.shape.len() == 1 && len == t.shape[0]);
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

#[cfg(test)]
pub(super) fn preflight_special_roles(args: &[Value]) -> BuiltinResult<()> {
    let _ = next_format(args, 0)?;
    Ok(())
}

pub(super) struct PreparedFprintfCall {
    pub(super) target: OutputTarget,
    pub(super) format: String,
    pub(super) data_args: Vec<Value>,
}

enum FormatPlan {
    Host { index: usize, text: String },
    Resident { index: usize },
}

pub(super) async fn prepare(args: &[Value]) -> BuiltinResult<PreparedFprintfCall> {
    if args.is_empty() {
        return Err(fprintf_error_with_detail(
            &FPRINTF_ERROR_INVALID_INPUT,
            "not enough input arguments",
        ));
    }
    let mut staged = vec![None; args.len()];
    let mut start = 0;
    let (format_index, format) = loop {
        let plan = next_format(args, start)?.ok_or_else(|| {
            fprintf_error_with_detail(&FPRINTF_ERROR_INVALID_INPUT, "missing format string")
        })?;
        match plan {
            FormatPlan::Host { index, text } => break (index, text),
            FormatPlan::Resident { index } => {
                let host = gather_value(&args[index]).await?;
                let text = map_string_result(
                    coerce_to_format_string(&host),
                    &FPRINTF_ERROR_INVALID_INPUT,
                )?;
                staged[index] = Some(host);
                if let Some(Value::String(text)) = text {
                    break (index, text);
                }
                start = index + 1;
            }
        }
    };

    let (target_index, target) = select_target(args, format_index, &mut staged).await?;
    let mut data_args = Vec::with_capacity(args.len().saturating_sub(1));
    for (index, value) in args.iter().enumerate() {
        if index == format_index || target_index == Some(index) {
            continue;
        }
        data_args.push(staged[index].take().unwrap_or_else(|| value.clone()));
    }
    Ok(PreparedFprintfCall {
        target,
        format,
        data_args,
    })
}

fn next_format(args: &[Value], start: usize) -> BuiltinResult<Option<FormatPlan>> {
    for (index, value) in args.iter().enumerate().skip(start) {
        if match_stream_label(value).is_some() {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FPRINTF_STREAM_LABEL_EXTENSION,
                BUILTIN_NAME,
            )?;
            continue;
        }
        if !is_format_value(value) {
            continue;
        }
        preflight_format(value)?;
        if matches!(value, Value::GpuTensor(_)) {
            return Ok(Some(FormatPlan::Resident { index }));
        }
        if let Some(Value::String(text)) =
            map_string_result(coerce_to_format_string(value), &FPRINTF_ERROR_INVALID_INPUT)?
        {
            return Ok(Some(FormatPlan::Host { index, text }));
        }
    }
    Ok(None)
}

async fn select_target(
    args: &[Value],
    format_index: usize,
    staged: &mut [Option<Value>],
) -> BuiltinResult<(Option<usize>, OutputTarget)> {
    for (index, value) in args.iter().enumerate().take(format_index) {
        if let Some(stream) = match_stream_label(value) {
            let target = match stream {
                SpecialStream::Stdout => OutputTarget::Stdout,
                SpecialStream::Stderr => OutputTarget::Stderr,
            };
            return Ok((Some(index), target));
        }
    }
    for (index, value) in args.iter().enumerate().take(format_index) {
        if !is_fid_candidate(value) {
            continue;
        }
        preflight_fid(value)?;
        let host = match staged[index].as_ref() {
            Some(host) => host.clone(),
            None if matches!(value, Value::GpuTensor(_)) => {
                let host = gather_value(value).await?;
                staged[index] = Some(host.clone());
                host
            }
            None => value.clone(),
        };
        if let Ok(fid) = parse_fid(&host) {
            return Ok((Some(index), target_from_fid(fid)?));
        }
    }
    Ok((None, OutputTarget::Stdout))
}

fn is_fid_candidate(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.len() == 1)
        || matches!(value, Value::GpuTensor(handle) if handle.shape().iter().product::<usize>() == 1)
}

fn is_format_value(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_)
    ) || matches!(value, Value::Tensor(tensor) if tensor.len() >= 2 && try_tensor_char_row_as_string(value).is_some())
        || matches!(value, Value::GpuTensor(handle)
            if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle)
                && ((handle.shape().len() == 2 && handle.shape()[0] == 1 && handle.shape()[1] >= 2)
                    || (handle.shape().len() == 1 && handle.shape()[0] >= 2)))
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

pub(super) fn coerce_single_integer_string_argument(
    format: &str,
    mut args: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    if args.len() != 1 || !has_only_string_conversion(format) {
        return Ok(args);
    }
    let Some(value) = args.pop() else {
        return Ok(args);
    };
    let converted = match value {
        Value::Int(value) => Value::String(integer_codes_to_string([value])?),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let storage = tensor.integer_storage().expect("checked integer storage");
            let values = (0..storage.len()).map(|index| {
                storage
                    .value_at(index)
                    .expect("index within integer format storage")
            });
            Value::String(integer_codes_to_string(values)?)
        }
        other => other,
    };
    Ok(vec![converted])
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

fn has_only_string_conversion(format: &str) -> bool {
    let mut chars = format.chars().peekable();
    let mut conversions = Vec::new();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            continue;
        }
        if chars.peek() == Some(&'%') {
            chars.next();
            continue;
        }
        for next in chars.by_ref() {
            if next.is_ascii_alphabetic() {
                conversions.push(next);
                break;
            }
        }
    }
    conversions == ['s']
}

fn char_from_int_value(value: &IntValue) -> Option<char> {
    value
        .try_to_u64()
        .and_then(|code| u32::try_from(code).ok())
        .and_then(char::from_u32)
}

pub(super) fn coerce_to_format_string(value: &Value) -> Result<Option<Value>, String> {
    match value {
        Value::String(s) => Ok(Some(Value::String(s.clone()))),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(Some(Value::String(sa.data[0].clone()))),
        Value::CharArray(ca) => {
            let s: String = ca.data.iter().collect();
            Ok(Some(Value::String(s)))
        }
        Value::Tensor(t) => {
            // Only accept numeric codepoint vectors of length >= 2 as formatSpec.
            // This avoids misinterpreting stray 1x1 numerics (e.g., accidental stack values)
            // as a valid format string.
            if tensor::tensor_element_len(t) >= 2 {
                match try_tensor_char_row_as_string(value) {
                    Some(Ok(s)) => Ok(Some(Value::String(s))),
                    Some(Err(e)) => Err(e),
                    None => Ok(None),
                }
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}
