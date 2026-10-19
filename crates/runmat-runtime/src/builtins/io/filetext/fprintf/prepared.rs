use super::admission::{
    coerce_to_format_string, is_fid_candidate, is_format_value, preflight_fid, preflight_format,
};
use super::target::{match_stream_label, parse_fid, target_from_fid, OutputTarget, SpecialStream};
use super::*;
use runmat_builtins::{FPRINTF_ERROR_INVALID_INPUT, FPRINTF_STREAM_LABEL_EXTENSION};

pub(super) struct PreparedFprintfCall {
    pub(super) target: OutputTarget,
    pub(super) format: String,
    pub(super) data_args: Vec<Value>,
}

enum FormatPlan {
    Host { index: usize, text: String },
    Resident { index: usize },
}

pub(super) async fn prepare(
    runtime: Option<&crate::context::RuntimeContext>,
    args: &[Value],
) -> BuiltinResult<PreparedFprintfCall> {
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
                validate_unambiguous_host_target(runtime, args, index)?;
                let host = super::super::cooperation::await_io(
                    runtime,
                    BUILTIN_NAME,
                    gather_value(&args[index]),
                )
                .await??;
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

    let (target_index, target) = select_target(runtime, args, format_index, &mut staged).await?;
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

fn validate_unambiguous_host_target(
    runtime: Option<&crate::context::RuntimeContext>,
    args: &[Value],
    format_index: usize,
) -> BuiltinResult<()> {
    // A stream label encountered before a later actual format would outrank a
    // numeric fid. Only inspect a host fid when that override is impossible.
    if args
        .iter()
        .take(format_index)
        .any(|value| match_stream_label(value).is_some())
    {
        return Ok(());
    }
    for value in args.iter().take(format_index) {
        if !is_fid_candidate(value) || matches!(value, Value::GpuTensor(_)) {
            continue;
        }
        let parsed = parse_fid(value);
        if format_index == 1 {
            let fid = map_string_result(parsed, &FPRINTF_ERROR_INVALID_INPUT)?;
            target_from_fid(runtime, fid)?;
            break;
        }
        if let Ok(fid) = parsed {
            target_from_fid(runtime, fid)?;
            break;
        }
    }
    Ok(())
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
    runtime: Option<&crate::context::RuntimeContext>,
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
                let host =
                    super::super::cooperation::await_io(runtime, BUILTIN_NAME, gather_value(value))
                        .await??;
                staged[index] = Some(host.clone());
                host
            }
            None => value.clone(),
        };
        let parsed = parse_fid(&host);
        if format_index == 1 && index == 0 {
            let fid = map_string_result(parsed, &FPRINTF_ERROR_INVALID_INPUT)?;
            return Ok((Some(index), target_from_fid(runtime, fid)?));
        }
        if let Ok(fid) = parsed {
            return Ok((Some(index), target_from_fid(runtime, fid)?));
        }
    }
    Ok((None, OutputTarget::Stdout))
}

#[cfg(test)]
pub(super) fn preflight_special_roles(args: &[Value]) -> BuiltinResult<()> {
    let _ = next_format(args, 0)?;
    Ok(())
}
