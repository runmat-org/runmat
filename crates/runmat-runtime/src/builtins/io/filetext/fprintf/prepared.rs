use super::admission::{
    coerce_to_format_string, is_fid_candidate, is_format_value, preflight_fid, preflight_format,
};
use super::target::{match_stream_label, parse_fid, target_from_fid, OutputTarget, SpecialStream};
use super::*;
use runmat_builtins::{FPRINTF_ERROR_INVALID_INPUT, FPRINTF_STREAM_LABEL_EXTENSION};
use runmat_value::Utf8Text;
mod cache;
use cache::Cache;

pub(super) struct PreparedFprintfCall {
    pub(super) target: OutputTarget,
    pub(super) format: Utf8Text,
    pub(super) data_args: crate::call::BuiltinArguments,
}

enum FormatPlan {
    Host { index: usize, text: Utf8Text },
    Resident { index: usize },
}

pub(super) async fn prepare(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<PreparedFprintfCall> {
    if args.is_empty() {
        return Err(fprintf_error_with_detail(
            &FPRINTF_ERROR_INVALID_INPUT,
            "not enough input arguments",
        ));
    }
    let mut staged = Cache::new(args.len(), runtime)?;
    let mut start = 0;
    let (format_index, format) = loop {
        let plan = next_format(args.view(), start)?.ok_or_else(|| {
            fprintf_error_with_detail(&FPRINTF_ERROR_INVALID_INPUT, "missing format string")
        })?;
        match plan {
            FormatPlan::Host { index, text } => break (index, text),
            FormatPlan::Resident { index } => {
                validate_unambiguous_host_target(runtime, args.view(), index)?;
                let host = super::super::cooperation::await_io(
                    runtime,
                    BUILTIN_NAME,
                    gather_value(&args[index]),
                )
                .await??;
                let text = coerce_to_format_string(&host)?;
                staged.put(index, host)?;
                if let Some(Value::String(runmat_value::StringElement::Text(text))) = text {
                    break (index, text);
                }
                start = index + 1;
            }
        }
    };

    let (target_index, target) =
        select_target(runtime, args.view(), format_index, &mut staged).await?;
    staged.checkpoint()?;
    let count = args.len() - 1 - usize::from(target_index.is_some());
    let mut cursor = 0usize;
    let data_args = args.try_map_preserving_roots(count, runtime, |_, source| {
        while cursor == format_index || target_index == Some(cursor) {
            cursor += 1;
        }
        let index = cursor;
        cursor += 1;
        Ok(staged.take(index).unwrap_or_else(|| source[index].clone()))
    })?;
    staged.checkpoint()?;
    Ok(PreparedFprintfCall {
        target,
        format,
        data_args,
    })
}

fn validate_unambiguous_host_target(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
    format_index: usize,
) -> BuiltinResult<()> {
    // A stream label encountered before a later actual format would outrank a
    // numeric fid. Only inspect a host fid when that override is impossible.
    if args
        .iter()
        .take(format_index)
        .try_fold(false, |found, value| {
            Ok::<_, RuntimeError>(found || match_stream_label(value)?.is_some())
        })?
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

fn next_format(
    args: crate::call::FunctionArgumentView<'_>,
    start: usize,
) -> BuiltinResult<Option<FormatPlan>> {
    for (index, value) in args.iter().enumerate().skip(start) {
        if match_stream_label(value)?.is_some() {
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
        if let Some(Value::String(runmat_value::StringElement::Text(text))) =
            coerce_to_format_string(value)?
        {
            return Ok(Some(FormatPlan::Host { index, text }));
        }
    }
    Ok(None)
}

async fn select_target(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
    format_index: usize,
    staged: &mut Cache<'_>,
) -> BuiltinResult<(Option<usize>, OutputTarget)> {
    for (index, value) in args.iter().enumerate().take(format_index) {
        if let Some(stream) = match_stream_label(value)? {
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
        let host = match staged.get(index) {
            Some(host) => host.clone(),
            None if matches!(value, Value::GpuTensor(_)) => {
                let host =
                    super::super::cooperation::await_io(runtime, BUILTIN_NAME, gather_value(value))
                        .await??;
                staged.put(index, host.clone())?;
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
    let _ = next_format(crate::call::FunctionArgumentView::from_slice(args), 0)?;
    Ok(())
}
