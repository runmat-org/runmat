//! Host-owned file dialog admission and results.
use super::file_dialog::{default_filters, parse_filter_spec, scalar_text, selected_path_parts};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, UIPUTFILE_DESCRIPTOR, UIPUTFILE_ERROR_HOST,
    UIPUTFILE_ERROR_INVALID_ARGUMENT, UIPUTFILE_ERROR_INVALID_SELECTION,
    UIPUTFILE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_filesystem::{SaveFileDialogRequest, SaveFileDialogSelection};
use runmat_macros::runtime_builtin;
use runmat_value::Value;
use std::path::PathBuf;
const NAME: &str = "uiputfile";
#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
struct UiputfileOptions {
    request: SaveFileDialogRequest,
}

fn uiputfile_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(detail: String) -> RuntimeError {
    uiputfile_error(&UIPUTFILE_ERROR_INVALID_ARGUMENT, detail)
}

fn invalid_selection(detail: String) -> RuntimeError {
    uiputfile_error(&UIPUTFILE_ERROR_INVALID_SELECTION, detail)
}

fn host_error(source: RuntimeError) -> RuntimeError {
    let detail = source.message().to_string();
    let mut error = uiputfile_error(&UIPUTFILE_ERROR_HOST, detail);
    error.source = Some(Box::new(source));
    error
}

fn host_io_error(source: std::io::Error) -> RuntimeError {
    let detail = source.to_string();
    let mut error = uiputfile_error(&UIPUTFILE_ERROR_HOST, detail);
    error.source = Some(Box::new(source));
    error
}

#[runtime_builtin(
    name = "uiputfile",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::uiputfile"
)]
async fn uiputfile_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &UIPUTFILE_DESCRIPTOR,
        &UIPUTFILE_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )?;
    let options = parse_options(args.view())?;
    let request = runmat_filesystem::resolve_save_file_dialog_request(&options.request)
        .map_err(host_io_error)?;
    let selection = crate::context::select_file_save(runtime.as_ref(), request)
        .await
        .map_err(host_error)?;
    outputs_for_selection(selection, &options)
}

fn parse_options(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<UiputfileOptions> {
    if args.len() > 3 {
        return Err(invalid_argument(
            "expected filter, title, and defaultName".to_string(),
        ));
    }

    let filters = if !args.is_empty() {
        parse_filter_spec(&args[0], invalid_argument)?
    } else {
        default_filters()
    };
    let title = if args.len() >= 2 {
        Some(scalar_text(&args[1], "title", invalid_argument)?)
    } else {
        None
    };
    let default_path = if args.len() >= 3 {
        Some(PathBuf::from(scalar_text(
            &args[2],
            "defaultName",
            invalid_argument,
        )?))
    } else {
        None
    };

    Ok(UiputfileOptions {
        request: SaveFileDialogRequest {
            title,
            default_path,
            filters,
        },
    })
}

fn outputs_for_selection(
    selection: Option<SaveFileDialogSelection>,
    options: &UiputfileOptions,
) -> BuiltinResult<Value> {
    let outputs = match selection {
        Some(selection) => selected_outputs(selection, options)?,
        None => vec![Value::Num(0.0), Value::Num(0.0), Value::Num(0.0)],
    };

    if let Some(out_count) = crate::output_count::current_output_count() {
        return Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        ));
    }

    Ok(outputs.into_iter().next().unwrap_or(Value::Num(0.0)))
}

fn selected_outputs(
    selection: SaveFileDialogSelection,
    options: &UiputfileOptions,
) -> BuiltinResult<Vec<Value>> {
    let filter_index = selection.filter_index.unwrap_or(1);
    if filter_index == 0 || filter_index > options.request.filters.len() {
        return Err(invalid_selection(
            "provider returned an invalid filter index".to_string(),
        ));
    }

    let selected = selected_path_parts(&selection.path, invalid_selection)?;
    Ok(vec![
        Value::CharArray(runmat_value::CharArray::new_row(&selected.file_name)),
        Value::CharArray(runmat_value::CharArray::new_row(&selected.directory)),
        Value::Num(filter_index as f64),
    ])
}
