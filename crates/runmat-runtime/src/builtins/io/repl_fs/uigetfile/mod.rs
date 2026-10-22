//! Host-owned file dialog admission and results.
use super::file_dialog::{
    default_filters, ensure_same_directory, parse_filter_spec, scalar_text, selected_path_parts,
    try_scalar_text,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, UIGETFILE_DESCRIPTOR, UIGETFILE_ERROR_HOST,
    UIGETFILE_ERROR_INVALID_ARGUMENT, UIGETFILE_ERROR_INVALID_SELECTION,
    UIGETFILE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_filesystem::{OpenFileDialogRequest, OpenFileDialogSelection};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, Value};
use std::path::PathBuf;
const NAME: &str = "uigetfile";
#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
struct UigetfileOptions {
    request: OpenFileDialogRequest,
}

fn uigetfile_error(
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

fn invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    uigetfile_error(&UIGETFILE_ERROR_INVALID_ARGUMENT, detail)
}

fn invalid_selection(detail: impl AsRef<str>) -> RuntimeError {
    uigetfile_error(&UIGETFILE_ERROR_INVALID_SELECTION, detail)
}

fn host_error(source: RuntimeError) -> RuntimeError {
    let detail = source.message().to_string();
    let mut error = uigetfile_error(&UIGETFILE_ERROR_HOST, detail);
    error.source = Some(Box::new(source));
    error
}

fn host_io_error(source: std::io::Error) -> RuntimeError {
    let detail = source.to_string();
    let mut error = uigetfile_error(&UIGETFILE_ERROR_HOST, detail);
    error.source = Some(Box::new(source));
    error
}

#[runtime_builtin(
    name = "uigetfile",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::uigetfile"
)]
async fn uigetfile_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &UIGETFILE_DESCRIPTOR,
        &UIGETFILE_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )?;
    let options = parse_options(args.view())?;
    let request = runmat_filesystem::resolve_open_file_dialog_request(&options.request)
        .map_err(host_io_error)?;
    let selection = crate::context::select_file_open(runtime.as_ref(), request)
        .await
        .map_err(host_error)?;
    outputs_for_selection(selection, &options)
}

fn parse_options(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<UigetfileOptions> {
    let mut end = args.len();
    let mut multiselect = false;
    while end >= 2 {
        let option_name = match try_scalar_text(&args[end - 2], "option name", invalid_argument)? {
            Some(name) => name,
            None => break,
        };
        if !option_name.eq_ignore_ascii_case("MultiSelect") {
            break;
        }
        multiselect = parse_multiselect(&args[end - 1])?;
        end -= 2;
    }
    if end > 3 {
        return Err(invalid_argument(
            "expected filter, title, defaultName, and optional 'MultiSelect' name-value pair",
        ));
    }

    let filters = if end >= 1 {
        parse_filter_spec(&args[0], invalid_argument)?
    } else {
        default_filters()
    };
    let title = if end >= 2 {
        Some(scalar_text(&args[1], "title", invalid_argument)?)
    } else {
        None
    };
    let default_path = if end >= 3 {
        Some(PathBuf::from(scalar_text(
            &args[2],
            "defaultName",
            invalid_argument,
        )?))
    } else {
        None
    };

    Ok(UigetfileOptions {
        request: OpenFileDialogRequest {
            title,
            default_path,
            filters,
            multiselect,
        },
    })
}

fn parse_multiselect(value: &Value) -> BuiltinResult<bool> {
    let extension_value = match value {
        Value::Bool(enabled) => *enabled,
        Value::Num(number) if number.is_finite() => *number != 0.0,
        Value::Int(int) => !int.is_zero(),
        other => {
            let text = scalar_text(other, "MultiSelect value", invalid_argument)?;
            match text.trim().to_ascii_lowercase().as_str() {
                "on" => return Ok(true),
                "off" => return Ok(false),
                "true" | "yes" => true,
                "false" | "no" => false,
                _ => {
                    return Err(invalid_argument(
                        "MultiSelect must be 'on', 'off', true, or false",
                    ))
                }
            }
        }
    };
    crate::compatibility::ensure_builtin_extension_enabled(
        &runmat_builtins::UIGETFILE_MULTISELECT_EXTENSION,
        NAME,
    )?;
    Ok(extension_value)
}

fn outputs_for_selection(
    selection: Option<OpenFileDialogSelection>,
    options: &UigetfileOptions,
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
    selection: OpenFileDialogSelection,
    options: &UigetfileOptions,
) -> BuiltinResult<Vec<Value>> {
    if selection.paths.is_empty() {
        return Err(invalid_selection("provider returned no selected paths"));
    }

    let filter_index = selection.filter_index.unwrap_or(1);
    if filter_index == 0 || filter_index > options.request.filters.len() {
        return Err(invalid_selection(
            "provider returned an invalid filter index",
        ));
    }
    if !options.request.multiselect && selection.paths.len() > 1 {
        return Err(invalid_selection(
            "provider returned multiple paths for a single-selection request",
        ));
    }

    let first_path = selected_path_parts(&selection.paths[0], invalid_selection)?;
    let directory = first_path.directory;
    if options.request.multiselect && selection.paths.len() > 1 {
        ensure_same_directory(&selection.paths, &directory, invalid_selection)?;
        let mut names = Vec::with_capacity(selection.paths.len());
        for path in &selection.paths {
            names.push(Value::CharArray(CharArray::new_row(
                &selected_path_parts(path, invalid_selection)?.file_name,
            )));
        }
        let file_count = names.len();
        let files = CellArray::new(names, 1, file_count).map_err(invalid_selection)?;
        return Ok(vec![
            Value::Cell(files),
            Value::CharArray(CharArray::new_row(&directory)),
            Value::Num(filter_index as f64),
        ]);
    }

    Ok(vec![
        Value::CharArray(CharArray::new_row(&first_path.file_name)),
        Value::CharArray(CharArray::new_row(&directory)),
        Value::Num(filter_index as f64),
    ])
}
