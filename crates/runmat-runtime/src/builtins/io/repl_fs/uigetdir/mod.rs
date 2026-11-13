//! Host-owned directory selection with preflight admission.
use super::file_dialog::scalar_text;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, UIGETDIR_DESCRIPTOR, UIGETDIR_ERROR_HOST,
    UIGETDIR_ERROR_INVALID_ARGUMENT, UIGETDIR_ERROR_INVALID_SELECTION,
    UIGETDIR_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_filesystem::{DirectoryDialogRequest, DirectoryDialogSelection};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};
use std::path::PathBuf;
const NAME: &str = "uigetdir";
#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
struct UigetdirOptions {
    request: DirectoryDialogRequest,
}

fn uigetdir_error(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
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
    uigetdir_error(&UIGETDIR_ERROR_INVALID_ARGUMENT, detail)
}

fn invalid_selection(detail: impl AsRef<str>) -> RuntimeError {
    uigetdir_error(&UIGETDIR_ERROR_INVALID_SELECTION, detail)
}

fn host_error(source: RuntimeError) -> RuntimeError {
    let detail = source.message().to_string();
    let mut error = uigetdir_error(&UIGETDIR_ERROR_HOST, detail);
    error.source = Some(Box::new(source));
    error
}

fn host_io_error(source: std::io::Error) -> RuntimeError {
    let detail = source.to_string();
    let mut error = uigetdir_error(&UIGETDIR_ERROR_HOST, detail);
    error.source = Some(Box::new(source));
    error
}

#[runtime_builtin(
    name = "uigetdir",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::uigetdir"
)]
async fn uigetdir_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &UIGETDIR_DESCRIPTOR,
        &UIGETDIR_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )?;
    let options = parse_options(args.view())?;
    let request = runmat_filesystem::resolve_directory_dialog_request(&options.request)
        .map_err(host_io_error)?;
    let selection = crate::context::select_directory(runtime.as_ref(), request)
        .await
        .map_err(host_error)?;
    outputs_for_selection(selection)
}

fn parse_options(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<UigetdirOptions> {
    if args.len() > 2 {
        return Err(invalid_argument("expected startPath and title"));
    }
    let default_path = if !args.is_empty() {
        Some(PathBuf::from(scalar_text(
            &args[0],
            "startPath",
            invalid_argument,
        )?))
    } else {
        None
    };
    let title = if args.len() >= 2 {
        Some(scalar_text(&args[1], "title", invalid_argument)?)
    } else {
        None
    };

    Ok(UigetdirOptions {
        request: DirectoryDialogRequest {
            title,
            default_path,
        },
    })
}

fn outputs_for_selection(selection: Option<DirectoryDialogSelection>) -> BuiltinResult<Value> {
    let output = match selection {
        Some(selection) => selected_output(selection)?,
        None => Value::Num(0.0),
    };

    if let Some(out_count) = crate::output_count::current_output_count() {
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![output],
        ));
    }

    Ok(output)
}

fn selected_output(selection: DirectoryDialogSelection) -> BuiltinResult<Value> {
    if selection.path.as_os_str().is_empty() {
        return Err(invalid_selection("provider returned an empty folder path"));
    }
    Ok(Value::CharArray(CharArray::new_row(
        &selection.path.to_string_lossy(),
    )))
}
