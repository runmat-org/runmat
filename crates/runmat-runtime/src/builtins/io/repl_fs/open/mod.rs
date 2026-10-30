//! File and workspace document dispatch.
mod path;
#[cfg(test)]
mod tests;
use crate::builtins::common::fs::path_to_string;
use crate::builtins::common::path_search::should_treat_as_path;
use crate::execution::RootedValue;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use path::resolve_open_path;
use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{
    OPEN_DESCRIPTOR, OPEN_ERROR_ARG_COUNT, OPEN_ERROR_HANDLER, OPEN_ERROR_NAME,
    OPEN_ERROR_NOT_FOUND, OPEN_ERROR_OUTPUT_COUNT, OPEN_ERROR_UNSUPPORTED,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
use std::path::{Path, PathBuf};
const BUILTIN_NAME: &str = "open";
#[runtime_builtin(
    name = "open",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::open"
)]
async fn open_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    if args.len() != 1 {
        return Err(open_error(
            &OPEN_ERROR_ARG_COUNT,
            OPEN_ERROR_ARG_COUNT.message,
        ));
    }
    if requested_output_count() > 1 {
        return Err(open_error(
            &OPEN_ERROR_OUTPUT_COUNT,
            format!(
                "open: expected at most one output, got {}",
                requested_output_count()
            ),
        ));
    }

    let name = open_name(&args[0])?;
    if name.is_empty() {
        return Err(open_error(&OPEN_ERROR_NAME, "open: name must not be empty"));
    }

    if !should_treat_as_path(&name)
        && crate::workspace::try_lookup(&name)
            .map_err(open_workspace_flow)?
            .is_some()
    {
        crate::context::request_document(
            runtime.as_ref(),
            crate::context::HostDocumentRequest::OpenVariable { name },
        )
        .await
        .map_err(|err| open_flow_error("open", err))?;
        return RootedValue::new(empty_array()).map_err(|error| error.into_runtime_error());
    }

    let path = resolve_open_path(&name).await?;
    open_path(runtime.as_ref(), &path).await
}

async fn open_path(
    runtime: Option<&crate::context::RuntimeContext>,
    path: &Path,
) -> BuiltinResult<RootedValue> {
    let extension = normalized_extension(path);
    let path_text = path_to_string(path);

    if extension == "fig" {
        let args = [Value::string(path_text)];
        let opened = match runtime {
            Some(runtime) => {
                crate::call_builtin_async_with_outputs_in_context(runtime, "openfig", &args, 1)
                    .await
            }
            None => crate::call_builtin_async_with_outputs("openfig", &args, 1).await,
        }
        .map_err(|err| open_flow_error("open", err))?;
        return Ok(opened);
    }

    if extension == "mat" {
        let eval = crate::builtins::io::mat::load::evaluate(&[Value::string(path_text)])
            .await
            .map_err(|err| open_flow_error("open", err))?;
        return eval.first_output();
    }

    if let Some(handler_result) = call_extension_handler(runtime, &extension, path).await? {
        return Ok(handler_result);
    }

    if is_viewer_or_editor_extension(&extension) {
        crate::context::request_document(
            runtime,
            crate::context::HostDocumentRequest::OpenFile {
                path: path.to_path_buf(),
            },
        )
        .await
        .map_err(|err| open_flow_error("open", err))?;
        return RootedValue::new(empty_array()).map_err(|error| error.into_runtime_error());
    }

    Err(open_error(
        &OPEN_ERROR_UNSUPPORTED,
        format!(
            "open: unsupported file type '{}' for '{}'",
            extension,
            path.display()
        ),
    ))
}

fn open_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
        Value::CharArray(chars)
            if chars.rows == 1
                && crate::builtins::common::shape::effective_rank(&chars.shape) <= 2 =>
        {
            crate::builtins::io::utf16::decode_utf16_with_error(chars.data(), || {
                open_error(
                    &OPEN_ERROR_NAME,
                    "open: name contains malformed UTF-16 text",
                )
            })
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned())
        }
        _ => Err(open_error(&OPEN_ERROR_NAME, OPEN_ERROR_NAME.message)),
    }
}

async fn call_extension_handler(
    runtime: Option<&crate::context::RuntimeContext>,
    extension: &str,
    path: &Path,
) -> BuiltinResult<Option<RootedValue>> {
    if extension.is_empty() {
        return Ok(None);
    }
    let handler_name = format!("open{extension}");
    let requested_outputs = crate::output_context::requested_output_count().unwrap_or(1);
    let requested_outputs = usize::from(requested_outputs > 0);
    let Some(result) = crate::user_functions::try_call_semantic_function_by_name_in_context(
        runtime,
        &handler_name,
        &[Value::string(path_to_string(path))],
        requested_outputs,
    )
    .await
    else {
        return Ok(None);
    };
    result
        .and_then(|sequence| {
            sequence
                .try_map_owned(
                    |sequence| {
                        let values = sequence.into_output_storage();
                        match values.len() {
                            0 => Ok(empty_array()),
                            1 => Ok(values
                                .try_into_first()
                                .map_err(|_| {
                                    crate::sequence::sequence_error_to_runtime(
                                        runmat_value::ValueSequenceError::SharedOutputStorage,
                                    )
                                })?
                                .expect("single output")),
                            count => Err(crate::runtime_error::semantic_error(
                                "RunMat:OpenHandlerOutputCount",
                                format!(
                        "{handler_name} returned {count} outputs where at most one was requested"
                    ),
                            )),
                        }
                    },
                    |error| error.into_runtime_error(),
                )
                .map(Some)
        })
        .map_err(|err| {
            let message = format!("{handler_name}: {}", err.message());
            let mut builder = build_runtime_error(message)
                .with_builtin(BUILTIN_NAME)
                .with_source(err);
            if let Some(identifier) = OPEN_ERROR_HANDLER.identifier {
                builder = builder.with_identifier(identifier);
            }
            builder.build()
        })
}

fn normalized_extension(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.trim_start_matches('.').to_ascii_lowercase())
        .unwrap_or_default()
}

fn is_viewer_or_editor_extension(extension: &str) -> bool {
    matches!(
        extension,
        "m" | "mlx"
            | "mlapp"
            | "txt"
            | "csv"
            | "json"
            | "xml"
            | "dat"
            | "html"
            | "htm"
            | "pdf"
            | "doc"
            | "docx"
            | "ppt"
            | "pptx"
            | "xls"
            | "xlsx"
            | "xlsm"
            | "mdl"
            | "slx"
            | "slxc"
            | "prj"
            | "mltbx"
            | "mlappinstall"
            | "exe"
    )
}

fn requested_output_count() -> usize {
    crate::output_count::current_output_count()
        .or_else(crate::output_context::requested_output_count)
        .unwrap_or(1)
}

fn empty_array() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn open_error(error: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn open_flow_error(context: &str, err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{context}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn open_workspace_flow(mut error: RuntimeError) -> RuntimeError {
    error.context = error.context.with_builtin(BUILTIN_NAME);
    error
}
