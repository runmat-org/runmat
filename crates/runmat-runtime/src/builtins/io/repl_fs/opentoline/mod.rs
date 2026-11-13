//! Typed editor navigation.
mod position;
#[cfg(test)]
mod tests;
use crate::builtins::common::path_search::{find_file_with_extensions, GENERAL_FILE_EXTENSIONS};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use position::{admit_position, resolve_position};
use runmat_builtins::{BuiltinErrorDescriptor, BuiltinExtensionDescriptor};
pub use runmat_builtins::{
    OPENTOLINE_DESCRIPTOR, OPENTOLINE_ERROR_ARG_COUNT, OPENTOLINE_ERROR_FILENAME,
    OPENTOLINE_ERROR_NOT_FOUND, OPENTOLINE_ERROR_OPTION, OPENTOLINE_ERROR_OUTPUT_COUNT,
    OPENTOLINE_ERROR_POSITION, OPENTOLINE_EXTENSIONS, OPENTOLINE_INTEGER_CAPABILITIES,
    OPENTOLINE_INTEGER_COLUMN_EXTENSION, OPENTOLINE_INTEGER_LINE_EXTENSION,
    OPENTOLINE_RESIDENT_POSITION_EXTENSION,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
use std::path::PathBuf;
const BUILTIN_NAME: &str = "opentoline";
#[runtime_builtin(
    name = "opentoline",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::repl_fs::opentoline"
)]
async fn opentoline_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !(2..=4).contains(&args.len()) {
        return Err(opentoline_error(
            &OPENTOLINE_ERROR_ARG_COUNT,
            OPENTOLINE_ERROR_ARG_COUNT.message,
        ));
    }
    if requested_output_count() > 0 {
        return Err(opentoline_error(
            &OPENTOLINE_ERROR_OUTPUT_COUNT,
            format!(
                "opentoline: expected no output arguments, got {}",
                requested_output_count()
            ),
        ));
    }

    gate_position_extensions(&args[1], &OPENTOLINE_INTEGER_LINE_EXTENSION)?;
    if let Some(column) = args.get(2) {
        gate_position_extensions(column, &OPENTOLINE_INTEGER_COLUMN_EXTENSION)?;
    }
    let filename = filename_arg(&args[0])?;
    if filename.is_empty() {
        return Err(opentoline_error(
            &OPENTOLINE_ERROR_FILENAME,
            "opentoline: filename must not be empty",
        ));
    }
    let line = admit_position(&args[1], "line")?;
    let column = args
        .get(2)
        .map(|value| admit_position(value, "column"))
        .transpose()?;
    if let Some(option) = args.get(3) {
        parse_option(option)?;
    }

    let line = resolve_position(line, "line").await?;
    let column = match column {
        Some(value) => resolve_position(value, "column").await?,
        None => 1,
    };
    let path = resolve_file(&filename).await?;
    let location = crate::context::HostTextLocation {
        line: std::num::NonZeroUsize::new(line).ok_or_else(|| {
            opentoline_error(
                &OPENTOLINE_ERROR_POSITION,
                OPENTOLINE_ERROR_POSITION.message,
            )
        })?,
        column: std::num::NonZeroUsize::new(column).ok_or_else(|| {
            opentoline_error(
                &OPENTOLINE_ERROR_POSITION,
                OPENTOLINE_ERROR_POSITION.message,
            )
        })?,
        selection: if args.len() == 4 {
            crate::context::HostTextSelection::Select
        } else {
            crate::context::HostTextSelection::Caret
        },
    };
    crate::context::request_document(
        runtime.as_ref(),
        crate::context::HostDocumentRequest::Navigate { path, location },
    )
    .await
    .map_err(|err| opentoline_flow_error("opentoline", err))?;

    Ok(empty_array())
}

fn gate_position_extensions(
    value: &Value,
    integer_extension: &BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(integer_extension, BUILTIN_NAME)?;
    }
    if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &OPENTOLINE_RESIDENT_POSITION_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn filename_arg(value: &Value) -> BuiltinResult<String> {
    super::file_dialog::try_scalar_text(value, "filename", |detail| {
        opentoline_error(&OPENTOLINE_ERROR_FILENAME, detail)
    })?
    .ok_or_else(|| {
        opentoline_error(
            &OPENTOLINE_ERROR_FILENAME,
            OPENTOLINE_ERROR_FILENAME.message,
        )
    })
}
fn parse_option(value: &Value) -> BuiltinResult<()> {
    let option = super::file_dialog::try_scalar_text(value, "option", |detail| {
        opentoline_error(&OPENTOLINE_ERROR_OPTION, detail)
    })?
    .ok_or_else(|| opentoline_error(&OPENTOLINE_ERROR_OPTION, OPENTOLINE_ERROR_OPTION.message))?;
    if option.eq_ignore_ascii_case("select") {
        Ok(())
    } else {
        Err(opentoline_error(
            &OPENTOLINE_ERROR_OPTION,
            format!("opentoline: unsupported option '{option}'"),
        ))
    }
}
async fn resolve_file(name: &str) -> BuiltinResult<PathBuf> {
    let path = find_file_with_extensions(name, GENERAL_FILE_EXTENSIONS, BUILTIN_NAME)
        .await
        .map_err(|error| error.into_runtime_error(BUILTIN_NAME))?
        .ok_or_else(|| {
            opentoline_error(
                &OPENTOLINE_ERROR_NOT_FOUND,
                format!("opentoline: file was not found: {name}"),
            )
        })?;
    Ok(vfs::canonicalize_async(&path).await.unwrap_or(path))
}

fn requested_output_count() -> usize {
    crate::output_count::current_output_count()
        .or_else(crate::output_context::requested_output_count)
        .unwrap_or(0)
}

fn empty_array() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn opentoline_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn opentoline_flow_error(context: &str, err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{context}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
