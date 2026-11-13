//! MATLAB-compatible `importdata` builtin for legacy text imports.

use std::path::{Path, PathBuf};

#[cfg(test)]
use runmat_builtins::IMPORTDATA_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, IMPORTDATA_ERROR_ARGUMENT, IMPORTDATA_ERROR_IO, IMPORTDATA_ERROR_PARSE,
};
use runmat_filesystem as fs;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "importdata";

fn importdata_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    importdata_error_with(error, error.message)
}

fn importdata_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn importdata_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: crate::runtime_error::RuntimeErrorSourceBound,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "importdata",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::importdata"
)]
async fn importdata_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 3) {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_ARGUMENT,
            "importdata: at most three outputs are available",
        ));
    }
    if rest.len() > 2 {
        return Err(importdata_error(&IMPORTDATA_ERROR_ARGUMENT));
    }
    if crate::dispatcher::value_contains_gpu(&filename)
        || rest.iter().any(crate::dispatcher::value_contains_gpu)
    {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_ARGUMENT,
            "importdata: interactive gpuArray arguments are not supported",
        ));
    }
    let path = resolve_path(&filename)?;

    let delimiter = if let Some(value) = rest.first() {
        Some(parse_delimiter_arg(value)?)
    } else {
        None
    };
    let header_lines = if let Some(value) = rest.get(1) {
        Some(parse_header_lines(value)?)
    } else {
        None
    };

    let input = input::ImportInput::capture(path)?;
    let mut cooperation = cooperation::ImportCooperation::new(runtime.as_ref());
    cooperation.completed(0).await?;
    let bytes = cooperation
        .await_io(input.provider.read(&input.path), &input)
        .await?
        .map_err(|err| {
            importdata_error_with_source(
                &IMPORTDATA_ERROR_IO,
                format!(
                    "importdata: unable to read \"{}\" ({err})",
                    input.path.display()
                ),
                err,
            )
        })?;
    cooperation.completed(0).await?;
    let text = String::from_utf8(bytes).map_err(|err| {
        importdata_error_with_source(
            &IMPORTDATA_ERROR_IO,
            format!(
                "importdata: unable to decode \"{}\" ({err})",
                input.path.display()
            ),
            err,
        )
    })?;
    let imported =
        import_text_data(&text, delimiter.as_deref(), header_lines, &mut cooperation).await?;
    input.check_resource()?;
    cooperation.completed(0).await?;
    if let Some(output_count) = crate::output_count::current_output_count() {
        if output_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        let outputs = vec![
            imported.value,
            Value::CharArray(runmat_value::CharArray::new_row(&imported.delimiter)),
            Value::Num(imported.header_lines as f64),
        ];
        return Ok(crate::output_count::output_list_with_padding(
            output_count,
            outputs,
        ));
    }
    Ok(imported.value)
}

mod text;
mod input;
mod cooperation;
use text::import_text_data;

fn parse_delimiter_arg(value: &Value) -> BuiltinResult<String> {
    let text = string_scalar(value, "delimiterIn")?;
    match text.as_str() {
        "\\t" => Ok("\t".to_string()),
        "\\n" => Ok("\n".to_string()),
        "\\r" => Ok("\r".to_string()),
        _ => Ok(text),
    }
}

fn parse_header_lines(value: &Value) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer.try_to_usize().ok_or_else(|| {
            importdata_error_with(
                &IMPORTDATA_ERROR_ARGUMENT,
                "importdata: headerlinesIn must be a nonnegative integer scalar",
            )
        });
    }

    let raw = match value {
        Value::Num(n) => *n,
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => tensor::tensor_value_f64(t, 0),
        _ => {
            return Err(importdata_error_with(
                &IMPORTDATA_ERROR_ARGUMENT,
                "importdata: headerlinesIn must be a nonnegative integer scalar",
            ));
        }
    };
    if !raw.is_finite() || raw < 0.0 || raw.fract() != 0.0 {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_ARGUMENT,
            "importdata: headerlinesIn must be a nonnegative integer scalar",
        ));
    }
    if raw > usize::MAX.saturating_sub(1) as f64 {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_ARGUMENT,
            "importdata: headerlinesIn is too large",
        ));
    }
    let parsed = raw.round() as usize;
    if parsed as f64 != raw || parsed == usize::MAX {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_ARGUMENT,
            "importdata: headerlinesIn is too large",
        ));
    }
    Ok(parsed)
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    let text = crate::builtins::common::validation::value_text_scalar_string(value)
        .ok_or_else(|| importdata_error(&IMPORTDATA_ERROR_ARGUMENT))?;
    normalize_path(&text)
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(importdata_error_with(
            &IMPORTDATA_ERROR_ARGUMENT,
            "importdata: filename must not be empty",
        ));
    }
    let expanded = expand_user_path(raw, BUILTIN_NAME)
        .map_err(|msg| importdata_error_with(&IMPORTDATA_ERROR_ARGUMENT, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
        importdata_error_with(
            &IMPORTDATA_ERROR_ARGUMENT,
            format!("importdata: expected {context} as a string scalar or character vector"),
        )
    })
}

#[cfg(test)]
#[path = "importdata/tests.rs"]
mod tests;
