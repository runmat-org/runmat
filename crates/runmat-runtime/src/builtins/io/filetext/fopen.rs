//! MATLAB-compatible `fopen` builtin exposing host file streams.

use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinErrorDescriptor, FOPEN_ERROR_INTERNAL, FOPEN_ERROR_INVALID_ENCODING,
    FOPEN_ERROR_INVALID_INPUT, FOPEN_ERROR_INVALID_MACHINEFMT, FOPEN_ERROR_INVALID_PERMISSION,
    FOPEN_INTEGER_ID_EXTENSION, FOPEN_LEGACY_ALL_EXTENSION, FOPEN_RESIDENT_ID_EXTENSION,
    FOPEN_SINGLE_ID_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{FOPEN_DESCRIPTOR, FOPEN_INTEGER_CAPABILITIES as INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::Tensor;
use runmat_value::Value;

use crate::builtins::common::tensor;
use crate::builtins::io::filetext::{
    helpers::{extract_scalar_string, normalize_encoding_label},
    registry,
};
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fopen";

fn fopen_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fopen_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fopen_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FOPEN_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "fopen",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::filetext::fopen"
)]
async fn fopen_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let eval = evaluate_retained(runtime.as_ref(), args).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs(),
        ));
    }
    Ok(eval.first_output())
}

mod output;
mod permission;
#[path = "fopen/routes.rs"]
mod routes;
use routes::{handle_all, handle_open, handle_query};

pub use output::FopenEval;
use output::OpenOutputs;
use permission::Permission;

pub async fn evaluate(args: &[Value]) -> BuiltinResult<FopenEval> {
    evaluate_in(None, args).await
}

pub async fn evaluate_in(
    runtime: Option<&crate::context::RuntimeContext>,
    args: &[Value],
) -> BuiltinResult<FopenEval> {
    let args = crate::call::BuiltinArguments::from_borrowed(args, runtime)?;
    evaluate_retained(runtime, args).await
}

async fn evaluate_retained(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<FopenEval> {
    if args.is_empty()
        || args
            .first()
            .is_some_and(|value| matches_keyword(value, "all"))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FOPEN_LEGACY_ALL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if let Some(first) = args.first() {
        preflight_query_fid(first)?;
    }
    preflight_form(args.view())?;
    let filesystem = if let Some(first) = args.first() {
        if !matches_keyword(first, "all")
            && !is_numeric_value(first)
            && !matches!(first, Value::GpuTensor(_))
        {
            let path = value_to_path(first)?;
            match registry::FileFilesystem::capture(&path) {
                Ok(filesystem) => Some(std::sync::Arc::new(filesystem)),
                Err(error) => return Ok(FopenEval::open(OpenOutputs::failure(error.to_string()))),
            }
        } else {
            None
        }
    } else {
        None
    };
    let gathered =
        super::cooperation::await_io(runtime, BUILTIN_NAME, gather_args(runtime, &args)).await??;
    if gathered.is_empty() {
        return handle_all(runtime, crate::call::FunctionArgumentView::from_slice(&[]));
    }
    let first = &gathered[0];
    if matches_keyword(first, "all") {
        return handle_all(runtime, gathered.view().range(1..)?);
    }
    if is_numeric_value(first) {
        return handle_query(runtime, first, gathered.view().range(1..)?);
    }
    handle_open(runtime, first, gathered.view().range(1..)?, filesystem).await
}

/// Reject controls whose host value or resident metadata already proves the
/// call invalid. This runs before any unrelated resident operand is gathered.
fn preflight_form(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    let Some(first) = args.first() else {
        return Ok(());
    };
    if matches_keyword(first, "all") {
        if args.len() > 2 {
            return Err(fopen_error_with_detail(
                &FOPEN_ERROR_INVALID_INPUT,
                "too many input arguments",
            ));
        }
        if let Some(filter) = args.get(1) {
            parse_machinefmt(Some(filter))?;
        }
        return Ok(());
    }
    if is_numeric_value(first) || matches!(first, Value::GpuTensor(_)) {
        if args.len() != 1 {
            return Err(fopen_error_with_detail(
                &FOPEN_ERROR_INVALID_INPUT,
                "too many input arguments",
            ));
        }
        if let Value::GpuTensor(handle) = first {
            if runmat_accelerate_api::handle_is_logical(handle)
                || handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real
                || handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |n, d| n.checked_mul(*d))
                    != Some(1)
            {
                return Err(fopen_error_with_detail(
                    &FOPEN_ERROR_INVALID_INPUT,
                    "file identifier must be a numeric scalar",
                ));
            }
        } else {
            parse_fid(first)?;
        }
        return Ok(());
    }
    if args.len() > 4 {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "too many input arguments",
        ));
    }
    value_to_path(first)?;
    let permission = Permission::parse(args.get(1))?;
    parse_machinefmt(args.get(2))?;
    parse_encoding(args.get(3), &permission)?;
    Ok(())
}

async fn gather_args(
    runtime: Option<&crate::context::RuntimeContext>,
    args: &crate::call::BuiltinArguments,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    args.try_map_preserving_roots_async(args.len(), runtime, async |index| {
        gather_if_needed_async(&args[index])
            .await
            .map_err(map_control_flow)
    })
    .await
}

fn matches_keyword(value: &Value, keyword: &str) -> bool {
    extract_scalar_string(value)
        .map(|text| text.eq_ignore_ascii_case(keyword))
        .unwrap_or(false)
}

fn is_numeric_value(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.len() == 1)
}

fn preflight_query_fid(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FOPEN_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FOPEN_INTEGER_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F32 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FOPEN_SINGLE_ID_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FOPEN_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FOPEN_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FOPEN_SINGLE_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
    if let Value::Int(value) = value {
        return value.try_to_i32().ok_or_else(|| {
            fopen_error_with_detail(
                &FOPEN_ERROR_INVALID_INPUT,
                "file identifier is out of range",
            )
        });
    }
    if let Value::Tensor(tensor) = value {
        if tensor::is_scalar_tensor(tensor) {
            if let Some(integer) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return integer.try_to_i32().ok_or_else(|| {
                    fopen_error_with_detail(
                        &FOPEN_ERROR_INVALID_INPUT,
                        "file identifier is out of range",
                    )
                });
            }
        }
    }
    let num: f64 = value.try_into().map_err(|_| {
        fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "file identifier must be numeric",
        )
    })?;
    if !num.is_finite() {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "file identifier must be finite",
        ));
    }
    let rounded = num.round();
    if (rounded - num).abs() > f64::EPSILON {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "file identifier must be an integer",
        ));
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "file identifier is out of range",
        ));
    }
    Ok(rounded as i32)
}

fn value_to_path(value: &Value) -> BuiltinResult<PathBuf> {
    let raw = scalar_string(
        value,
        "expected filename as a string scalar or character vector",
        &FOPEN_ERROR_INVALID_INPUT,
    )?;
    normalize_path(&raw)
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "filename must not be empty",
        ));
    }
    Ok(Path::new(raw).to_path_buf())
}

fn parse_machinefmt(value: Option<&Value>) -> BuiltinResult<String> {
    match value {
        None => Ok("native".to_string()),
        Some(v) => {
            let text = scalar_string(
                v,
                "expected machine format as a string scalar or character vector",
                &FOPEN_ERROR_INVALID_MACHINEFMT,
            )?;
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Err(fopen_error_with_detail(
                    &FOPEN_ERROR_INVALID_MACHINEFMT,
                    "machine format must not be empty",
                ));
            }
            let lower = trimmed.to_ascii_lowercase();
            let collapsed: String = lower
                .chars()
                .filter(|c| !matches!(c, '-' | '_' | ' '))
                .collect();
            if matches!(collapsed.as_str(), "native" | "n" | "system" | "default") {
                return Ok("native".to_string());
            }
            if matches!(collapsed.as_str(), "l" | "le" | "littleendian" | "pc") {
                return Ok("ieee-le".to_string());
            }
            if matches!(collapsed.as_str(), "b" | "be" | "bigendian" | "mac") {
                return Ok("ieee-be".to_string());
            }
            if matches!(collapsed.as_str(), "vaxd" | "vaxg" | "cray") {
                return Ok(collapsed);
            }
            if let Some(suffix) = lower.strip_prefix("ieee-le") {
                return Ok(format!("ieee-le{suffix}"));
            }
            if let Some(suffix) = lower.strip_prefix("ieee-be") {
                return Ok(format!("ieee-be{suffix}"));
            }
            Err(fopen_error_with_detail(
                &FOPEN_ERROR_INVALID_MACHINEFMT,
                format!("unsupported machine format '{trimmed}'"),
            ))
        }
    }
}

fn parse_encoding(value: Option<&Value>, permission: &Permission) -> BuiltinResult<String> {
    match value {
        None => {
            if permission.binary {
                Ok("binary".to_string())
            } else {
                Ok("UTF-8".to_string())
            }
        }
        Some(v) => {
            let text = scalar_string(
                v,
                "expected encoding as a string scalar or character vector",
                &FOPEN_ERROR_INVALID_ENCODING,
            )?;
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Err(fopen_error_with_detail(
                    &FOPEN_ERROR_INVALID_ENCODING,
                    "encoding name must not be empty",
                ));
            }
            Ok(normalize_encoding_label(trimmed))
        }
    }
}

fn scalar_string<'a>(
    value: &'a Value,
    err_detail: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<std::borrow::Cow<'a, str>> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => {
            Ok(std::borrow::Cow::Borrowed(s.as_str()))
        }
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                fopen_error_with_detail(error, err_detail)
            })
            .map(std::borrow::Cow::Owned)
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => {
            Ok(std::borrow::Cow::Borrowed(
                sa.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str(),
            ))
        }
        _ => Err(fopen_error_with_detail(error, err_detail)),
    }
}

fn make_cell_column(values: Vec<Value>) -> BuiltinResult<Value> {
    let len = values.len();
    if len == 0 {
        make_cell(values, 0, 0).map_err(|err| fopen_error_with_detail(&FOPEN_ERROR_INTERNAL, &err))
    } else {
        make_cell(values, len, 1)
            .map_err(|err| fopen_error_with_detail(&FOPEN_ERROR_INTERNAL, &err))
    }
}

#[cfg(test)]
pub(crate) mod tests;
