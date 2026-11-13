//! RunMat source packaging; this container is not a cryptographic boundary.
mod codec;
mod source;
#[cfg(test)]
mod tests;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
#[cfg(test)]
use codec::RUNMAT_PCODE_MAGIC;
pub use codec::{decode_pcode_bytes, encode_pcode_source, PcodeAlgorithm};
use runmat_builtins::{
    BuiltinErrorDescriptor, PCODE_ERROR_ARG_TYPE, PCODE_ERROR_EMPTY_ITEM,
    PCODE_ERROR_FILE_NOT_FOUND, PCODE_ERROR_FILE_READ, PCODE_ERROR_FILE_WRITE,
    PCODE_ERROR_INVALID_OPTION, PCODE_ERROR_INVALID_PCODE, PCODE_ERROR_NOT_ENOUGH_INPUTS,
    PCODE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
use source::{destination_for_source, resolve_sources};
pub use source::{prefer_pcode_source_path, read_source_text_async, PcodeSourceReadError};
use std::path::PathBuf;
const BUILTIN_NAME: &str = "pcode";

#[derive(Debug, Clone)]
pub struct PcodeResult {
    pub generated: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
struct PcodeOptions {
    algorithm: PcodeAlgorithm,
    inplace: bool,
    items: Vec<String>,
}

#[runtime_builtin(
    name = "pcode",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::repl_fs::pcode"
)]
async fn pcode_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate_view(args.view()).await?;
    Ok(empty_array())
}

pub async fn evaluate(args: &[Value]) -> BuiltinResult<PcodeResult> {
    evaluate_view(crate::call::FunctionArgumentView::from_slice(args)).await
}

pub async fn evaluate_view(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<PcodeResult> {
    if requested_output_count() > 0 {
        return Err(pcode_error(&PCODE_ERROR_TOO_MANY_OUTPUTS));
    }
    let options = parse_options(args)?;
    let sources = resolve_sources(&options.items).await?;
    if sources.is_empty() {
        return Err(pcode_error(&PCODE_ERROR_FILE_NOT_FOUND));
    }

    let mut generated = Vec::with_capacity(sources.len());
    for source in sources {
        let text = vfs::read_to_string_async(&source).await.map_err(|err| {
            pcode_error_with_detail(
                &PCODE_ERROR_FILE_READ,
                format!("{} ({err})", source.display()),
            )
        })?;
        let destination = destination_for_source(&source, options.inplace)?;
        if let Some(parent) = destination.parent() {
            if !parent.as_os_str().is_empty() {
                vfs::create_dir_all_async(parent).await.map_err(|err| {
                    pcode_error_with_detail(
                        &PCODE_ERROR_FILE_WRITE,
                        format!("{} ({err})", parent.display()),
                    )
                })?;
            }
        }
        let source_name = source.to_string_lossy();
        let encoded = encode_pcode_source(&text, &source_name, options.algorithm);
        vfs::write_async(&destination, encoded.as_bytes())
            .await
            .map_err(|err| {
                pcode_error_with_detail(
                    &PCODE_ERROR_FILE_WRITE,
                    format!("{} ({err})", destination.display()),
                )
            })?;
        generated.push(destination);
    }
    Ok(PcodeResult { generated })
}

fn parse_options(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<PcodeOptions> {
    if args.is_empty() {
        return Err(pcode_error(&PCODE_ERROR_NOT_ENOUGH_INPUTS));
    }
    let mut options = PcodeOptions {
        algorithm: PcodeAlgorithm::R2007b,
        inplace: false,
        items: Vec::new(),
    };

    for arg in args {
        let text =
            value_to_string_scalar(arg)?.ok_or_else(|| pcode_error(&PCODE_ERROR_ARG_TYPE))?;
        if text.is_empty() {
            return Err(pcode_error(&PCODE_ERROR_EMPTY_ITEM));
        }
        let lowered = text.to_ascii_lowercase();
        match lowered.as_str() {
            "-inplace" => options.inplace = true,
            "-r2007b" => options.algorithm = PcodeAlgorithm::R2007b,
            "-r2022a" => options.algorithm = PcodeAlgorithm::R2022a,
            option if option.starts_with('-') => {
                return Err(pcode_error_with_detail(&PCODE_ERROR_INVALID_OPTION, text));
            }
            _ => options.items.push(text),
        }
    }

    if options.items.is_empty() {
        return Err(pcode_error(&PCODE_ERROR_NOT_ENOUGH_INPUTS));
    }
    Ok(options)
}

fn value_to_string_scalar(value: &Value) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(Some(text.as_str().to_owned()))
        }
        Value::CharArray(array)
            if array.rows == 1
                && crate::builtins::common::shape::effective_rank(&array.shape) <= 2 =>
        {
            crate::builtins::io::utf16::decode_utf16_with_error(array.data(), || {
                pcode_error(&PCODE_ERROR_ARG_TYPE)
            })
            .map(Some)
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            ))
        }
        _ => Ok(None),
    }
}

fn empty_array() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn requested_output_count() -> usize {
    crate::output_count::current_output_count()
        .or_else(crate::output_context::requested_output_count)
        .unwrap_or(0)
}

fn pcode_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    pcode_error_with_detail(error, "")
}

pub fn invalid_pcode_runtime_error(detail: impl AsRef<str>) -> RuntimeError {
    pcode_error_with_detail(&PCODE_ERROR_INVALID_PCODE, detail)
}

fn pcode_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
