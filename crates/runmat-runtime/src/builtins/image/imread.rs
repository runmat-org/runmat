use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Duration;

use image::io::Reader as ImageReader;
use image::{DynamicImage, ImageFormat};
use runmat_builtins::{
    BuiltinErrorDescriptor, IMREAD_ERROR_DECODE, IMREAD_ERROR_FILE_READ, IMREAD_ERROR_HTTP_STATUS,
    IMREAD_ERROR_INVALID_ARGUMENT, IMREAD_ERROR_INVALID_FILENAME, IMREAD_ERROR_INVALID_FILE_URL,
    IMREAD_ERROR_INVALID_FORMAT, IMREAD_ERROR_INVALID_HEADER, IMREAD_ERROR_NETWORK,
    IMREAD_ERROR_SHAPE, IMREAD_ERROR_TIMEOUT, IMREAD_ERROR_TOO_MANY_INPUTS,
    IMREAD_ERROR_TOO_MANY_OUTPUTS, IMREAD_ERROR_UNSUPPORTED_FORMAT,
    IMREAD_ERROR_UNSUPPORTED_SCHEME,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericStorage, Tensor, Value};
use url::Url;

use crate::builtins::image::io_cooperation::ImageCooperation;
use crate::builtins::io::http::transport::{
    self, HttpMethod, HttpRequest, TransportError, TransportErrorKind,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod decode;
mod source;
#[cfg(test)]
mod tests;

use self::decode::{decode_image_bytes, empty_tensor_value, materialize_image};
use self::source::{parse_format_hint, read_source_bytes, string_arg};

const BUILTIN_NAME: &str = "imread";
const DEFAULT_TIMEOUT_SECONDS: f64 = 60.0;
const DEFAULT_USER_AGENT: &str = "RunMat imread/0.0";
fn imread_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn imread_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("imread:") {
        detail.to_string()
    } else {
        format!("{}: {}", error.message, detail)
    };
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn imread_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: E,
) -> RuntimeError
where
    E: crate::runtime_error::RuntimeErrorSourceBound,
{
    build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(error.identifier.unwrap_or("RunMat:imread:Internal"))
        .with_source(source)
        .build()
}

#[runtime_builtin(
    name = "imread",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::imread"
)]
async fn imread_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    source: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    if requested_outputs.is_some_and(|count| count > 3) {
        return Err(imread_error(&IMREAD_ERROR_TOO_MANY_OUTPUTS));
    }
    if rest.len() > 1 {
        return Err(imread_error(&IMREAD_ERROR_TOO_MANY_INPUTS));
    }
    let source = string_arg("filename", &source)?;
    if source.is_empty() {
        return Err(imread_error_with_detail(
            &IMREAD_ERROR_INVALID_FILENAME,
            "filename must not be empty",
        ));
    }

    let format_hint = match (rest.len(), rest.first()) {
        (0, _) => None,
        (1, Some(format)) => Some(parse_format_hint(&string_arg("format", format)?)?),
        _ => return Err(imread_error(&IMREAD_ERROR_TOO_MANY_INPUTS)),
    };

    let mut cooperation = ImageCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    cooperation.completed(0).await?;
    let bytes = read_source_bytes(&source, runtime.as_ref()).await?;
    cooperation.completed(0).await?;
    let decoded = decode_image_bytes(&bytes, format_hint)?;
    let materialized = materialize_image(decoded, &mut cooperation).await?;

    match requested_outputs {
        None => Ok(Value::Tensor(materialized.image)),
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![Value::Tensor(materialized.image)])),
        Some(2) => Ok(Value::output_list(vec![
            Value::Tensor(materialized.image),
            empty_tensor_value()?,
        ])),
        Some(3) => Ok(Value::output_list(vec![
            Value::Tensor(materialized.image),
            empty_tensor_value()?,
            materialized
                .alpha
                .map(Value::Tensor)
                .unwrap_or(empty_tensor_value()?),
        ])),
        Some(_) => Err(imread_error(&IMREAD_ERROR_TOO_MANY_OUTPUTS)),
    }
}
