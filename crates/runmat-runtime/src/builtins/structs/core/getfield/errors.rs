use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, GETFIELD_ERROR_INDEX_INVALID, GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
    GETFIELD_ERROR_INDEX_SHAPE, GETFIELD_ERROR_INTERNAL,
};

pub(super) const BUILTIN_NAME: &str = "getfield";

pub(super) fn internal(message: impl Into<String>) -> RuntimeError {
    with_message(message, &GETFIELD_ERROR_INTERNAL)
}

pub(super) fn from_descriptor(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    with_message(error.message, error)
}

pub(super) fn with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn remap(err: RuntimeError, prefix: Option<&str>) -> RuntimeError {
    let mut message = err.message().to_string();
    if let Some(prefix) = prefix {
        if !message.starts_with(prefix) {
            message = format!("{prefix}{message}");
        }
    }
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}

pub(super) fn remap_index(err: RuntimeError, prefix: Option<&str>) -> RuntimeError {
    let descriptor = match err.identifier() {
        Some("RunMat:IndexOutOfBounds" | "RunMat:SubscriptOutOfBounds") => {
            &GETFIELD_ERROR_INDEX_OUT_OF_BOUNDS
        }
        Some("RunMat:IndexShape" | "RunMat:ShapeMismatch") => &GETFIELD_ERROR_INDEX_SHAPE,
        Some(
            "RunMat:IndexStepZero"
            | "RunMat:MissingIndex"
            | "RunMat:MissingNumericIndex"
            | "RunMat:NumericRequired"
            | "RunMat:ScalarRequired"
            | "RunMat:UnsupportedIndexType",
        ) => &GETFIELD_ERROR_INDEX_INVALID,
        _ => &GETFIELD_ERROR_INTERNAL,
    };
    remap_with_descriptor(err, prefix, descriptor)
}

fn remap_with_descriptor(
    err: RuntimeError,
    prefix: Option<&str>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut message = err.message().to_string();
    if let Some(prefix) = prefix {
        if !message.starts_with(prefix) {
            message = format!("{prefix}{message}");
        }
    }
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}

pub(crate) fn is_undefined_function(err: &RuntimeError) -> bool {
    err.identifier() == Some(crate::IDENT_UNDEFINED_FUNCTION)
}
