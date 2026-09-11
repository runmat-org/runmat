use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, SETFIELD_ERROR_INDEX_INVALID, SETFIELD_ERROR_INDEX_OUT_OF_BOUNDS,
    SETFIELD_ERROR_INDEX_SHAPE, SETFIELD_ERROR_INTERNAL, SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
    SETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS, SETFIELD_ERROR_PROPERTY_STATIC_ACCESS,
};

pub(super) const BUILTIN_NAME: &str = "setfield";

pub(super) fn internal(message: impl Into<String>) -> RuntimeError {
    with_message(
        format!("{}: {}", SETFIELD_ERROR_INTERNAL.message, message.into()),
        &SETFIELD_ERROR_INTERNAL,
    )
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

pub(super) fn private_access(message: impl Into<String>) -> RuntimeError {
    with_message(message, &SETFIELD_ERROR_PROPERTY_PRIVATE_ACCESS)
}

pub(super) fn static_access(message: impl Into<String>) -> RuntimeError {
    with_message(message, &SETFIELD_ERROR_PROPERTY_STATIC_ACCESS)
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
            &SETFIELD_ERROR_INDEX_OUT_OF_BOUNDS
        }
        Some("RunMat:IndexShape" | "RunMat:ShapeMismatch" | "RunMat:Assignment") => {
            &SETFIELD_ERROR_INDEX_SHAPE
        }
        Some(
            "RunMat:IndexStepZero"
            | "RunMat:MissingIndex"
            | "RunMat:MissingNumericIndex"
            | "RunMat:NumericRequired"
            | "RunMat:ScalarRequired"
            | "RunMat:UnsupportedIndexType",
        ) => &SETFIELD_ERROR_INDEX_INVALID,
        Some(
            "RunMat:ObjectArrayAssignment" | "RunMat:StructAssignment" | "RunMat:StructIndexing",
        ) => &SETFIELD_ERROR_NON_STRUCT_ASSIGNMENT,
        _ => &SETFIELD_ERROR_INTERNAL,
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
