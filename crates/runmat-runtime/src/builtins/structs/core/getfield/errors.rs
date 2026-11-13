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
    // Forward the exact funded diagnostic when no presentation is requested.
    // Cancellation is execution control, not getfield-domain text to rewrite.
    // In particular, do not add a wrapper that erases its outward provenance
    // or changes the immediate typed source into another RuntimeError.
    if prefix.is_none() || crate::runtime_error::error_has_control_cancellation_origin(&err) {
        return err;
    }
    let mut message = err.message().to_string();
    if let Some(prefix) = prefix {
        if !message.starts_with(prefix) {
            message = format!("{prefix}{message}");
        }
    }
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_failure_origin(err.failure_origin());
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}

pub(super) fn remap_index(err: RuntimeError, prefix: Option<&str>) -> RuntimeError {
    if crate::runtime_error::error_has_control_cancellation_origin(&err) {
        return err;
    }
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
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_failure_origin(err.failure_origin());
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}

pub(crate) fn is_undefined_function(err: &RuntimeError) -> bool {
    err.identifier() == Some(crate::IDENT_UNDEFINED_FUNCTION)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_error::RuntimeFailureOrigin;
    use std::error::Error;

    #[test]
    fn remapping_preserves_control_text_and_direct_typed_source_without_rewrapping() {
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        runtime.set_error_namespace("SelectedGetfield");
        let mappers: [fn(RuntimeError, Option<&str>) -> RuntimeError; 2] = [remap, remap_index];
        for map in mappers {
            for prefix in [None, Some("getfield: ")] {
                let original = crate::execution::ExecutionServiceError::Cancelled
                    .into_runtime_error_in_context(Some(&runtime));
                let message = original.message.as_ptr();
                let identifier = original.identifier.as_ref().unwrap().as_ptr();
                let source = original.source().unwrap() as *const (dyn Error + 'static);
                let error = map(original, prefix);
                assert_eq!(error.identifier(), Some("SelectedGetfield:Cancelled"));
                assert_eq!(
                    error.failure_origin(),
                    RuntimeFailureOrigin::ExecutionControlCancellation
                );
                assert_eq!(error.message.as_ptr(), message);
                assert_eq!(error.identifier.as_ref().unwrap().as_ptr(), identifier);
                assert!(std::ptr::eq(error.source().unwrap(), source));
                assert!(matches!(
                    error
                        .source()
                        .unwrap()
                        .downcast_ref::<crate::execution::ExecutionServiceError>(),
                    Some(crate::execution::ExecutionServiceError::Cancelled)
                ));
            }
        }
    }

    #[test]
    fn ordinary_presentation_keeps_existing_origin_and_does_not_infer_control_from_text() {
        let original = build_runtime_error("ordinary failure")
            .with_identifier("SelectedGetfield:Cancelled")
            .with_failure_origin(RuntimeFailureOrigin::LanguageEvaluation)
            .with_source(std::io::Error::other("ordinary source"))
            .build();
        let message = original.message.as_ptr();
        let direct = remap(original, None);
        assert_eq!(direct.message.as_ptr(), message);
        assert!(direct.source().unwrap().is::<std::io::Error>());

        let prefixed = remap(direct, Some("getfield: "));
        assert_eq!(prefixed.message(), "getfield: ordinary failure");
        assert_eq!(prefixed.identifier(), Some("SelectedGetfield:Cancelled"));
        assert_eq!(
            prefixed.failure_origin(),
            RuntimeFailureOrigin::LanguageEvaluation
        );
        assert!(!crate::runtime_error::error_has_control_cancellation_origin(&prefixed));
        let indexed = remap_index(prefixed, Some("getfield: "));
        assert_eq!(indexed.identifier(), GETFIELD_ERROR_INTERNAL.identifier);
        assert_eq!(
            indexed.failure_origin(),
            RuntimeFailureOrigin::LanguageEvaluation
        );
        assert!(indexed.source().unwrap().is::<RuntimeError>());
    }
}
