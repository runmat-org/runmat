pub use runmat_async::{
    runtime_error as build_runtime_error, CallFrame, CallFrameDiagnosticSource,
    CallFrameSourceOwner, ErrorContext, GpuGatherRetry, RuntimeError, RuntimeErrorBuilder,
    RuntimeErrorOwner, RuntimeErrorSourceBound, RuntimeFailureOrigin, SharedRuntimeErrorSource,
};

mod cleanup;
#[cfg(test)]
mod provider_sources;
pub use cleanup::complete_with_cleanup;

/// An evaluated assertion condition failed. The originating builtin attaches
/// this cause independently of its user-selected diagnostic identifier.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("assertion condition failed")]
pub struct RuntimeAssertionFailure;

/// Attach an owning provider failure without erasing a nested language error.
/// Provider failures are not implicit gather-and-retry requests.
pub fn with_provider_execution_source(
    mut fallback: RuntimeError,
    source: runmat_accelerate_api::ProviderExecutionError,
) -> RuntimeError {
    let mut cursor: &(dyn std::error::Error + 'static) = &source;
    loop {
        if let Some(runtime) = cursor.downcast_ref::<RuntimeError>() {
            fallback.message = runtime.message.clone();
            fallback.identifier = runtime.identifier.clone();
            fallback.span = runtime.span;
            fallback.context = runtime.context.clone();
            fallback.failure_origin = runtime.failure_origin;
            break;
        }
        let Some(next) = cursor.source() else { break };
        cursor = next;
    }
    fallback.gpu_gather_retry = GpuGatherRetry::Never;
    fallback.source = Some(Box::new(source));
    fallback
}

/// Whether an error chain carries one of Runtime's typed execution-control
/// cancellation causes. Presentation owners use this after an outer scope has
/// retained the reason (for example, user cancellation versus a deadline).
/// Identifiers and messages are deliberately not authority for this decision.
pub fn error_has_control_cancellation_origin(error: &RuntimeError) -> bool {
    error.failure_origin() == RuntimeFailureOrigin::ExecutionControlCancellation
}

/// Construct a language-semantic runtime error in the active session's
/// namespace. Executors may attach frame and source-span information, but the
/// identifier and message are executor-neutral runtime semantics.
pub fn semantic_error(identifier: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(semantic_identifier(identifier))
        .build()
}

fn semantic_identifier(identifier: &str) -> String {
    let suffix = identifier
        .split_once(':')
        .map_or(identifier, |(_, suffix)| suffix);
    let namespace =
        crate::context::legacy::error_namespace().unwrap_or_else(|| "RunMat".to_string());
    format!("{namespace}:{suffix}")
}

/// Preserve an owned operational cause under a language-semantic diagnostic.
/// The source retains its own diagnostic allocation and error chain; the
/// wrapper owns only its new message and namespace-qualified identifier.
pub fn semantic_error_with_source(
    identifier: &str,
    message: impl Into<String>,
    source: impl RuntimeErrorSourceBound,
) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(semantic_identifier(identifier))
        .with_source(source)
        .build()
}

/// Materialize the language-level exception value caught by `try`/`catch`.
///
/// Executors own control transfer and frame/source decoration; Runtime owns
/// the stable conversion from its semantic error contract to `MException`.
pub fn exception_from_error(error: impl Into<RuntimeErrorOwner>) -> runmat_value::MException {
    let original = error.into();
    let error = original.as_ref();
    let (identifier, message) = if let Some(identifier) = error.identifier() {
        (identifier.to_string(), error.message().to_string())
    } else if let Some(index) = error.message().rfind(": ") {
        let (identifier, detail) = error.message().split_at(index);
        (
            exception_identifier(identifier),
            detail.trim_start_matches(':').trim().to_string(),
        )
    } else if let Some(index) = error.message().rfind(':') {
        let (identifier, detail) = error.message().split_at(index);
        (
            exception_identifier(identifier),
            detail.trim_start_matches(':').trim().to_string(),
        )
    } else {
        (exception_identifier(""), error.message().to_string())
    };
    let stack = if error.context.call_stack.is_empty() {
        error
            .context
            .call_frames
            .iter()
            .map(|frame| frame.function.clone())
            .collect()
    } else {
        error.context.call_stack.clone()
    };
    runmat_value::MException::from_runtime_error(identifier, message, stack, original)
}

/// Rethrow an exception using its original typed runtime failure when it was
/// caught from execution. User-constructed exceptions retain their public
/// identifier, message, and stack without inventing an opaque source.
pub fn error_from_exception(exception: runmat_value::MException) -> RuntimeError {
    let parts = exception.into_parts();
    if let Some(original) = parts.original {
        return RuntimeError::from_shared_original(original);
    }
    build_runtime_error(parts.message.into_string_unbounded())
        .with_identifier(parts.identifier.into_string_unbounded())
        .with_call_stack(parts.stack.into_strings_unbounded())
        .build()
}

fn exception_identifier(identifier: &str) -> String {
    if identifier.trim().is_empty() {
        let namespace =
            crate::context::legacy::error_namespace().unwrap_or_else(|| "RunMat".to_string());
        format!("{namespace}:error")
    } else {
        identifier.trim().to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayErrorKind {
    UnsupportedSchema,
    PayloadTooLarge,
    DecodeFailed,
    ExportRejected,
    ImportRejected,
}

impl ReplayErrorKind {
    pub fn identifier(self) -> &'static str {
        match self {
            Self::UnsupportedSchema => "RunMat:ReplayUnsupportedSchema",
            Self::PayloadTooLarge => "RunMat:ReplayPayloadTooLarge",
            Self::DecodeFailed => "RunMat:ReplayDecodeFailed",
            Self::ExportRejected => "RunMat:ReplayExportRejected",
            Self::ImportRejected => "RunMat:ReplayImportRejected",
        }
    }
}

pub fn replay_error(kind: ReplayErrorKind, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("replay")
        .with_identifier(kind.identifier())
        .build()
}

pub fn replay_error_with_source(
    kind: ReplayErrorKind,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("replay")
        .with_identifier(kind.identifier())
        .with_source(source)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{RuntimeContext, RuntimeContextGuard};
    use crate::execution::RuntimeExecutionService;
    use std::error::Error;
    use std::rc::Rc;

    #[test]
    fn semantic_errors_use_the_active_session_namespace() {
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        context.set_error_namespace("Acme");
        let _scope = RuntimeContextGuard::enter(context);

        assert_eq!(
            semantic_error("MATLAB:badsubscript", "bad index").identifier(),
            Some("Acme:badsubscript")
        );
        assert_eq!(
            semantic_error("IndexOutOfBounds", "bad index").identifier(),
            Some("Acme:IndexOutOfBounds")
        );
    }

    #[test]
    fn control_cancellation_follows_retained_runtime_owners_but_not_text() {
        let original = crate::execution::ExecutionServiceError::Cancelled.into_runtime_error();
        let retained: RuntimeErrorOwner = original.into();
        let outward = RuntimeError::from_shared_original(retained.clone());
        assert!(error_has_control_cancellation_origin(&outward));

        let coincidental = build_runtime_error("execution was cancelled")
            .with_identifier("RunMat:parallel:Cancelled")
            .with_source(std::io::Error::other("ordinary language failure"))
            .build();
        assert!(!error_has_control_cancellation_origin(&coincidental));
    }

    #[test]
    fn typed_source_alone_does_not_mint_control_origin() {
        let checkpoint = build_runtime_error("checkpoint stopped")
            .with_source(
                crate::context::host_allocation::HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Cancelled,
                ),
            )
            .build();
        assert!(!error_has_control_cancellation_origin(&checkpoint));
    }

    #[test]
    fn semantic_source_keeps_the_original_diagnostic_lease_until_wrapper_drop() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        #[derive(Debug)]
        struct Lease(Arc<AtomicUsize>);
        impl Drop for Lease {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        context.set_error_namespace("Acme");
        let _scope = RuntimeContextGuard::enter(context);
        let drops = Arc::new(AtomicUsize::new(0));
        let original = build_runtime_error("typed inner")
            .with_identifier("Acme:Inner")
            .with_retained_diagnostic_allocation(runmat_async::RetainedDiagnosticAllocation::Owned(
                Box::new(Lease(drops.clone())),
            ))
            .build();
        let wrapped =
            semantic_error_with_source("RunMat:Workspace:LexicalPublication", "outer", original);
        assert_eq!(
            wrapped.identifier(),
            Some("Acme:Workspace:LexicalPublication")
        );
        assert_eq!(
            wrapped
                .source()
                .and_then(|source| source.downcast_ref::<RuntimeError>())
                .and_then(RuntimeError::identifier),
            Some("Acme:Inner")
        );
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(wrapped);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn caught_exception_materialization_preserves_structured_and_legacy_errors() {
        let structured = semantic_error("IndexOutOfBounds", "bad index");
        let exception = exception_from_error(structured);
        assert_eq!(exception.identifier, "RunMat:IndexOutOfBounds");
        assert_eq!(exception.message, "bad index");

        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        context.set_error_namespace("Acme");
        let _scope = RuntimeContextGuard::enter(context);
        let legacy = crate::build_runtime_error("legacy detail").build();
        let exception = exception_from_error(legacy);
        assert_eq!(exception.identifier, "Acme:error");
        assert_eq!(exception.message, "legacy detail");
    }

    #[test]
    fn rethrow_preserves_original_frames_policy_and_downcastable_source() {
        let original = build_runtime_error("inner failure")
            .with_identifier("RunMat:diagnostics:Inner")
            .with_builtin("source")
            .with_call_frames(vec![CallFrame {
                function: "origin".into(),
                source_id: Some(7),
                span: Some((3, 8)),
                source_owner: None,
                diagnostic_source: None,
            }])
            .with_span((3, 5).into())
            .with_phase("execution")
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .with_failure_origin(RuntimeFailureOrigin::LanguageEvaluation)
            .with_source(std::io::Error::other("typed original cause"))
            .build();
        let caught = exception_from_error(original);
        assert!(caught
            .stack
            .iter()
            .map(runmat_value::Utf8Text::as_str)
            .eq(["origin"]));
        let other_owner = caught.clone();
        let reraised = error_from_exception(caught);
        assert_eq!(reraised.identifier(), Some("RunMat:diagnostics:Inner"));
        assert_eq!(reraised.context.call_frames[0].source_id, Some(7));
        assert_eq!(reraised.context.phase.as_deref(), Some("execution"));
        assert_eq!(reraised.gpu_gather_retry, GpuGatherRetry::Never);
        assert_eq!(
            reraised.failure_origin,
            RuntimeFailureOrigin::LanguageEvaluation
        );
        let shared = reraised
            .source()
            .unwrap()
            .downcast_ref::<runmat_async::SharedRuntimeErrorSource>()
            .expect("exact original runtime error owner");
        assert!(shared.original().source().unwrap().is::<std::io::Error>());
        assert!(other_owner.original().is_some());
    }

    #[test]
    fn uniquely_owned_rethrow_recovers_exact_original_without_wrapper() {
        let original = build_runtime_error("inner failure")
            .with_identifier("RunMat:diagnostics:Inner")
            .with_source(std::io::Error::other("distinctive original cause"))
            .build();
        let reraised = error_from_exception(exception_from_error(original));
        assert!(reraised.source().unwrap().is::<std::io::Error>());
    }
}
