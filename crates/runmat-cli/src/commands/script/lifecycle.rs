//! Preserve delivered language failure when foreign lifecycle also fails.
use runmat_core::abi::{DiagnosticSeverity, ExecutionOutcome};
use runmat_runtime::RuntimeError;

#[expect(
    clippy::result_large_err,
    reason = "Preserves the canonical owned RuntimeError contract; its storage policy belongs to runmat-runtime"
)]
pub(super) fn complete(
    outcome: &ExecutionOutcome,
    cleanup: Result<(), RuntimeError>,
) -> Result<(), RuntimeError> {
    let Some(primary) = outcome
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
    else {
        return cleanup;
    };
    let Err(cleanup) = cleanup else {
        return Ok(());
    };
    // The execution ABI has already projected the original error into this
    // structured diagnostic. Do not replace its code, span, callstack or text
    // with the later host-retirement failure; the caller renders it unchanged.
    let mut builder = runmat_runtime::build_runtime_error(primary.message.clone())
        .with_identifier(primary.code.clone())
        .with_call_stack(primary.callstack.clone())
        .with_call_frames_elided(primary.callstack_elided);
    if let Some(span) = primary.span {
        builder = builder.with_span(miette::SourceSpan::new(
            span.start.into(),
            span.end.saturating_sub(span.start).max(1),
        ));
    }
    runmat_runtime::runtime_error::complete_with_cleanup(Err(builder.build()), Err(cleanup))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_projected_failure_remains_primary_with_retirement_cause() {
        let mut outcome = ExecutionOutcome::default();
        outcome
            .diagnostics
            .push(runmat_core::abi::RuntimeDiagnostic {
                code: "RunMat:NativeFFI:HostCrashed".into(),
                severity: DiagnosticSeverity::Error,
                message: "original native crash".into(),
                span: Some(runmat_parser::Span { start: 2, end: 7 }),
                callstack: vec!["main @ main.m:1:1".into()],
                callstack_elided: 2,
            });
        let before = outcome.diagnostics.clone();
        let cleanup = runmat_runtime::build_runtime_error("unverified original operation")
            .with_identifier("RunMat:NativeFFI:ShutdownInterrupted")
            .build();
        let error = complete(&outcome, Err(cleanup)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:NativeFFI:HostCrashed"));
        assert_eq!(error.message(), before[0].message);
        assert_eq!(error.context.call_stack, before[0].callstack);
        assert_eq!(error.context.call_frames_elided, 2);
        assert_eq!(error.span, Some(miette::SourceSpan::new(2.into(), 5)));
        assert_eq!(
            outcome.diagnostics, before,
            "presentation retains the original structured payload"
        );
        let cause = error
            .source
            .as_deref()
            .unwrap()
            .source()
            .and_then(|cause| cause.downcast_ref::<RuntimeError>())
            .unwrap();
        assert_eq!(
            cause.identifier(),
            Some("RunMat:NativeFFI:ShutdownInterrupted")
        );
        complete(&outcome, Ok(())).unwrap();

        let cleanup = runmat_runtime::build_runtime_error("cleanup only")
            .with_identifier("RunMat:NativeFFI:RetirementUnverified")
            .build();
        assert_eq!(
            complete(&ExecutionOutcome::default(), Err(cleanup))
                .unwrap_err()
                .identifier(),
            Some("RunMat:NativeFFI:RetirementUnverified")
        );
    }
}
