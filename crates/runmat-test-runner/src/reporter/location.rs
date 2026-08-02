use runmat_test::result::{AttemptResult, Diagnostic, TestResult};

pub(super) fn attempt_diagnostic(result: &AttemptResult) -> Option<&Diagnostic> {
    result.diagnostics.first()
}

pub(super) fn primary_diagnostic(result: &TestResult) -> Option<&Diagnostic> {
    result.attempts.last().and_then(attempt_diagnostic)
}

pub(super) fn source_label(diagnostic: &Diagnostic) -> Option<String> {
    diagnostic.source.as_ref().map(|source| {
        format!(
            "{}:{}:{}",
            source.relative_path, source.span.start_line, source.span.start_column
        )
    })
}
