use runmat_test::identity::TestId;
use runmat_test::lifecycle::ExecutionPhase;
use runmat_test::result::{
    AttemptResult, Diagnostic, DiagnosticSeverity, ResultState, TerminalDisposition,
};

pub(super) fn terminal_attempt(
    test_id: TestId,
    attempt: u32,
    disposition: TerminalDisposition,
    identifier: &str,
    message: impl Into<String>,
) -> AttemptResult {
    let failed = matches!(
        disposition,
        TerminalDisposition::Failed | TerminalDisposition::TimedOut | TerminalDisposition::Crashed
    );
    AttemptResult {
        test_id,
        attempt,
        state: ResultState {
            failed,
            incomplete: disposition != TerminalDisposition::Passed,
            disposition,
        },
        diagnostics: vec![Diagnostic {
            identifier: identifier.into(),
            message: message.into(),
            severity: DiagnosticSeverity::Error,
            phase: ExecutionPhase::TestBody,
            source: None,
            details: Vec::new(),
        }],
        artifacts: Vec::new(),
        output: String::new(),
        abort_run: false,
    }
}
