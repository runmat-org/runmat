mod common;

use common::{execute, lifecycle_case, scope, step, FakeExecutor};
use runmat_test::context::TestCommand;
use runmat_test::descriptor::FixtureScope;
use runmat_test::executor::ExecutionResponse;
use runmat_test::lifecycle::{ExecutionPhase, QualificationKind};
use runmat_test::result::{Diagnostic, DiagnosticSeverity, ResultState, TerminalDisposition};

#[test]
fn qualification_truth_table_is_locked() {
    let cases = [
        (
            QualificationKind::VerificationFailed,
            ResultState {
                failed: true,
                incomplete: false,
                disposition: TerminalDisposition::Failed,
            },
            false,
        ),
        (
            QualificationKind::AssumptionFailed,
            ResultState {
                failed: false,
                incomplete: true,
                disposition: TerminalDisposition::Filtered,
            },
            false,
        ),
        (
            QualificationKind::AssertionFailed,
            ResultState {
                failed: true,
                incomplete: true,
                disposition: TerminalDisposition::Failed,
            },
            false,
        ),
        (
            QualificationKind::FatalAssertionFailed,
            ResultState {
                failed: true,
                incomplete: true,
                disposition: TerminalDisposition::Failed,
            },
            true,
        ),
    ];
    for (qualification, expected, abort_run) in cases {
        let test_scope = scope(FixtureScope::Test, "test");
        let case = lifecycle_case(Vec::new(), "body", vec![step(test_scope, "teardown")]);
        let mut executor = FakeExecutor::default().responding(
            "body",
            Ok(ExecutionResponse {
                commands: vec![TestCommand::Qualify {
                    qualification,
                    diagnostic: Diagnostic {
                        identifier: "qualification".into(),
                        message: "evidence".into(),
                        severity: DiagnosticSeverity::Error,
                        phase: ExecutionPhase::TestBody,
                        source: None,
                        details: Vec::new(),
                    },
                }],
                output: String::new(),
            }),
        );
        let (outcome, _) = execute(&case, &mut executor);
        assert_eq!(outcome.attempt.state, expected, "{qualification:?}");
        assert_eq!(outcome.attempt.abort_run, abort_run, "{qualification:?}");
        assert_eq!(executor.calls, ["body", "teardown"], "{qualification:?}");
    }
}
