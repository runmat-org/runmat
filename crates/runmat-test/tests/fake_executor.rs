mod common;

use common::{execute, lifecycle_case, FakeExecutor};
use runmat_test::executor::ExecutionFault;
use runmat_test::result::TerminalDisposition;

#[test]
fn executor_faults_have_canonical_terminal_classification() {
    let cases = [
        (
            ExecutionFault::Uncaught("error".into()),
            TerminalDisposition::Failed,
            true,
        ),
        (
            ExecutionFault::TimedOut("timeout".into()),
            TerminalDisposition::TimedOut,
            true,
        ),
        (
            ExecutionFault::Cancelled("cancelled".into()),
            TerminalDisposition::Cancelled,
            false,
        ),
        (
            ExecutionFault::WorkerCrashed("crash".into()),
            TerminalDisposition::Crashed,
            true,
        ),
    ];
    for (fault, disposition, failed) in cases {
        let case = lifecycle_case(Vec::new(), "body", Vec::new());
        let mut executor = FakeExecutor::default().faulting("body", fault);
        let (outcome, _) = execute(&case, &mut executor);
        assert_eq!(outcome.attempt.state.disposition, disposition);
        assert_eq!(outcome.attempt.state.failed, failed);
        assert!(outcome.attempt.state.incomplete);
    }
}
