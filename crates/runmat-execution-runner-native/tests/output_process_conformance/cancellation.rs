//! Original cancellation evidence from a real request-gated execution worker.
use runmat_execution::{console::ConsoleCompletion, state::TaskState, CancellationReason};
use runmat_execution_runner::{
    AttemptExecutionTermination, TaskCancellationCause, TaskExecutionEvidence,
};

pub(super) fn verify_evidence(evidence: &TaskExecutionEvidence) {
    evidence.validate().unwrap();
    assert_eq!(evidence.task.state, TaskState::Cancelled);
    assert_eq!(
        evidence.task.cancellation.as_ref().unwrap().cause,
        TaskCancellationCause::Task {
            task_id: evidence.task.id(),
            reason: CancellationReason::Quota,
        }
    );
    assert_eq!(evidence.attempts.len(), 1);
    let terminal = evidence.attempts[0].terminal.as_ref().unwrap();
    assert!(matches!(
        terminal.execution.as_ref().unwrap().termination,
        AttemptExecutionTermination::ProcessExited { success: false, .. }
    ));
    let prefix = terminal
        .console
        .as_ref()
        .expect("cancelled original console prefix");
    assert_eq!(prefix.completion(), ConsoleCompletion::Interrupted);
    assert_eq!(prefix.records().len(), 1);
    assert_eq!(
        prefix.records()[0].text.as_ref(),
        "original before termination"
    );
}
