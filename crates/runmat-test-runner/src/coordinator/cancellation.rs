use runmat_test::identity::{RunId, TestId};
use runmat_test::lifecycle::ExecutionPhase;
use runmat_test::result::{Diagnostic, DiagnosticSeverity, ResultState, TerminalDisposition};

use crate::host::Clock;
use crate::worker::{CancelRequest, WorkerBackend, WorkerExecution};

use super::recovery::terminal_attempt;

pub(super) struct CancellationRequest {
    pub run_id: RunId,
    pub test_id: TestId,
    pub attempt: u32,
    pub reason: String,
    pub grace_ms: u64,
    pub disposition: TerminalDisposition,
}

pub(super) async fn cancel_or_terminate<B: WorkerBackend, C: Clock>(
    backend: &B,
    session: &B::Session,
    clock: &C,
    cancellation: CancellationRequest,
) -> (WorkerExecution, bool) {
    let grace_deadline_ms = clock.now_ms().saturating_add(cancellation.grace_ms);
    let cancel_request = CancelRequest {
        run_id: cancellation.run_id.clone(),
        reason: cancellation.reason.clone(),
        grace_deadline_ms,
    };
    let cancel = backend.cancel(session, cancel_request).boxed_local();
    let grace = clock.sleep_until(grace_deadline_ms).boxed_local();
    let response = match select(cancel, grace).await {
        Either::Left((response, _)) => response,
        Either::Right(((), _)) => {
            let _ = backend.terminate(session).await;
            return (
                cancelled_execution(&cancellation, "cancellation grace period elapsed"),
                true,
            );
        }
    };
    match response {
        Ok(Some(execution))
            if execution.result.test_id == cancellation.test_id
                && execution.result.attempt == cancellation.attempt =>
        {
            (
                classify_controlled_completion(execution, &cancellation),
                false,
            )
        }
        _ => {
            let _ = backend.terminate(session).await;
            (
                cancelled_execution(&cancellation, "worker did not stop"),
                true,
            )
        }
    }
}

fn classify_controlled_completion(
    mut execution: WorkerExecution,
    cancellation: &CancellationRequest,
) -> WorkerExecution {
    if execution.result.state.disposition == cancellation.disposition {
        return execution;
    }
    execution.result.state = ResultState {
        failed: cancellation.disposition == TerminalDisposition::TimedOut,
        incomplete: true,
        disposition: cancellation.disposition,
    };
    let (identifier, message) = match cancellation.disposition {
        TerminalDisposition::TimedOut => (
            "runmat:test:TimedOut",
            format!("test timed out: {}", cancellation.reason),
        ),
        _ => (
            "runmat:test:Cancelled",
            format!("test cancelled: {}", cancellation.reason),
        ),
    };
    execution.result.diagnostics.push(Diagnostic {
        identifier: identifier.into(),
        message,
        severity: DiagnosticSeverity::Error,
        phase: ExecutionPhase::TestBody,
        source: None,
        details: Vec::new(),
    });
    execution
}

fn cancelled_execution(cancellation: &CancellationRequest, escalation: &str) -> WorkerExecution {
    let (identifier, label) = match cancellation.disposition {
        TerminalDisposition::TimedOut => ("runmat:test:TimedOut", "timed out"),
        _ => ("runmat:test:Cancelled", "cancelled"),
    };
    WorkerExecution {
        result: terminal_attempt(
            cancellation.test_id.clone(),
            cancellation.attempt,
            cancellation.disposition,
            identifier,
            format!("test {label}: {} ({escalation})", cancellation.reason),
        ),
        events: Vec::new(),
        coverage: Vec::new(),
    }
}
use futures::future::{select, Either};
use futures::FutureExt;
