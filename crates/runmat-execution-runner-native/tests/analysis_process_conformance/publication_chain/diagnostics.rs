//! Bounded original evidence for failed process witnesses; never accounting synthesis.
use runmat_execution_runner::TaskExecutionEvidence;
use runmat_execution_runner_native::RemotePoolDriver;

pub(super) fn attempt_summary(evidence: &TaskExecutionEvidence) -> String {
    let attempts = evidence
        .attempts
        .iter()
        .take(8)
        .map(|attempt| {
            let terminal = attempt.terminal.as_ref();
            (
                attempt.request.ordinal,
                attempt.request.id,
                attempt.request.worker_id,
                attempt.request.driver_fence,
                attempt.state,
                terminal.map(|value| value.disposition),
                terminal.map(|value| value.resources.is_some()),
                terminal
                    .and_then(|value| value.execution.as_ref())
                    .map(|value| value.termination),
                terminal
                    .and_then(|value| value.diagnostic.as_ref())
                    .map(|value| value.chars().take(512).collect::<String>()),
            )
        })
        .collect::<Vec<_>>();
    format!("task={}, attempt_count={}, first8(ordinal,id,worker,fence,state,disposition,accounting,termination,diagnostic)={attempts:?}",
        evidence.task.submission.request.id, evidence.attempts.len())
}

pub(super) fn remote_timeout_summary(pool: &RemotePoolDriver) -> String {
    let snapshot = pool.snapshot();
    let pending = snapshot
        .tasks
        .iter()
        .filter(|(_, task)| !matches!(task.state, runmat_execution::state::TaskState::Succeeded))
        .take(8)
        .map(|(id, task)| (id, task.state, &task.submission.request.resources))
        .collect::<Vec<_>>();
    format!(
        "tasks={}, attempts={}, events={}, pending(first 8)={pending:?}, pools={:?}, last_events={:?}",
        snapshot.tasks.len(),
        snapshot.attempts.len(),
        snapshot.events.len(),
        snapshot.pools,
        snapshot.events.iter().rev().take(8).collect::<Vec<_>>()
    )
}
