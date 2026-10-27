//! One nearest wake-up from the driver's existing admitted records.
use crate::cancellation::CancellationEscalation;

use super::state::attempt_state_is_terminal;
use super::Driver;

impl Driver {
    /// Absolute driver-clock time of the next deadline or control escalation.
    /// Fencing never proves retirement and needs no repeated polling timer.
    pub fn next_timer_deadline_millis(&self) -> Option<u64> {
        let task_deadlines = self
            .snapshot
            .tasks
            .values()
            .filter(|task| task.cancellation.is_none())
            .filter(|task| {
                !matches!(
                    task.state,
                    runmat_execution::state::TaskState::Succeeded
                        | runmat_execution::state::TaskState::Failed
                        | runmat_execution::state::TaskState::Cancelled
                        | runmat_execution::state::TaskState::Indeterminate
                )
            })
            .filter_map(|task| task.submission.request.deadline_unix_millis);
        let policy = self.snapshot.config.cancellation_escalation;
        let attempt_deadlines = self
            .snapshot
            .attempts
            .values()
            .filter(|attempt| !attempt_state_is_terminal(attempt.state))
            .filter_map(|attempt| match attempt.cancellation_requested_at {
                None if attempt.unverified_retirement.is_none() => Some(
                    attempt
                        .assigned_at_millis
                        .saturating_add(attempt.request.task.resources.max_wall_millis),
                ),
                None => None,
                Some(at) => match attempt.cancellation_escalation {
                    Some(CancellationEscalation::Fence) => None,
                    Some(CancellationEscalation::Terminate) => {
                        Some(at.saturating_add(policy.terminate_millis))
                    }
                    None | Some(CancellationEscalation::Request) => {
                        Some(at.saturating_add(policy.graceful_millis.min(policy.terminate_millis)))
                    }
                },
            });
        task_deadlines.chain(attempt_deadlines).min()
    }
}
