use runmat_execution::state::{AttemptState, TaskState};

use crate::driver::{Driver, DriverEvent, DriverEventKind, DriverSnapshot};
use crate::pool::WorkerLifecycle;
use crate::scheduler::{QueueEntry, ResourceAllocation};
use crate::task::{retry_decision, RetryCause, RetryDecision};
use crate::{RunnerError, RunnerResult};

use super::next_driver_fence;

pub fn reconcile_snapshot(snapshot: DriverSnapshot) -> RunnerResult<Driver> {
    let replacement_fence = next_driver_fence(snapshot.driver_fence)?;
    reconcile_snapshot_at_fence(snapshot, replacement_fence)
}

/// Restore under a replacement authority whose fence may skip prior generations.
/// Historical attempts and commits retain their original fencing identities.
pub fn reconcile_snapshot_at_fence(
    mut snapshot: DriverSnapshot,
    replacement_fence: u64,
) -> RunnerResult<Driver> {
    snapshot.validate()?;
    if replacement_fence == 0 || replacement_fence <= snapshot.driver_fence {
        return Err(RunnerError::Invalid(
            "replacement driver fence must be strictly newer than the saved fence".into(),
        ));
    }
    snapshot.driver_fence = replacement_fence;
    let closed_at_driver_millis = snapshot.now_millis;
    let lost = snapshot
        .attempts
        .values_mut()
        .filter(|attempt| {
            matches!(
                attempt.state,
                AttemptState::Assigned | AttemptState::Starting | AttemptState::Running
            )
        })
        .map(|attempt| {
            attempt.state = AttemptState::Lost;
            attempt.terminal = Some(crate::task::AttemptTerminal {
                disposition: crate::task::AttemptDisposition::Lost,
                closed_at_driver_millis,
                resources: None,
            });
            (attempt.request.id, attempt.request.task_id)
        })
        .collect::<Vec<_>>();

    for pool in snapshot.pools.values_mut() {
        pool.active_attempts = 0;
        pool.allocated = ResourceAllocation::default();
        for worker in pool.workers.values_mut() {
            worker.active_attempts = 0;
            worker.allocated = ResourceAllocation::default();
            if worker.lifecycle == WorkerLifecycle::Ready {
                worker.lifecycle = WorkerLifecycle::Lost;
            }
        }
    }

    for (attempt_id, task_id) in lost {
        append_event(
            &mut snapshot,
            DriverEventKind::AttemptLost {
                task_id,
                attempt_id,
            },
        );
        let task = snapshot
            .tasks
            .get_mut(&task_id)
            .expect("attempt task exists in a valid snapshot");
        if matches!(
            task.state,
            TaskState::Succeeded
                | TaskState::Failed
                | TaskState::Cancelled
                | TaskState::Indeterminate
        ) {
            continue;
        }
        task.active_attempt = None;
        let state = match retry_decision(
            task.submission.request.retry,
            task.attempt_count,
            RetryCause::Lost,
        ) {
            RetryDecision::Retry => TaskState::Ready,
            RetryDecision::Fail => TaskState::Failed,
            RetryDecision::Indeterminate => TaskState::Indeterminate,
        };
        task.state = state;
        if state == TaskState::Ready {
            task.enqueued_sequence = snapshot.next_event_sequence;
            snapshot.ready.insert(QueueEntry {
                priority: task.submission.priority,
                enqueued_sequence: task.enqueued_sequence,
                task_id,
            });
        } else {
            snapshot
                .deadlines
                .remove(task_id, task.submission.request.deadline_unix_millis);
        }
        append_event(
            &mut snapshot,
            DriverEventKind::TaskStateChanged { task_id, state },
        );
    }
    Driver::from_snapshot(snapshot)
}

fn append_event(snapshot: &mut DriverSnapshot, kind: DriverEventKind) {
    let sequence = snapshot.next_event_sequence;
    snapshot.next_event_sequence = sequence.saturating_add(1);
    snapshot.events.push(DriverEvent {
        sequence,
        driver_fence: snapshot.driver_fence,
        kind,
    });
}
