use runmat_execution::state::{AttemptState, TaskState};

use crate::driver::{Driver, DriverEvent, DriverEventKind, DriverSnapshot};
use crate::pool::WorkerLifecycle;
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
    let observed_at_driver_millis = snapshot.now_millis;
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
            attempt.unverified_retirement.get_or_insert_with(|| {
                Box::new(crate::task::AttemptRetirementUnverified {
                    observed_at_driver_millis,
                    console: None,
                    console_scopes: Vec::new(),
                    delegated_console_root: None,
                    resources: None,
                    execution: None,
                    diagnostic: "driver authority replaced before a retirement certificate".into(),
                    failure: None,
                    failure_kind: None,
                    retired_at_driver_millis: None,
                })
            });
            (attempt.request.id, attempt.request.task_id)
        })
        .collect::<Vec<_>>();

    for pool in snapshot.pools.values_mut() {
        for worker in pool.workers.values_mut() {
            if worker.active_attempts > 0 || worker.lifecycle == WorkerLifecycle::Ready {
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
        // Replacing authority fences publication, not physical work. Keep the
        // exact old attempt and reservation for a later original-fence proof.
        task.active_attempt = Some(attempt_id);
        let state = TaskState::Indeterminate;
        task.state = state;
        snapshot.ready.remove_task(task_id);
        snapshot
            .deadlines
            .remove(task_id, task.submission.request.deadline_unix_millis);
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
