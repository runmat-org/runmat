use runmat_execution::identity::WorkerId;
use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution::{PoolId, TaskId};

use crate::cancellation::CancellationEscalation;
use crate::pool::{PoolRecord, WorkerRecord};
use crate::scheduler::QueueEntry;
use crate::task::TaskRecord;
use crate::{RunnerError, RunnerResult};

use super::{Driver, DriverAction, DriverEvent, DriverEventKind};

impl Driver {
    pub(super) fn activate_dependents(&mut self, task_id: TaskId) -> RunnerResult<()> {
        let dependents = self.snapshot.graph.dependents(task_id).collect::<Vec<_>>();
        for dependent in dependents {
            let ready = self
                .snapshot
                .graph
                .dependencies(dependent)
                .into_iter()
                .flatten()
                .all(|dependency| {
                    self.snapshot
                        .tasks
                        .get(dependency)
                        .is_some_and(|task| task.state == TaskState::Succeeded)
                });
            if ready
                && self
                    .snapshot
                    .tasks
                    .get(&dependent)
                    .is_some_and(|task| task.state == TaskState::Deferred)
            {
                let enqueue_sequence = self.snapshot.next_event_sequence;
                let task = self.task_mut(dependent)?;
                task.state = TaskState::Ready;
                task.enqueued_sequence = enqueue_sequence;
                let task = task.clone();
                self.enqueue(&task);
                self.emit(DriverEventKind::TaskStateChanged {
                    task_id: dependent,
                    state: TaskState::Ready,
                });
            }
        }
        Ok(())
    }

    pub(super) fn cancel_task(
        &mut self,
        task_id: TaskId,
        now_millis: u64,
        cause: crate::TaskCancellationCause,
        actions: &mut Vec<DriverAction>,
    ) -> RunnerResult<()> {
        let retained = self
            .task_record(task_id)?
            .active_attempt
            .and_then(|id| self.snapshot.attempts.get(&id))
            .is_some_and(|attempt| {
                attempt
                    .unverified_retirement
                    .as_ref()
                    .is_some_and(|evidence| evidence.retired_at_driver_millis.is_none())
            });
        if self.task_is_terminal(task_id)? && !retained {
            return Ok(());
        }
        let task = self.task_mut(task_id)?;
        let original = *task.cancellation.get_or_insert(crate::TaskCancellation {
            cause,
            requested_at_driver_millis: now_millis,
        });
        let active = task.active_attempt;
        self.snapshot.ready.remove_task(task_id);
        // Accepted work retains its attempt and reservation until its owning
        // backend proves retirement, independent of callable representation.
        if active.is_none() {
            self.transition_task(task_id, TaskState::Cancelled)?;
        }
        if let Some(attempt_id) = active {
            let attempt = self
                .snapshot
                .attempts
                .get_mut(&attempt_id)
                .ok_or(RunnerError::UnknownAttempt(attempt_id))?;
            let first_request = attempt.cancellation_requested_at.is_none();
            attempt
                .cancellation_requested_at
                .get_or_insert(original.requested_at_driver_millis);
            if first_request {
                attempt.cancellation_escalation = Some(CancellationEscalation::Request);
            }
            let request = attempt.request.clone();
            if first_request {
                actions.push(DriverAction::Cancel(request));
            }
        }
        Ok(())
    }

    pub(super) fn expire_attempt_wall_times(
        &mut self,
        now_millis: u64,
        actions: &mut Vec<DriverAction>,
    ) -> RunnerResult<()> {
        let expired = self
            .snapshot
            .attempts
            .values()
            .filter(|attempt| !attempt_state_is_terminal(attempt.state))
            .filter(|attempt| attempt.cancellation_requested_at.is_none())
            .filter(|attempt| {
                attempt
                    .assigned_at_millis
                    .saturating_add(attempt.request.task.resources.max_wall_millis)
                    <= now_millis
            })
            .map(|attempt| (attempt.request.task_id, attempt.request.id))
            .collect::<Vec<_>>();
        for (task_id, attempt_id) in expired {
            if !self.task_is_terminal(task_id)? {
                self.emit(DriverEventKind::DeadlineExpired { task_id });
                self.cancel_task(
                    task_id,
                    now_millis,
                    crate::TaskCancellationCause::WallTime { attempt_id },
                    actions,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn escalate_cancellations(
        &mut self,
        now_millis: u64,
        actions: &mut Vec<DriverAction>,
    ) -> RunnerResult<()> {
        let candidates = self
            .snapshot
            .attempts
            .values()
            .filter(|attempt| !attempt_state_is_terminal(attempt.state))
            .filter_map(|attempt| {
                attempt
                    .cancellation_requested_at
                    .map(|requested| (attempt.request.id, requested))
            })
            .collect::<Vec<_>>();
        for (attempt_id, requested_at) in candidates {
            let level = self
                .snapshot
                .config
                .cancellation_escalation
                .level(requested_at, now_millis);
            let current = self.snapshot.attempts[&attempt_id].cancellation_escalation;
            match level {
                CancellationEscalation::Request => {}
                CancellationEscalation::Terminate
                    if current == Some(CancellationEscalation::Request) =>
                {
                    let attempt = self
                        .snapshot
                        .attempts
                        .get_mut(&attempt_id)
                        .expect("candidate attempt exists");
                    attempt.cancellation_escalation = Some(CancellationEscalation::Terminate);
                    actions.push(DriverAction::Terminate(attempt.request.clone()));
                }
                CancellationEscalation::Fence if current != Some(CancellationEscalation::Fence) => {
                    let attempt = self.snapshot.attempts[&attempt_id].clone();
                    if current == Some(CancellationEscalation::Request) {
                        actions.push(DriverAction::Terminate(attempt.request.clone()));
                    }
                    self.snapshot
                        .attempts
                        .get_mut(&attempt_id)
                        .expect("candidate attempt exists")
                        .cancellation_escalation = Some(CancellationEscalation::Fence);
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub(super) fn transition_task(
        &mut self,
        task_id: TaskId,
        state: TaskState,
    ) -> RunnerResult<()> {
        let task = self.task_mut(task_id)?;
        task.state = state;
        task.active_attempt = None;
        task.indeterminate_dependency = None;
        if task_state_is_terminal(state) {
            self.remove_deadline(task_id);
        }
        self.emit(DriverEventKind::TaskStateChanged { task_id, state });
        if matches!(
            state,
            TaskState::Failed | TaskState::Cancelled | TaskState::Indeterminate
        ) {
            self.propagate_dependency_terminal(task_id, state)?;
        }
        Ok(())
    }

    fn propagate_dependency_terminal(
        &mut self,
        task_id: TaskId,
        state: TaskState,
    ) -> RunnerResult<()> {
        let dependents = self.snapshot.graph.dependents(task_id).collect::<Vec<_>>();
        for dependent in dependents {
            if self.task_is_terminal(dependent)? {
                continue;
            }
            self.snapshot.ready.remove_task(dependent);
            let cancellation = if state == TaskState::Cancelled {
                Some(crate::TaskCancellation {
                    cause: crate::TaskCancellationCause::Dependency { task_id },
                    requested_at_driver_millis: self.snapshot.now_millis,
                })
            } else {
                None
            };
            let task = self.task_mut(dependent)?;
            task.cancellation = cancellation;
            task.state = state;
            task.active_attempt = None;
            task.indeterminate_dependency = (state == TaskState::Indeterminate).then_some(task_id);
            self.remove_deadline(dependent);
            self.emit(DriverEventKind::TaskStateChanged {
                task_id: dependent,
                state,
            });
            self.propagate_dependency_terminal(dependent, state)?;
        }
        Ok(())
    }

    /// Reopen only descendants whose indeterminate state was caused by this
    /// exact dependency. Other unresolved parents retain their own provenance.
    pub(super) fn reopen_certified_dependency_descendants(
        &mut self,
        parent: TaskId,
        actions: &mut Vec<DriverAction>,
    ) -> RunnerResult<()> {
        let dependents = self.snapshot.graph.dependents(parent).collect::<Vec<_>>();
        for dependent in dependents {
            let record = self.task_record(dependent)?;
            if record.indeterminate_dependency != Some(parent) {
                continue;
            }
            if record.state != TaskState::Indeterminate
                || record.active_attempt.is_some()
                || record.attempt_count != 0
                || record.cancellation.is_some()
            {
                return Err(RunnerError::Invalid(
                    "dependency uncertainty has independent attempt or terminal authority".into(),
                ));
            }
            let dependencies = record
                .submission
                .dependencies
                .iter()
                .copied()
                .collect::<Vec<_>>();
            let now = self.snapshot.now_millis;
            if let Some(other) = dependencies
                .iter()
                .find(|id| self.snapshot.tasks[id].state == TaskState::Indeterminate)
            {
                self.task_mut(dependent)?.indeterminate_dependency = Some(*other);
                continue;
            }
            let next = if dependencies
                .iter()
                .any(|id| self.snapshot.tasks[id].state == TaskState::Failed)
            {
                TaskState::Failed
            } else if dependencies
                .iter()
                .any(|id| self.snapshot.tasks[id].state == TaskState::Cancelled)
            {
                TaskState::Cancelled
            } else if dependencies
                .iter()
                .all(|id| self.snapshot.tasks[id].state == TaskState::Succeeded)
            {
                TaskState::Ready
            } else {
                TaskState::Deferred
            };
            self.task_mut(dependent)?.indeterminate_dependency = None;
            match next {
                TaskState::Failed => self.transition_task(dependent, TaskState::Failed)?,
                TaskState::Cancelled => {
                    let source = dependencies
                        .iter()
                        .find(|id| self.snapshot.tasks[id].state == TaskState::Cancelled)
                        .expect("cancelled dependency was checked");
                    self.task_mut(dependent)?.cancellation = Some(crate::TaskCancellation {
                        cause: crate::TaskCancellationCause::Dependency { task_id: *source },
                        requested_at_driver_millis: now,
                    });
                    self.transition_task(dependent, TaskState::Cancelled)?;
                }
                TaskState::Ready | TaskState::Deferred => {
                    self.task_mut(dependent)?.state = next;
                    let deadline = self
                        .task_record(dependent)?
                        .submission
                        .request
                        .deadline_unix_millis;
                    if deadline.is_some_and(|at| at <= now) {
                        self.emit(DriverEventKind::DeadlineExpired { task_id: dependent });
                        self.cancel_task(
                            dependent,
                            now,
                            crate::TaskCancellationCause::Deadline,
                            actions,
                        )?;
                    } else {
                        self.snapshot.deadlines.insert(dependent, deadline);
                        if next == TaskState::Ready {
                            let record = self.task_record(dependent)?.clone();
                            self.enqueue(&record);
                        }
                        self.emit(DriverEventKind::TaskStateChanged {
                            task_id: dependent,
                            state: next,
                        });
                    }
                }
                _ => unreachable!("dependency recovery only restores a queued or terminal state"),
            }
            self.reopen_certified_dependency_descendants(dependent, actions)?;
        }
        Ok(())
    }

    pub(super) fn enqueue(&mut self, task: &TaskRecord) {
        self.snapshot.ready.insert(QueueEntry {
            priority: task.submission.priority,
            enqueued_sequence: task.enqueued_sequence,
            task_id: task.id(),
        });
    }

    pub(super) fn remove_deadline(&mut self, task_id: TaskId) {
        let deadline = self
            .snapshot
            .tasks
            .get(&task_id)
            .and_then(|task| task.submission.request.deadline_unix_millis);
        self.snapshot.deadlines.remove(task_id, deadline);
    }

    pub(super) fn emit(&mut self, kind: DriverEventKind) {
        let sequence = self.snapshot.next_event_sequence;
        self.snapshot.next_event_sequence = sequence.saturating_add(1);
        self.snapshot.events.push(DriverEvent {
            sequence,
            driver_fence: self.snapshot.driver_fence,
            kind,
        });
    }

    pub(super) fn pool_mut(&mut self, pool_id: PoolId) -> RunnerResult<&mut PoolRecord> {
        self.snapshot
            .pools
            .get_mut(&pool_id)
            .ok_or(RunnerError::UnknownPool(pool_id))
    }

    pub(super) fn worker_mut(&mut self, worker_id: WorkerId) -> RunnerResult<&mut WorkerRecord> {
        self.snapshot
            .pools
            .values_mut()
            .find_map(|pool| pool.workers.get_mut(&worker_id))
            .ok_or(RunnerError::UnknownWorker(worker_id))
    }

    pub(super) fn task_mut(&mut self, task_id: TaskId) -> RunnerResult<&mut TaskRecord> {
        self.snapshot
            .tasks
            .get_mut(&task_id)
            .ok_or(RunnerError::UnknownTask(task_id))
    }

    pub(super) fn task_is_terminal(&self, task_id: TaskId) -> RunnerResult<bool> {
        self.snapshot
            .tasks
            .get(&task_id)
            .map(|task| task_state_is_terminal(task.state))
            .ok_or(RunnerError::UnknownTask(task_id))
    }
}

pub(super) fn attempt_state_is_terminal(state: AttemptState) -> bool {
    matches!(
        state,
        AttemptState::Completed
            | AttemptState::Lost
            | AttemptState::Rejected
            | AttemptState::Cancelled
    )
}

fn task_state_is_terminal(state: TaskState) -> bool {
    matches!(
        state,
        TaskState::Succeeded | TaskState::Failed | TaskState::Cancelled | TaskState::Indeterminate
    )
}
