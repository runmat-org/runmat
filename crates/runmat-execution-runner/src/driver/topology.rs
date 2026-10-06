use runmat_execution::identity::WorkerId;
use runmat_execution::state::TaskState;

use crate::pool::{WorkerLifecycle, WorkerRecord, WorkerSpec};
use crate::port::BackendReport;
use crate::task::{AttemptReport, TaskRecord, TaskSubmission};
use crate::{RunnerError, RunnerResult};

use super::state::attempt_state_is_terminal;
use super::{Driver, DriverAction, DriverEventKind};

impl Driver {
    /// Borrow the current scheduler-owned pool while the caller holds driver authority.
    /// Inspection does not copy task histories or authorize a provider launch.
    pub fn pool_record(
        &self,
        pool_id: runmat_execution::PoolId,
    ) -> RunnerResult<&crate::pool::PoolRecord> {
        self.snapshot
            .pools
            .get(&pool_id)
            .ok_or(RunnerError::UnknownPool(pool_id))
    }

    pub(super) fn register_worker(&mut self, spec: WorkerSpec) -> RunnerResult<()> {
        let pool = self.pool_mut(spec.pool_id)?;
        if let Some(existing) = pool.workers.get(&spec.id) {
            if existing.lifecycle == WorkerLifecycle::Lost
                && existing.active_attempts == 0
                && existing.spec == spec
            {
                let worker_id = spec.id;
                let pool_id = spec.pool_id;
                pool.workers.insert(worker_id, WorkerRecord::new(spec));
                self.emit(DriverEventKind::WorkerRegistered { worker_id, pool_id });
                return Ok(());
            }
        }
        let live_worker_count = pool
            .workers
            .values()
            .filter(|worker| {
                !matches!(
                    worker.lifecycle,
                    WorkerLifecycle::Lost | WorkerLifecycle::Stopped
                )
            })
            .count();
        if live_worker_count >= pool.spec.max_workers as usize {
            return Err(RunnerError::Invalid(format!(
                "pool {} is at its worker limit",
                spec.pool_id
            )));
        }
        if pool.workers.contains_key(&spec.id) {
            return Err(RunnerError::Invalid(format!(
                "worker {} already exists",
                spec.id
            )));
        }
        let worker_id = spec.id;
        let pool_id = spec.pool_id;
        pool.workers.insert(worker_id, WorkerRecord::new(spec));
        self.emit(DriverEventKind::WorkerRegistered { worker_id, pool_id });
        Ok(())
    }

    pub(super) fn drain_worker(&mut self, worker_id: WorkerId) -> RunnerResult<()> {
        self.worker_mut(worker_id)?.lifecycle = WorkerLifecycle::Draining;
        self.emit(DriverEventKind::WorkerDraining { worker_id });
        Ok(())
    }

    pub(super) fn worker_lost(
        &mut self,
        worker_id: WorkerId,
        actions: &mut Vec<DriverAction>,
    ) -> RunnerResult<()> {
        self.worker_mut(worker_id)?.lifecycle = WorkerLifecycle::Lost;
        self.emit(DriverEventKind::WorkerLost { worker_id });
        self.close_lost_worker_attempts(worker_id, actions)
    }

    pub(super) fn worker_lost_with_report(
        &mut self,
        worker_id: WorkerId,
        report: BackendReport,
        actions: &mut Vec<DriverAction>,
    ) -> RunnerResult<()> {
        if report.worker_id != worker_id || !matches!(report.report, AttemptReport::Lost { .. }) {
            return Err(RunnerError::Invalid(
                "worker loss requires its original provider loss report".into(),
            ));
        }
        // Validate the triggering attempt before changing worker authority. A
        // stale, foreign or malformed report cannot fence unrelated live work.
        if let Some(reason) = self.backend_report_rejection_reason(&report) {
            self.discard_backend_report(report, reason, actions);
            return Ok(());
        }
        self.worker_mut(worker_id)?.lifecycle = WorkerLifecycle::Lost;
        self.emit(DriverEventKind::WorkerLost { worker_id });
        self.apply_backend_report(report, actions)?;
        self.close_lost_worker_attempts(worker_id, actions)
    }

    fn close_lost_worker_attempts(
        &mut self,
        worker_id: WorkerId,
        actions: &mut Vec<DriverAction>,
    ) -> RunnerResult<()> {
        let reports = self
            .snapshot
            .attempts
            .values()
            .filter(|attempt| {
                attempt.request.worker_id == worker_id && !attempt_state_is_terminal(attempt.state)
            })
            .map(|attempt| {
                BackendReport::for_request(
                    &attempt.request,
                    AttemptReport::Lost {
                        message: "worker was lost".into(),
                    },
                )
            })
            .collect::<Vec<_>>();
        for report in reports {
            self.apply_backend_report(report, actions)?;
        }
        Ok(())
    }

    pub(super) fn submit(&mut self, submission: TaskSubmission) -> RunnerResult<()> {
        submission.retry_resource_envelope()?;
        submission.request.resources.validate().map_err(|error| {
            RunnerError::Invalid(format!("task resource request is invalid: {error}"))
        })?;
        if !self
            .snapshot
            .cancellation
            .contains(submission.request.scope_id)
        {
            return Err(RunnerError::Invalid(
                "task execution scope is not registered".into(),
            ));
        }
        if self
            .snapshot
            .cancellation
            .state(submission.request.scope_id)
            .is_some()
        {
            return Err(RunnerError::Invalid(
                "cannot submit into a cancelled execution scope".into(),
            ));
        }
        let pool = self
            .snapshot
            .pools
            .get(&submission.request.pool_id)
            .ok_or(RunnerError::UnknownPool(submission.request.pool_id))?;
        if !crate::scheduler::fits(
            &pool.spec.resource_limit,
            &Default::default(),
            &submission.request.resources,
        ) {
            return Err(RunnerError::Invalid(
                "task resource request cannot be satisfied by the target pool".into(),
            ));
        }
        if self.snapshot.tasks.contains_key(&submission.request.id) {
            return Err(RunnerError::Invalid(format!(
                "task {} already exists",
                submission.request.id
            )));
        }
        for dependency in &submission.dependencies {
            if !self.snapshot.tasks.contains_key(dependency) {
                return Err(RunnerError::UnknownTask(*dependency));
            }
        }
        let dependency_state = submission
            .dependencies
            .iter()
            .filter_map(|dependency| self.snapshot.tasks.get(dependency).map(|task| task.state))
            .fold(None, merge_dependency_terminal_state);
        let dependencies_succeeded = !submission.dependencies.is_empty()
            && submission
                .dependencies
                .iter()
                .all(|dependency| self.snapshot.tasks[dependency].state == TaskState::Succeeded);
        let task_id = submission.request.id;
        self.snapshot
            .graph
            .insert(task_id, submission.dependencies.clone())?;
        let mut record = TaskRecord::new(submission, self.snapshot.next_event_sequence);
        if let Some(state) = dependency_state {
            record.state = state;
            record.active_attempt = None;
            if state == TaskState::Cancelled {
                let dependency = record
                    .submission
                    .dependencies
                    .iter()
                    .find(|id| self.snapshot.tasks[id].state == TaskState::Cancelled)
                    .expect("merged cancellation has an original cancelled dependency");
                record.cancellation = Some(crate::TaskCancellation {
                    cause: crate::TaskCancellationCause::Dependency {
                        task_id: *dependency,
                    },
                    requested_at_driver_millis: self.snapshot.now_millis,
                });
            }
        } else if dependencies_succeeded {
            record.state = TaskState::Ready;
        }
        let state = record.state;
        if state == TaskState::Ready {
            self.enqueue(&record);
        }
        if !matches!(
            state,
            TaskState::Succeeded
                | TaskState::Failed
                | TaskState::Cancelled
                | TaskState::Indeterminate
        ) {
            self.snapshot
                .deadlines
                .insert(task_id, record.submission.request.deadline_unix_millis);
        }
        self.snapshot.tasks.insert(task_id, record);
        self.emit(DriverEventKind::TaskSubmitted { task_id, state });
        Ok(())
    }
}

fn merge_dependency_terminal_state(
    current: Option<TaskState>,
    candidate: TaskState,
) -> Option<TaskState> {
    let candidate_rank = dependency_terminal_rank(candidate)?;
    match current {
        Some(current)
            if dependency_terminal_rank(current).unwrap_or_default() >= candidate_rank =>
        {
            Some(current)
        }
        _ => Some(candidate),
    }
}

const fn dependency_terminal_rank(state: TaskState) -> Option<u8> {
    match state {
        TaskState::Cancelled => Some(1),
        TaskState::Failed => Some(2),
        TaskState::Indeterminate => Some(3),
        TaskState::Deferred
        | TaskState::Ready
        | TaskState::Assigned
        | TaskState::Running
        | TaskState::Committing
        | TaskState::Succeeded => None,
    }
}
