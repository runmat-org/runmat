use runmat_execution::identity::WorkerId;
use runmat_execution::state::TaskState;

use crate::pool::{WorkerLifecycle, WorkerRecord, WorkerSpec};
use crate::port::BackendReport;
use crate::task::{AttemptReport, TaskRecord, TaskSubmission};
use crate::{RunnerError, RunnerResult};

use super::state::attempt_state_is_terminal;
use super::{Driver, DriverAction, DriverEventKind};

impl Driver {
    pub(super) fn register_worker(&mut self, spec: WorkerSpec) -> RunnerResult<()> {
        let pool = self.pool_mut(spec.pool_id)?;
        if pool.workers.len() >= pool.spec.max_workers as usize {
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
        let task_id = submission.request.id;
        self.snapshot
            .graph
            .insert(task_id, submission.dependencies.clone())?;
        let record = TaskRecord::new(submission, self.snapshot.next_event_sequence);
        let state = record.state;
        if state == TaskState::Ready {
            self.enqueue(&record);
        }
        self.snapshot
            .deadlines
            .insert(task_id, record.submission.request.deadline_unix_millis);
        self.snapshot.tasks.insert(task_id, record);
        self.emit(DriverEventKind::TaskSubmitted { task_id, state });
        Ok(())
    }
}
