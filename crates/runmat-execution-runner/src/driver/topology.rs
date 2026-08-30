use runmat_execution::identity::WorkerId;
use runmat_execution::state::PoolState;
use runmat_execution::state::TaskState;
use runmat_execution::PoolId;

use crate::pool::{WorkerLifecycle, WorkerRecord, WorkerSpec};
use crate::port::BackendReport;
use crate::task::{AttemptReport, TaskRecord, TaskSubmission};
use crate::{RunnerError, RunnerResult};

use super::state::attempt_state_is_terminal;
use super::{Driver, DriverAction, DriverCommand, DriverEventKind};
use crate::pool::ResizeRequest;

impl Driver {
    /// Reconciles a pool whose complete worker inventory is already registered.
    ///
    /// Process-local and browser hosts use stable logical workers whose backing
    /// execution contexts are created on demand. Resizing those pools changes
    /// which registered workers may accept new attempts; it does not provision
    /// infrastructure. Remote hosts consume `DriverAction::ResizePool` instead.
    pub fn resize_registered_pool(
        &mut self,
        pool_id: PoolId,
        desired_workers: u32,
    ) -> RunnerResult<Vec<DriverAction>> {
        let resize_actions = self.handle(DriverCommand::ResizePool {
            pool_id,
            request: ResizeRequest { desired_workers },
        })?;
        let requested_resize = resize_actions.iter().any(|action| {
            matches!(
                action,
                DriverAction::ResizePool {
                    pool_id: action_pool,
                    desired_workers: action_workers,
                } if *action_pool == pool_id && *action_workers == desired_workers
            )
        });
        if !requested_resize {
            return Ok(resize_actions);
        }

        let pool = self
            .snapshot
            .pools
            .get(&pool_id)
            .ok_or(RunnerError::UnknownPool(pool_id))?;
        let ready = pool
            .workers
            .values()
            .filter(|worker| worker.accepts_work())
            .map(|worker| worker.spec.id)
            .collect::<Vec<_>>();
        let ready_count = u32::try_from(ready.len())
            .map_err(|_| RunnerError::Invalid("pool worker count exceeds u32".into()))?;
        let mut actions = Vec::new();
        if desired_workers < ready_count {
            let count = usize::try_from(ready_count - desired_workers)
                .map_err(|_| RunnerError::Invalid("pool resize count exceeds usize".into()))?;
            for worker_id in ready.into_iter().rev().take(count) {
                actions.extend(self.handle(DriverCommand::DrainWorker(worker_id))?);
            }
        } else if desired_workers > ready_count {
            let count = usize::try_from(desired_workers - ready_count)
                .map_err(|_| RunnerError::Invalid("pool resize count exceeds usize".into()))?;
            let candidates = pool
                .workers
                .values()
                .filter(|worker| {
                    matches!(
                        worker.lifecycle,
                        WorkerLifecycle::Draining | WorkerLifecycle::Stopped
                    )
                })
                .map(|worker| worker.spec.id)
                .take(count)
                .collect::<Vec<_>>();
            if candidates.len() != count {
                return Err(RunnerError::Invalid(format!(
                    "pool {pool_id} does not have enough registered workers to reach {desired_workers}"
                )));
            }
            for worker_id in candidates {
                actions.extend(self.handle(DriverCommand::ActivateWorker(worker_id))?);
            }
        }
        actions.extend(self.handle(DriverCommand::SetPoolState {
            pool_id,
            state: PoolState::Ready,
        })?);
        Ok(actions)
    }

    pub(super) fn register_worker(&mut self, spec: WorkerSpec) -> RunnerResult<()> {
        spec.host.validate().map_err(|error| {
            RunnerError::Invalid(format!("worker host inventory is invalid: {error}"))
        })?;
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

    pub(super) fn activate_worker(&mut self, worker_id: WorkerId) -> RunnerResult<()> {
        let worker = self.worker_mut(worker_id)?;
        if worker.lifecycle == WorkerLifecycle::Lost {
            return Err(RunnerError::Invalid(format!(
                "lost worker {worker_id} cannot be activated"
            )));
        }
        worker.lifecycle = WorkerLifecycle::Ready;
        self.emit(DriverEventKind::WorkerActivated { worker_id });
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
        submission.request.host.validate().map_err(|error| {
            RunnerError::Invalid(format!("task host requirement is invalid: {error}"))
        })?;
        for input in &submission.request.inputs {
            input
                .validate_for_transport(
                    runmat_execution::value::ValueLimits::default(),
                    runmat_execution::value::ValueTransportContext::Portable,
                )
                .map_err(|error| {
                    RunnerError::Invalid(format!("task input is not portable: {error}"))
                })?;
        }
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
        if !crate::scheduler::scalar_resources_fit(
            &pool.spec.resource_limit,
            &Default::default(),
            &submission.request.resources,
        ) || (!pool.workers.is_empty()
            && !pool.workers.values().any(|worker| {
                submission
                    .request
                    .host
                    .is_satisfied_by(&worker.spec.host)
                    .is_ok()
                    && crate::scheduler::fits(
                        &worker.spec.resources,
                        &Default::default(),
                        &submission.request.resources,
                    )
            }))
        {
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

    pub(super) fn submit_batch(&mut self, submissions: Vec<TaskSubmission>) -> RunnerResult<()> {
        if submissions.is_empty() {
            return Err(RunnerError::Invalid(
                "task submission batch must not be empty".into(),
            ));
        }
        let mut staged = Driver::from_snapshot(self.snapshot.clone())?;
        for submission in submissions {
            staged.submit(submission)?;
        }
        self.snapshot = staged.snapshot;
        Ok(())
    }
}
