use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution::TaskId;

use crate::scheduler::{self, QueueEntry};
use crate::task::{AttemptRecord, AttemptRequest};
use crate::{RunnerError, RunnerResult};

use super::{Driver, DriverAction, DriverEventKind};

impl Driver {
    pub(super) fn schedule(&mut self, actions: &mut Vec<DriverAction>) -> RunnerResult<()> {
        loop {
            let active = self
                .snapshot
                .pools
                .values()
                .map(|pool| pool.active_attempts)
                .sum::<u32>();
            if active >= self.snapshot.config.max_in_flight || self.snapshot.ready.is_empty() {
                return Ok(());
            }
            let entries = self.snapshot.ready.ordered().collect::<Vec<_>>();
            let mut fairness = self.snapshot.fairness;
            let Some(priority) = fairness.select_priority(
                entries.iter().map(|entry| entry.priority),
                self.snapshot.config.fairness,
            ) else {
                return Ok(());
            };
            let preferred = entries
                .iter()
                .filter(|entry| entry.priority == priority)
                .find_map(|entry| {
                    self.placement_for(entry.task_id)
                        .map(|worker| (*entry, worker))
                });
            let placement = preferred.or_else(|| {
                entries.iter().find_map(|entry| {
                    self.placement_for(entry.task_id)
                        .map(|worker| (*entry, worker))
                })
            });
            let Some((entry, worker_id)) = placement else {
                return Ok(());
            };
            if entry.priority == priority {
                self.snapshot.fairness = fairness;
            } else {
                self.snapshot.fairness.record(entry.priority);
            }
            self.assign(entry, worker_id, actions)?;
        }
    }

    fn placement_for(&self, task_id: TaskId) -> Option<WorkerId> {
        let task = self.snapshot.tasks.get(&task_id)?;
        let pool = self.snapshot.pools.get(&task.submission.request.pool_id)?;
        if !pool.accepts_work() || !pool.fits(&task.submission.request.resources) {
            return None;
        }
        scheduler::choose_worker(pool.workers.values(), &task.submission.request.resources)
            .map(|candidate| candidate.worker_id)
    }

    fn assign(
        &mut self,
        entry: QueueEntry,
        worker_id: WorkerId,
        actions: &mut Vec<DriverAction>,
    ) -> RunnerResult<()> {
        let task = self
            .snapshot
            .tasks
            .get_mut(&entry.task_id)
            .ok_or(RunnerError::UnknownTask(entry.task_id))?;
        task.attempt_count = task
            .attempt_count
            .checked_add(1)
            .ok_or_else(|| RunnerError::Invalid("task attempt count overflow".into()))?;
        let ordinal = task.attempt_count;
        let fence = self.snapshot.driver_fence.to_be_bytes();
        let ordinal_bytes = ordinal.to_be_bytes();
        let attempt_id = AttemptId::derive(&[
            entry.task_id.bytes().as_slice(),
            ordinal_bytes.as_slice(),
            fence.as_slice(),
        ]);
        let request = AttemptRequest {
            id: attempt_id,
            task_id: entry.task_id,
            scope_id: task.submission.request.scope_id,
            worker_id,
            ordinal,
            driver_fence: self.snapshot.driver_fence,
            task: task.submission.request.clone(),
        };
        task.state = TaskState::Assigned;
        task.active_attempt = Some(attempt_id);
        self.snapshot.ready.remove_task(entry.task_id);
        let pool = self.pool_mut(request.task.pool_id)?;
        scheduler::reserve(&mut pool.allocated, &request.task.resources)?;
        let worker = pool
            .workers
            .get_mut(&worker_id)
            .ok_or(RunnerError::UnknownWorker(worker_id))?;
        scheduler::reserve(&mut worker.allocated, &request.task.resources)?;
        worker.active_attempts = worker.active_attempts.saturating_add(1);
        pool.active_attempts = pool.active_attempts.saturating_add(1);
        self.snapshot.attempts.insert(
            attempt_id,
            AttemptRecord {
                request: request.clone(),
                state: AttemptState::Assigned,
                assigned_at_millis: self.snapshot.now_millis,
                cancellation_requested_at: None,
                cancellation_escalation: None,
            },
        );
        self.emit(DriverEventKind::AttemptAssigned {
            task_id: entry.task_id,
            attempt_id,
            worker_id,
        });
        actions.push(DriverAction::Launch(request));
        Ok(())
    }
}
