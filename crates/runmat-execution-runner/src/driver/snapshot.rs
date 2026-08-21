use std::collections::BTreeMap;

use runmat_execution::identity::AttemptId;
use runmat_execution::{PoolId, TaskId};
use serde::{Deserialize, Serialize};

use crate::cancellation::{CancellationTree, DeadlineIndex};
use crate::pool::PoolRecord;
use crate::scheduler::{FairnessState, ReadyQueue};
use crate::task::{AttemptRecord, TaskGraph, TaskRecord};

use super::{DriverConfig, DriverEvent};
use crate::{RunnerError, RunnerResult};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DriverSnapshot {
    pub schema_version: u16,
    pub config: DriverConfig,
    pub driver_fence: u64,
    pub now_millis: u64,
    pub next_event_sequence: u64,
    pub graph: TaskGraph,
    pub tasks: BTreeMap<TaskId, TaskRecord>,
    pub attempts: BTreeMap<AttemptId, AttemptRecord>,
    pub pools: BTreeMap<PoolId, PoolRecord>,
    pub ready: ReadyQueue,
    pub cancellation: CancellationTree,
    pub deadlines: DeadlineIndex,
    pub fairness: FairnessState,
    pub events: Vec<DriverEvent>,
}

impl DriverSnapshot {
    pub fn validate(&self) -> RunnerResult<()> {
        if self.schema_version != 1 || self.driver_fence == 0 {
            return Err(RunnerError::Invalid(
                "driver snapshot schema or fence is invalid".into(),
            ));
        }
        if self.next_event_sequence != self.events.len() as u64
            || !self
                .events
                .iter()
                .enumerate()
                .all(|(index, event)| event.sequence == index as u64)
        {
            return Err(RunnerError::Invalid(
                "driver event sequence is not contiguous".into(),
            ));
        }
        for (pool_id, pool) in &self.pools {
            if pool.spec.id != *pool_id {
                return Err(RunnerError::Invalid("pool identity mismatch".into()));
            }
            pool.spec.validate()?;
            for (worker_id, worker) in &pool.workers {
                if worker.spec.id != *worker_id || worker.spec.pool_id != *pool_id {
                    return Err(RunnerError::Invalid("worker identity mismatch".into()));
                }
            }
        }
        for (task_id, task) in &self.tasks {
            if task.submission.request.id != *task_id {
                return Err(RunnerError::Invalid("task identity mismatch".into()));
            }
            if !self.pools.contains_key(&task.submission.request.pool_id)
                || !self.snapshot_dependencies_exist(*task_id)
            {
                return Err(RunnerError::Invalid(
                    "task references an unknown pool or dependency".into(),
                ));
            }
            if let Some(attempt_id) = task.active_attempt {
                let attempt = self.attempts.get(&attempt_id).ok_or_else(|| {
                    RunnerError::Invalid("task references an unknown active attempt".into())
                })?;
                if attempt.request.task_id != *task_id {
                    return Err(RunnerError::Invalid(
                        "active attempt references a different task".into(),
                    ));
                }
            }
        }
        for (attempt_id, attempt) in &self.attempts {
            if attempt.request.id != *attempt_id
                || !self.tasks.contains_key(&attempt.request.task_id)
                || !self
                    .pools
                    .get(&attempt.request.task.pool_id)
                    .is_some_and(|pool| pool.workers.contains_key(&attempt.request.worker_id))
            {
                return Err(RunnerError::Invalid(
                    "attempt identity or ownership reference is invalid".into(),
                ));
            }
        }
        if self.ready.task_ids().any(|task_id| {
            !self
                .tasks
                .get(&task_id)
                .is_some_and(|task| task.state == runmat_execution::state::TaskState::Ready)
        }) {
            return Err(RunnerError::Invalid(
                "ready queue contains a task that is not ready".into(),
            ));
        }
        if self.tasks.iter().any(|(task_id, task)| {
            task.state == runmat_execution::state::TaskState::Ready
                && !self.ready.contains(*task_id)
        }) {
            return Err(RunnerError::Invalid(
                "ready task is missing from the ready queue".into(),
            ));
        }
        Ok(())
    }

    fn snapshot_dependencies_exist(&self, task_id: TaskId) -> bool {
        self.graph
            .dependencies(task_id)
            .is_some_and(|dependencies| dependencies.iter().all(|id| self.tasks.contains_key(id)))
    }
}
