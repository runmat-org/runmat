use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::identity::AttemptId;
use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution::{PoolId, TaskId};
use serde::{Deserialize, Serialize};

use crate::cancellation::{CancellationTree, DeadlineIndex};
use crate::pool::PoolRecord;
use crate::scheduler::{FairnessState, ReadyQueue};
use crate::task::{AttemptRecord, AttemptSuccess, ResultCommit, TaskGraph, TaskRecord};

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
        if self.schema_version != 2 || self.driver_fence == 0 {
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
        self.graph.validate_submissions(
            self.tasks
                .iter()
                .map(|(id, task)| (*id, &task.submission.dependencies)),
        )?;
        for (task_id, task) in &self.tasks {
            let retry = task.submission.retry_resource_envelope()?;
            if task.attempt_count > retry.maximum_attempts {
                return Err(RunnerError::Invalid(
                    "attempt history exceeds admitted retry envelope".into(),
                ));
            }
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
            self.validate_commit(*task_id, task)?;
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
        let mut ordinals: BTreeMap<TaskId, BTreeSet<u16>> = BTreeMap::new();
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
            attempt
                .validate_terminal(self.now_millis)
                .map_err(RunnerError::Invalid)?;
            let task = &self.tasks[&attempt.request.task_id];
            let request = &attempt.request;
            let expected_id = AttemptId::derive(&[
                request.task_id.bytes().as_slice(),
                request.ordinal.to_be_bytes().as_slice(),
                request.driver_fence.to_be_bytes().as_slice(),
            ]);
            if request.id != expected_id
                || request.task != task.submission.request
                || request.scope_id != task.submission.request.scope_id
                || request.ordinal == 0
                || request.ordinal > task.attempt_count
                || request.driver_fence == 0
                || request.driver_fence > self.driver_fence
                || !ordinals
                    .entry(request.task_id)
                    .or_default()
                    .insert(request.ordinal)
            {
                return Err(RunnerError::Invalid(
                    "snapshot attempt differs from its admitted task or fence".into(),
                ));
            }
        }
        for (id, task) in &self.tasks {
            if ordinals.get(id).map_or(0, BTreeSet::len) != usize::from(task.attempt_count) {
                return Err(RunnerError::Invalid(
                    "snapshot attempt history differs from retry count".into(),
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

    fn validate_commit(&self, task_id: TaskId, task: &TaskRecord) -> RunnerResult<()> {
        if (task.state == TaskState::Succeeded) != task.committed.is_some() {
            return Err(RunnerError::Invalid(
                "snapshot task success and committed result disagree".into(),
            ));
        }
        let Some(commit) = &task.committed else {
            return Ok(());
        };
        let attempt = self.attempts.get(&commit.attempt_id).ok_or_else(|| {
            RunnerError::Invalid("snapshot commit references an unknown attempt".into())
        })?;
        if task.active_attempt.is_some()
            || attempt.state != AttemptState::Completed
            || attempt.request.task_id != task_id
            || commit.driver_fence != attempt.request.driver_fence
        {
            return Err(RunnerError::Invalid(
                "snapshot commit does not belong to a completed task attempt".into(),
            ));
        }
        if attempt.terminal.as_ref().is_none_or(|terminal| {
            terminal.disposition != crate::task::AttemptDisposition::Succeeded
                || terminal.resources != commit.resources
        }) {
            return Err(RunnerError::Invalid(
                "commit differs from original terminal attempt evidence".into(),
            ));
        }
        let success = AttemptSuccess {
            outputs: commit.outputs.clone(),
            result_objects: commit.result_objects.clone(),
            resources: commit.resources.clone(),
        };
        success
            .validate_against(&attempt.request)
            .map_err(RunnerError::Invalid)?;
        if *commit != ResultCommit::from_success(commit.attempt_id, commit.driver_fence, success) {
            return Err(RunnerError::Invalid(
                "snapshot commit identity is invalid".into(),
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
