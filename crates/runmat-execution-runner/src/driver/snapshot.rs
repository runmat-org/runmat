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

mod allocation;

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
        if self.schema_version != 4 || self.driver_fence == 0 {
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
            if task.submission.request.id != *task_id {
                return Err(RunnerError::Invalid("task identity mismatch".into()));
            }
            if let Some(source) = task.indeterminate_dependency {
                if task.state != runmat_execution::state::TaskState::Indeterminate
                    || task.attempt_count != 0
                    || task.active_attempt.is_some()
                    || task.cancellation.is_some()
                    || !task.submission.dependencies.contains(&source)
                    || self.tasks.get(&source).is_none_or(|dependency| {
                        dependency.state != runmat_execution::state::TaskState::Indeterminate
                    })
                {
                    return Err(RunnerError::Invalid(
                        "dependency uncertainty lacks its exact unresolved source".into(),
                    ));
                }
            }
            if !self.pools.contains_key(&task.submission.request.pool_id)
                || !self.snapshot_dependencies_exist(*task_id)
            {
                return Err(RunnerError::Invalid(
                    "task references an unknown pool or dependency".into(),
                ));
            }
        }
        let mut histories: BTreeMap<TaskId, Vec<&AttemptRecord>> = BTreeMap::new();
        for (attempt_id, attempt) in &self.attempts {
            attempt.request.validate()?;
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
            histories
                .entry(attempt.request.task_id)
                .or_default()
                .push(attempt);
        }
        for (id, task) in &self.tasks {
            if let Some(cancellation) = task.cancellation {
                match cancellation.cause {
                    crate::TaskCancellationCause::Task { task_id, reason } => {
                        if task_id != *id || !self.events.iter().any(|event| matches!(event.kind,
                            super::DriverEventKind::TaskCancellationRequested { task_id: original, reason: original_reason, now_millis }
                                if original == task_id && original_reason == reason && now_millis == cancellation.requested_at_driver_millis)) {
                            return Err(RunnerError::Invalid("task cancellation differs from its original driver decision".into()));
                        }
                    }
                    crate::TaskCancellationCause::Scope { scope_id, reason } => {
                        let state = self.cancellation.state(task.submission.request.scope_id);
                        if !self.events.iter().any(|event| matches!(event.kind,
                            super::DriverEventKind::ScopeCancelled { scope_id: original_scope, reason: original_reason }
                                if original_scope == scope_id && original_reason == reason))
                            || !self
                            .cancellation
                            .is_ancestor(scope_id, task.submission.request.scope_id)
                            || state.is_none_or(|state| {
                                state.reason != reason
                                    || state.requested_at_millis
                                        != cancellation.requested_at_driver_millis
                            })
                        {
                            return Err(RunnerError::Invalid(
                                "task cancellation differs from its original scope decision".into(),
                            ));
                        }
                    }
                    crate::TaskCancellationCause::Gang { gang_id, generation, reason } => {
                        if !self.events.iter().any(|event| matches!(event.kind,
                            super::DriverEventKind::GangCancelled {
                                gang_id: original, generation: original_generation,
                                reason: original_reason, now_millis,
                            } if original == gang_id && original_generation == generation
                                && original_reason == reason
                                && now_millis == cancellation.requested_at_driver_millis))
                        {
                            return Err(RunnerError::Invalid(
                                "gang cancellation differs from its original driver decision".into(),
                            ));
                        }
                    }
                    crate::TaskCancellationCause::Dependency { task_id } => {
                        if self.tasks.get(&task_id).is_none_or(|parent| {
                            parent.state != runmat_execution::state::TaskState::Cancelled
                                || parent.cancellation.is_none_or(|cause| {
                                    cause.requested_at_driver_millis
                                        > cancellation.requested_at_driver_millis
                                })
                        }) {
                            return Err(RunnerError::Invalid(
                                "task cancellation differs from its original dependency decision"
                                    .into(),
                            ));
                        }
                    }
                    _ => {}
                }
            }
            crate::task::validate_history(
                task,
                histories.get(id).map_or(&[], Vec::as_slice),
                self.driver_fence,
                self.now_millis,
            )?;
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
        self.validate_allocations()?;
        Ok(())
    }

    fn snapshot_dependencies_exist(&self, task_id: TaskId) -> bool {
        self.graph
            .dependencies(task_id)
            .is_some_and(|dependencies| dependencies.iter().all(|id| self.tasks.contains_key(id)))
    }
}
