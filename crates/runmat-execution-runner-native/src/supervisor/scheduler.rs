//! One exclusively owned portable scheduler journal, never a post-execution witness.
use super::filesystem::{atomic_json, read_optional_json, unix_millis};
use super::{DurableExecutionContext, LocalSupervisorConfig};
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::{Digest, ExecutionScopeId, JobId, PoolId, TaskId};
use runmat_execution_runner::recovery::DriverCheckpoint;
use runmat_execution_runner::{
    Driver, DriverAction, DriverCommand, DriverConfig, TaskExecutionEvidence, TaskSubmission,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Journal<
    C = BTreeMap<Digest, ExecutionScopeId>,
    R = BTreeMap<TaskId, JobId>,
    S = DriverCheckpoint,
> {
    schema_version: u16,
    config_identity: Digest,
    root_scope: ExecutionScopeId,
    pool_id: PoolId,
    contexts: C,
    routes: R,
    checkpoint: S,
}

pub(super) struct DurableScheduler {
    path: std::path::PathBuf,
    config_identity: Digest,
    root_scope: ExecutionScopeId,
    pub(super) pool_id: PoolId,
    contexts: BTreeMap<Digest, ExecutionScopeId>,
    routes: BTreeMap<TaskId, JobId>,
    pub(super) driver: Driver,
    publication_pending: bool,
}

impl DurableScheduler {
    // Requiring the actual guard makes the native ownership boundary reviewable.
    // The caller retains it for the entire supervisor lifetime, including all reads.
    pub(super) fn open(
        config: &LocalSupervisorConfig,
        _authority: &runmat_filesystem::ExclusiveFileLock,
    ) -> NativeExecutionResult<Self> {
        let path = config.paths.root.join("scheduler.json");
        let config_identity = config.identity()?;
        let (root_scope, pool_id, contexts, routes, driver) =
            if let Some(journal) = read_optional_json::<Journal>(&path)? {
                journal.checkpoint.validate()?;
                if journal.schema_version != 1
                    || journal.config_identity != config_identity
                    || !journal
                        .checkpoint
                        .snapshot
                        .pools
                        .contains_key(&journal.pool_id)
                    || journal
                        .routes
                        .keys()
                        .any(|id| !journal.checkpoint.snapshot.tasks.contains_key(id))
                {
                    return Err(invalid(
                        "durable scheduler journal differs from its configured authority",
                    ));
                }
                (
                    journal.root_scope,
                    journal.pool_id,
                    journal.contexts,
                    journal.routes,
                    journal.checkpoint.continue_under_exclusive_ownership()?,
                )
            } else {
                let scope = ExecutionScopeId::derive(&[
                    b"native-durable-authority",
                    uuid::Uuid::new_v4().as_bytes(),
                ]);
                let mut driver = Driver::new(DriverConfig::default(), 1)?;
                let pool = crate::native_pool::register(&mut driver, &config.execution, scope)?;
                (scope, pool, BTreeMap::new(), BTreeMap::new(), driver)
            };
        let mut owner = Self {
            path,
            config_identity,
            root_scope,
            pool_id,
            contexts,
            routes,
            driver,
            publication_pending: true,
        };
        // Reaffirm retention before any new admission or dispatch after cold reopen.
        owner.persist()?;
        Ok(owner)
    }

    pub(super) fn job_has_provider(
        &self,
        store: &super::store::JobStore,
        job: JobId,
    ) -> NativeExecutionResult<bool> {
        for (&task, _) in self.routes.iter().filter(|(_, routed)| **routed == job) {
            for attempt in self.driver.task_attempts(task)? {
                if super::ownership::is_owned(&store.job_dir(job), &attempt.request)? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    pub(super) fn context(
        &mut self,
        key: Option<&str>,
    ) -> NativeExecutionResult<DurableExecutionContext> {
        self.ensure_published()?;
        let key = match key {
            Some(key) if !key.is_empty() && key.len() <= 256 && !key.contains('\0') => {
                Digest::sha256([b"native-durable-context/v1\0".as_slice(), key.as_bytes()].concat())
            }
            Some(_) => return Err(invalid("durable context idempotency key is invalid")),
            None => Digest::sha256(uuid::Uuid::new_v4().as_bytes()),
        };
        if let Some(scope_id) = self.contexts.get(&key) {
            return Ok(DurableExecutionContext {
                scope_id: *scope_id,
                pool_id: self.pool_id,
            });
        }
        let scope_id = ExecutionScopeId::derive(&[self.root_scope.bytes(), key.bytes()]);
        let before = self.driver.snapshot();
        self.driver.handle(DriverCommand::RegisterScope {
            scope_id,
            parent: Some(self.root_scope),
        })?;
        self.contexts.insert(key, scope_id);
        if let Err(error) = self.persist() {
            if !published(&error) {
                self.driver.restore_snapshot(before)?;
                self.contexts.remove(&key);
            }
            return Err(error);
        }
        Ok(DurableExecutionContext {
            scope_id,
            pool_id: self.pool_id,
        })
    }

    pub(super) fn admit(
        &mut self,
        job_id: JobId,
        task: &TaskSubmission,
    ) -> NativeExecutionResult<Vec<DriverAction>> {
        self.ensure_published()?;
        let request = &task.request;
        if request.pool_id != self.pool_id
            || !self.contexts.values().any(|id| *id == request.scope_id)
        {
            return Err(invalid(
                "durable task has no original supervisor execution context",
            ));
        }
        if let Some(route) = self.routes.get(&request.id) {
            if *route != job_id
                || self
                    .driver
                    .task_execution_evidence(request.id)?
                    .task
                    .submission
                    != *task
            {
                return Err(invalid(
                    "durable task identity is already bound to another original admission",
                ));
            }
            return Ok(Vec::new());
        }
        if self.routes.keys().any(|id| {
            self.driver
                .task_execution_evidence(*id)
                .is_ok_and(|history| history.task.submission.request.scope_id == request.scope_id)
        }) {
            return Err(invalid(
                "durable execution context is already assigned to a different task",
            ));
        }
        let before = self.driver.snapshot();
        let actions = match self
            .driver
            .handle(DriverCommand::Submit(Box::new(task.clone())))
        {
            Ok(actions) => actions,
            Err(error) => {
                self.driver.restore_snapshot(before)?;
                return Err(error.into());
            }
        };
        self.routes.insert(request.id, job_id);
        if let Err(error) = self.persist() {
            if !published(&error) {
                self.driver.restore_snapshot(before)?;
                self.routes.remove(&request.id);
            }
            return Err(error);
        }
        Ok(actions)
    }

    pub(super) fn command(
        &mut self,
        command: DriverCommand,
    ) -> NativeExecutionResult<Vec<DriverAction>> {
        self.ensure_published()?;
        let before = self.driver.snapshot();
        let actions = match self.driver.handle(command) {
            Ok(actions) => actions,
            Err(error) => {
                self.driver.restore_snapshot(before)?;
                return Err(error.into());
            }
        };
        if let Err(error) = self.persist() {
            if !published(&error) {
                self.driver.restore_snapshot(before)?;
            }
            return Err(error);
        }
        Ok(actions)
    }

    pub(super) fn route(&self, task: TaskId) -> NativeExecutionResult<JobId> {
        self.routes
            .get(&task)
            .copied()
            .ok_or_else(|| invalid("assigned task lacks its original durable routing record"))
    }

    pub(super) fn validate_context(&self, task: &TaskSubmission) -> NativeExecutionResult<()> {
        if task.request.pool_id != self.pool_id
            || !self
                .contexts
                .values()
                .any(|scope| *scope == task.request.scope_id)
        {
            return Err(invalid(
                "task differs from its original supervisor execution context",
            ));
        }
        Ok(())
    }

    pub(super) fn history(
        &self,
        job_id: JobId,
    ) -> NativeExecutionResult<Option<TaskExecutionEvidence>> {
        self.task_for_job(job_id)
            .map(|task| {
                self.driver
                    .task_execution_evidence(task)
                    .map_err(Into::into)
            })
            .transpose()
    }

    pub(super) fn task_for_job(&self, job_id: JobId) -> Option<TaskId> {
        self.routes
            .iter()
            .find_map(|(task, job)| (*job == job_id).then_some(*task))
    }

    fn ensure_published(&mut self) -> NativeExecutionResult<()> {
        if self.publication_pending {
            self.persist()?;
        }
        Ok(())
    }

    fn persist(&mut self) -> NativeExecutionResult<()> {
        let snapshot = self.driver.snapshot_ref();
        snapshot.validate()?;
        let journal = Journal {
            schema_version: 1,
            config_identity: self.config_identity,
            root_scope: self.root_scope,
            pool_id: self.pool_id,
            contexts: &self.contexts,
            routes: &self.routes,
            checkpoint: DriverCheckpoint::new(snapshot.next_event_sequence.max(1), snapshot),
        };
        self.publication_pending = true;
        atomic_json(&self.path, &journal)?;
        self.publication_pending = false;
        Ok(())
    }

    pub(super) fn tick_if_active(&mut self) -> NativeExecutionResult<Vec<DriverAction>> {
        if self.driver.task_records().any(|task| {
            !matches!(
                task.state,
                runmat_execution::state::TaskState::Succeeded
                    | runmat_execution::state::TaskState::Failed
                    | runmat_execution::state::TaskState::Cancelled
                    | runmat_execution::state::TaskState::Indeterminate
            ) || self
                .driver
                .task_attempts(task.id())
                .is_ok_and(|mut attempts| attempts.any(|attempt| attempt.terminal.is_none()))
        }) {
            self.tick()
        } else {
            Ok(Vec::new())
        }
    }

    pub(super) fn tick(&mut self) -> NativeExecutionResult<Vec<DriverAction>> {
        self.command(DriverCommand::Tick {
            now_millis: unix_millis().max(self.driver.now_millis()),
        })
    }
}

fn published(error: &NativeExecutionError) -> bool {
    matches!(error, NativeExecutionError::CheckpointDelivery { evidence, .. }
        if matches!(evidence.as_ref(), crate::NativeCheckpointEvidence::DurabilityIndeterminate { .. }))
}
fn invalid(message: &str) -> NativeExecutionError {
    NativeExecutionError::Protocol(message.into())
}

#[cfg(all(test, unix))]
mod tests;
