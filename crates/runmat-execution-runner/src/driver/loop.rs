use std::collections::BTreeMap;

use runmat_execution::state::PoolState;

use crate::cancellation::{CancellationTree, DeadlineIndex};
use crate::pool::{PoolRecord, ResizeDecision};
use crate::scheduler::{FairnessState, ReadyQueue};
use crate::task::TaskGraph;
use crate::{RunnerError, RunnerResult};

use super::{
    DriverAction, DriverCommand, DriverConfig, DriverEvent, DriverEventKind, DriverSnapshot,
};

pub struct Driver {
    pub(super) snapshot: DriverSnapshot,
}

impl Driver {
    pub fn new(config: DriverConfig, driver_fence: u64) -> RunnerResult<Self> {
        if config.max_in_flight == 0 || config.fairness.max_priority_burst == 0 {
            return Err(RunnerError::Invalid(
                "driver in-flight and fairness bounds must be non-zero".into(),
            ));
        }
        if driver_fence == 0 {
            return Err(RunnerError::Invalid(
                "driver fence must be greater than zero".into(),
            ));
        }
        Ok(Self {
            snapshot: DriverSnapshot {
                schema_version: 4,
                config,
                driver_fence,
                now_millis: 0,
                next_event_sequence: 0,
                graph: TaskGraph::default(),
                tasks: BTreeMap::new(),
                attempts: BTreeMap::new(),
                pools: BTreeMap::new(),
                ready: ReadyQueue::default(),
                cancellation: CancellationTree::default(),
                deadlines: DeadlineIndex::default(),
                fairness: FairnessState::default(),
                events: Vec::new(),
            },
        })
    }

    pub(crate) fn from_snapshot(snapshot: DriverSnapshot) -> RunnerResult<Self> {
        snapshot.validate()?;
        Ok(Self { snapshot })
    }

    /// Replaces the current in-memory state with a fully validated snapshot.
    ///
    /// Hosts use this to roll back a command whose surrounding durable admission could not be
    /// committed. No actions from the rejected state may have been dispatched before restoration.
    pub fn restore_snapshot(&mut self, snapshot: DriverSnapshot) -> RunnerResult<()> {
        snapshot.validate()?;
        self.snapshot = snapshot;
        Ok(())
    }

    pub fn snapshot(&self) -> DriverSnapshot {
        self.snapshot.clone()
    }

    /// Borrow complete original history for validation and serialization.
    /// The borrow prevents driver mutation until the reader releases it. Use an
    /// owned snapshot only when independent retention or rollback requires it.
    pub fn snapshot_ref(&self) -> &DriverSnapshot {
        &self.snapshot
    }

    /// Current logical driver clock without copying its task/attempt inventory.
    pub fn now_millis(&self) -> u64 {
        self.snapshot.now_millis
    }

    pub fn events(&self) -> &[DriverEvent] {
        &self.snapshot.events
    }

    fn advance_clock(&mut self, now_millis: u64) -> RunnerResult<()> {
        if now_millis < self.snapshot.now_millis {
            return Err(RunnerError::Invalid(
                "driver clock cannot move backwards".into(),
            ));
        }
        self.snapshot.now_millis = now_millis;
        Ok(())
    }

    pub fn handle(&mut self, command: DriverCommand) -> RunnerResult<Vec<DriverAction>> {
        let mut actions = Vec::new();
        match command {
            DriverCommand::RegisterScope { scope_id, parent } => {
                self.snapshot.cancellation.register(scope_id, parent)?;
                self.emit(DriverEventKind::ScopeRegistered { scope_id });
            }
            DriverCommand::CreatePool(spec) => {
                spec.validate()?;
                if self.snapshot.pools.contains_key(&spec.id) {
                    return Err(RunnerError::Invalid(format!(
                        "pool {} already exists",
                        spec.id
                    )));
                }
                let pool_id = spec.id;
                self.snapshot.pools.insert(pool_id, PoolRecord::new(spec));
                self.emit(DriverEventKind::PoolCreated { pool_id });
            }
            DriverCommand::RefreshPoolCapabilities { pool_id } => {
                let pool = self.pool_mut(pool_id)?;
                let capabilities = pool
                    .workers
                    .values()
                    .filter(|worker| worker.accepts_work())
                    .flat_map(|worker| worker.spec.resources.capabilities.iter().cloned())
                    .collect::<std::collections::BTreeSet<_>>();
                if pool.spec.resource_limit.capabilities != capabilities {
                    pool.spec.resource_limit.capabilities = capabilities;
                }
            }
            DriverCommand::SetPoolState { pool_id, state } => {
                self.pool_mut(pool_id)?.state = state;
                self.emit(DriverEventKind::PoolStateChanged { pool_id, state });
            }
            DriverCommand::ResizePool { pool_id, request } => {
                let pool = self.pool_mut(pool_id)?;
                let available_workers = pool
                    .workers
                    .values()
                    .filter(|worker| worker.accepts_work())
                    .count();
                let available_workers = u32::try_from(available_workers).map_err(|_| {
                    RunnerError::Invalid("pool worker count exceeds the supported range".into())
                })?;
                let decision = request.decide(&pool.spec, available_workers)?;
                if decision != ResizeDecision::Unchanged {
                    pool.state = PoolState::Resizing;
                    actions.push(DriverAction::ResizePool {
                        pool_id,
                        desired_workers: request.desired_workers,
                    });
                    self.emit(DriverEventKind::PoolResizeRequested {
                        pool_id,
                        desired_workers: request.desired_workers,
                    });
                }
            }
            DriverCommand::RegisterWorker(spec) => self.register_worker(spec)?,
            DriverCommand::AdmitWorkerSoftwareCapabilities {
                worker_id,
                capabilities,
            } => {
                // A ReadyV1 can be observed before a concurrent removal. The
                // canonical driver state, not that earlier observation, decides
                // whether this worker may receive software capabilities.
                if let Some(worker) = self
                    .snapshot
                    .pools
                    .values_mut()
                    .find_map(|pool| pool.workers.get_mut(&worker_id))
                    .filter(|worker| worker.accepts_work())
                {
                    worker.spec.resources.capabilities.extend(capabilities);
                }
            }
            DriverCommand::ActivateWorker(worker_id) => self.activate_worker(worker_id)?,
            DriverCommand::DrainWorker(worker_id) => self.drain_worker(worker_id)?,
            DriverCommand::WorkerLost(worker_id) => {
                self.worker_lost(worker_id, &mut actions)?;
            }
            DriverCommand::WorkerLostWithReport { worker_id, report } => {
                self.worker_lost_with_report(worker_id, *report, &mut actions)?;
            }
            DriverCommand::Submit(submission) => self.submit(*submission)?,
            DriverCommand::SubmitBatch(submissions) => self.submit_batch(submissions)?,
            DriverCommand::BackendReport(report) => {
                self.apply_backend_report(report, &mut actions)?;
            }
            DriverCommand::CancelTask {
                task_id,
                reason,
                now_millis,
            } => {
                self.advance_clock(now_millis)?;
                if !self.task_is_terminal(task_id)? {
                    self.emit(DriverEventKind::TaskCancellationRequested {
                        task_id,
                        reason,
                        now_millis,
                    });
                }
                self.cancel_task(
                    task_id,
                    now_millis,
                    crate::TaskCancellationCause::Task { task_id, reason },
                    &mut actions,
                )?;
            }
            DriverCommand::CancelScope {
                scope_id,
                reason,
                now_millis,
            } => {
                self.advance_clock(now_millis)?;
                let scopes = self
                    .snapshot
                    .cancellation
                    .cancel(scope_id, reason, now_millis)?;
                self.emit(DriverEventKind::ScopeCancelled { scope_id, reason });
                let tasks = self
                    .snapshot
                    .tasks
                    .iter()
                    .filter(|(_, task)| scopes.contains(&task.submission.request.scope_id))
                    .map(|(task_id, _)| *task_id)
                    .collect::<Vec<_>>();
                for task_id in tasks {
                    self.cancel_task(
                        task_id,
                        now_millis,
                        crate::TaskCancellationCause::Scope { scope_id, reason },
                        &mut actions,
                    )?;
                }
            }
            DriverCommand::CancelGang {
                gang,
                reason,
                now_millis,
            } => {
                gang.validate()
                    .map_err(|error| RunnerError::Invalid(error.to_string()))?;
                self.advance_clock(now_millis)?;
                let tasks = self
                    .snapshot
                    .tasks
                    .iter()
                    .filter_map(|(task_id, task)| {
                        match &task.submission.request.invocation_context {
                            runmat_execution::ProgramInvocationContext::SpmdTask { task }
                                if task.gang == gang =>
                            {
                                Some(*task_id)
                            }
                            _ => None,
                        }
                    })
                    .collect::<Vec<_>>();
                if tasks.is_empty() {
                    return Err(RunnerError::Invalid(
                        "SPMD gang has no tasks in this driver".into(),
                    ));
                }
                self.emit(DriverEventKind::GangCancelled {
                    gang_id: gang.id,
                    generation: gang.generation,
                    reason,
                    now_millis,
                });
                for task_id in tasks {
                    self.cancel_task(
                        task_id,
                        now_millis,
                        crate::TaskCancellationCause::Gang {
                            gang_id: gang.id,
                            generation: gang.generation,
                            reason,
                        },
                        &mut actions,
                    )?;
                }
            }
            DriverCommand::Tick { now_millis } => {
                self.advance_clock(now_millis)?;
                for task_id in self.snapshot.deadlines.expired(now_millis) {
                    if !self.task_is_terminal(task_id)? {
                        self.emit(DriverEventKind::DeadlineExpired { task_id });
                        self.cancel_task(
                            task_id,
                            now_millis,
                            crate::TaskCancellationCause::Deadline,
                            &mut actions,
                        )?;
                    }
                }
                self.expire_attempt_wall_times(now_millis, &mut actions)?;
                self.escalate_cancellations(now_millis, &mut actions)?;
            }
            DriverCommand::Checkpoint => {
                self.emit(DriverEventKind::CheckpointRequested);
                actions.push(DriverAction::Checkpoint);
            }
        }
        self.schedule(&mut actions)?;
        Ok(actions)
    }
}
