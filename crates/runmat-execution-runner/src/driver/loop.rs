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
                schema_version: 1,
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

    pub fn snapshot(&self) -> DriverSnapshot {
        self.snapshot.clone()
    }

    pub fn events(&self) -> &[DriverEvent] {
        &self.snapshot.events
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
            DriverCommand::SetPoolState { pool_id, state } => {
                self.pool_mut(pool_id)?.state = state;
                self.emit(DriverEventKind::PoolStateChanged { pool_id, state });
            }
            DriverCommand::ResizePool { pool_id, request } => {
                let pool = self.pool_mut(pool_id)?;
                let decision = request.decide(&pool.spec, pool.workers.len() as u32)?;
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
            DriverCommand::DrainWorker(worker_id) => self.drain_worker(worker_id)?,
            DriverCommand::WorkerLost(worker_id) => {
                self.worker_lost(worker_id, &mut actions)?;
            }
            DriverCommand::Submit(submission) => self.submit(*submission)?,
            DriverCommand::BackendReport(report) => {
                self.apply_backend_report(report, &mut actions)?;
            }
            DriverCommand::CancelScope {
                scope_id,
                reason,
                now_millis,
            } => {
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
                    self.cancel_task(task_id, now_millis, &mut actions)?;
                }
            }
            DriverCommand::Tick { now_millis } => {
                self.snapshot.now_millis = now_millis;
                for task_id in self.snapshot.deadlines.expired(now_millis) {
                    if !self.task_is_terminal(task_id)? {
                        self.emit(DriverEventKind::DeadlineExpired { task_id });
                        self.cancel_task(task_id, now_millis, &mut actions)?;
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
