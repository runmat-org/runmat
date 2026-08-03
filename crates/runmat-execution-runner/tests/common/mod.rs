#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap};

use runmat_execution::handle::OutputContract;
use runmat_execution::identity::{ArtifactId, WorkerId};
use runmat_execution::resource::{ResourceInventory, ResourceRequest};
use runmat_execution::state::PoolState;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_execution::{Digest, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_runner::driver::{DriverAction, DriverCommand, DriverConfig};
use runmat_execution_runner::{Driver, PoolSpec, TaskSubmission, WorkerSpec};

pub struct Fixture {
    pub driver: Driver,
    pub scope: ExecutionScopeId,
    pub pool: PoolId,
    pub workers: Vec<WorkerId>,
}

pub fn fixture(worker_count: usize, max_in_flight: u32) -> Fixture {
    let scope = ExecutionScopeId::derive(&[b"scope"]);
    let pool = PoolId::derive(&[b"pool"]);
    let mut driver = Driver::new(
        DriverConfig {
            max_in_flight,
            ..DriverConfig::default()
        },
        7,
    )
    .unwrap();
    driver
        .handle(DriverCommand::RegisterScope {
            scope_id: scope,
            parent: None,
        })
        .unwrap();
    driver
        .handle(DriverCommand::CreatePool(PoolSpec {
            id: pool,
            min_workers: 0,
            max_workers: worker_count.max(1) as u32,
            max_in_flight,
            resource_limit: ResourceInventory {
                cpu_millicores: 2_000 * worker_count.max(1) as u32,
                memory_bytes: 4 * 1024 * 1024 * worker_count.max(1) as u64,
                scratch_bytes: 4 * 1024 * 1024 * worker_count.max(1) as u64,
                accelerators: Vec::new(),
                capabilities: BTreeSet::new(),
            },
        }))
        .unwrap();
    driver
        .handle(DriverCommand::SetPoolState {
            pool_id: pool,
            state: PoolState::Ready,
        })
        .unwrap();
    let mut workers = Vec::new();
    for ordinal in 0..worker_count {
        let worker = WorkerId::derive(&[b"worker", &ordinal.to_be_bytes()]);
        driver
            .handle(DriverCommand::RegisterWorker(WorkerSpec {
                id: worker,
                pool_id: pool,
                resources: inventory(),
            }))
            .unwrap();
        workers.push(worker);
    }
    Fixture {
        driver,
        scope,
        pool,
        workers,
    }
}

pub fn inventory() -> ResourceInventory {
    ResourceInventory {
        cpu_millicores: 2_000,
        memory_bytes: 4 * 1024 * 1024,
        scratch_bytes: 4 * 1024 * 1024,
        accelerators: Vec::new(),
        capabilities: BTreeSet::new(),
    }
}

pub fn task(
    name: &str,
    scope: ExecutionScopeId,
    pool: PoolId,
    retry: RetryPolicy,
) -> TaskSubmission {
    TaskSubmission {
        request: TaskRequest {
            id: TaskId::derive(&[name.as_bytes()]),
            scope_id: scope,
            pool_id: pool,
            program_artifact_id: ArtifactId::derive(&[b"artifact"]),
            callable: Callable {
                owner_identity: "fixture".into(),
                qualified_name: name.into(),
                entrypoint_digest: Digest::sha256(name),
            },
            inputs: Vec::new(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            resources: ResourceRequest {
                cpu_millicores: 1_000,
                memory_bytes: 1024,
                scratch_bytes: 1024,
                max_wall_millis: 10_000,
                max_artifact_bytes: 1024,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: Vec::new(),
                required_capabilities: BTreeSet::new(),
            },
            retry,
            deadline_unix_millis: None,
        },
        dependencies: BTreeSet::new(),
        priority: 0,
    }
}

pub fn submit(
    driver: &mut Driver,
    task: TaskSubmission,
) -> runmat_execution_runner::AttemptRequest {
    driver
        .handle(DriverCommand::Submit(Box::new(task)))
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .expect("task should be launched")
}

pub fn success() -> runmat_execution_runner::AttemptReport {
    runmat_execution_runner::AttemptReport::Succeeded {
        result: runmat_execution_runner::AttemptSuccess {
            outputs: vec![ValuePayload::Inline(Box::new(InlineValue::Null))],
            result_objects: Vec::new(),
        },
    }
}

pub fn normalize_actions(actions: &[DriverAction]) -> HashMap<&'static str, usize> {
    let mut counts = HashMap::new();
    for action in actions {
        let name = match action {
            DriverAction::Launch(_) => "launch",
            DriverAction::Cancel(_) => "cancel",
            DriverAction::Terminate(_) => "terminate",
            DriverAction::ResizePool { .. } => "resize",
            DriverAction::Checkpoint => "checkpoint",
            DriverAction::GarbageCollectResults { .. } => "gc",
        };
        *counts.entry(name).or_default() += 1;
    }
    counts
}
