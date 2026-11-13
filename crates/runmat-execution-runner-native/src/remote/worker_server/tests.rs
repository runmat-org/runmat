use std::collections::BTreeSet;

use runmat_execution::resource::{ResourceInventory, ResourceRequest};
use runmat_execution_runner::scheduler::{release, ResourceAllocation};

use super::{reserve_attempt_capacity, ActiveAttempt, WorkerState};

struct SilentRoute;

#[async_trait::async_trait]
impl super::super::route::RemoteFrameRoute for SilentRoute {
    async fn send(
        &self,
        _frame: runmat_execution_transport_native::frame::WireFrame,
    ) -> crate::NativeExecutionResult<()> {
        Ok(())
    }

    async fn receive(
        &self,
    ) -> crate::NativeExecutionResult<runmat_execution_transport_native::frame::WireFrame> {
        std::future::pending().await
    }
}

fn inventory() -> ResourceInventory {
    ResourceInventory {
        cpu_millicores: 2_000,
        memory_bytes: 100,
        scratch_bytes: 50,
        accelerators: Vec::new(),
        capabilities: BTreeSet::new(),
    }
}

fn request(memory_bytes: u64) -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes,
        scratch_bytes: 20,
        max_wall_millis: 1_000,
        max_artifact_bytes: 100,
        max_egress_bytes: 100,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::new(),
    }
}

#[test]
fn worker_capacity_covers_concurrent_attempts_and_is_reusable_after_release() {
    let mut allocated = ResourceAllocation::default();
    let first = request(60);
    let second = request(50);
    reserve_attempt_capacity(&inventory(), &mut allocated, &first, &Default::default()).unwrap();
    let reserved = allocated.clone();
    assert_eq!(
        reserve_attempt_capacity(&inventory(), &mut allocated, &second, &Default::default()),
        Err("attempt resources exceed the worker's available inventory")
    );
    assert_eq!(allocated, reserved);

    release(&mut allocated, &first, &Default::default());
    reserve_attempt_capacity(&inventory(), &mut allocated, &second, &Default::default()).unwrap();
    assert_eq!(allocated.memory_bytes, 50);
    assert_eq!(allocated.cpu_millicores, 1_000);
}

#[test]
fn overflowing_reservation_does_not_partially_change_worker_capacity() {
    let mut allocated = ResourceAllocation {
        memory_bytes: u64::MAX - 1,
        ..ResourceAllocation::default()
    };
    let inventory = ResourceInventory {
        memory_bytes: u64::MAX,
        ..inventory()
    };
    let before = allocated.clone();
    assert_eq!(
        reserve_attempt_capacity(&inventory, &mut allocated, &request(2), &Default::default()),
        Err("attempt resource reservation overflowed")
    );
    assert_eq!(allocated, before);
}

#[test]
fn invalid_attempt_request_cannot_reserve_worker_capacity() {
    let mut allocated = ResourceAllocation::default();
    let mut invalid = request(50);
    invalid.cpu_millicores = 0;
    assert_eq!(
        reserve_attempt_capacity(&inventory(), &mut allocated, &invalid, &Default::default()),
        Err("attempt has an invalid resource request")
    );
    assert_eq!(allocated, ResourceAllocation::default());
}

#[tokio::test(flavor = "current_thread")]
async fn completed_attempt_releases_its_worker_reservation_once() {
    tokio::task::LocalSet::new()
        .run_until(completed_attempt_releases_its_worker_reservation_once_inner())
        .await;
}

async fn completed_attempt_releases_its_worker_reservation_once_inner() {
    let attempt_id = runmat_execution::identity::AttemptId::derive(&[b"reservation-lifecycle"]);
    let resources = request(60);
    let mut state = WorkerState {
        bundle: None,
        materialized_project: None,
        bundle_digest: None,
        attempts: Default::default(),
        retirement_receipts: std::collections::VecDeque::with_capacity(
            super::MAX_RETIREMENT_RECEIPTS,
        ),
        allocated: ResourceAllocation::default(),
        cancelled_attempts: Default::default(),
        values: Default::default(),
        objects: super::super::object_transfer::RemoteObjectStore::new().unwrap(),
        draining: false,
        bundle_cache: None,
        graph_staging: Default::default(),
        graph_stage_download_in_flight: Default::default(),
        graph_programs: Default::default(),
        graph_cache: Default::default(),
    };
    reserve_attempt_capacity(
        &inventory(),
        &mut state.allocated,
        &resources,
        &Default::default(),
    )
    .unwrap();
    let (_finished_sender, finished) = tokio::sync::watch::channel(false);
    let (fea_control, _fea_calls) = super::super::fea_control::WorkerControl::new(
        runmat_execution::ProgramExecutionAssignment {
            scope_id: runmat_execution::ExecutionScopeId::derive(&[b"reservation-scope"]),
            pool_id: runmat_execution::PoolId::derive(&[b"reservation-pool"]),
            task_id: runmat_execution::TaskId::derive(&[b"reservation-task"]),
            attempt_id,
            worker_id: runmat_execution::identity::WorkerId::derive(&[b"reservation-worker"]),
            backend: runmat_execution::PoolBackend::Remote,
            resources: Default::default(),
        },
        u64::from(resources.cpu_millicores),
        runmat_execution::cancellation::CancellationSource::new().token(),
    );
    state.attempts.insert(
        attempt_id,
        ActiveAttempt {
            assignment: Default::default(),
            task: tokio::spawn(async {}),
            reply: super::super::attempt_reply::AttemptReply::spawn(
                std::sync::Arc::new(SilentRoute),
                std::sync::Arc::new(tokio::sync::Mutex::new(
                    runmat_execution_transport_native::frame::EncryptedFrameSession::new(
                        "reservation-run",
                        [3; 16],
                        "worker-to-driver",
                        1,
                        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy(
                            [5; 32],
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                )),
                runmat_execution_transport_native::frame::FrameLimits::default(),
                "reservation-correlation".into(),
                None,
            ),
            resources,
            cancellation: Default::default(),
            collective: None,
            fea_control,
            cooperative: true,
            finished,
            request_digest: runmat_execution::Digest::sha256(b"reservation-lifecycle-request"),
        },
    );

    assert!(state.finish_attempt(attempt_id, true, None).is_some());
    // Completion is not retirement ACK: retain the physical grant and its
    // exact receipt until the driver has accepted the certificate.
    assert_eq!(state.allocated.memory_bytes, 60);
    let claim = state
        .retirement_receipts
        .front_mut()
        .unwrap()
        .claim
        .take()
        .unwrap();
    release(&mut state.allocated, &claim.0, &claim.1);
    assert!(state
        .retirement_receipts
        .front_mut()
        .unwrap()
        .claim
        .take()
        .is_none());
    assert!(state.finish_attempt(attempt_id, true, None).is_none());
    assert_eq!(state.allocated, ResourceAllocation::default());
}
