use std::collections::BTreeSet;

use runmat_execution::resource::{ResourceInventory, ResourceRequest};
use runmat_execution_runner::scheduler::{release, ResourceAllocation};

use super::{reserve_attempt_capacity, ActiveAttempt, WorkerState};

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
    reserve_attempt_capacity(&inventory(), &mut allocated, &first).unwrap();
    let reserved = allocated.clone();
    assert_eq!(
        reserve_attempt_capacity(&inventory(), &mut allocated, &second),
        Err("attempt resources exceed the worker's available inventory")
    );
    assert_eq!(allocated, reserved);

    release(&mut allocated, &first);
    reserve_attempt_capacity(&inventory(), &mut allocated, &second).unwrap();
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
        reserve_attempt_capacity(&inventory, &mut allocated, &request(2)),
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
        reserve_attempt_capacity(&inventory(), &mut allocated, &invalid),
        Err("attempt has an invalid resource request")
    );
    assert_eq!(allocated, ResourceAllocation::default());
}

#[tokio::test]
async fn completed_attempt_releases_its_worker_reservation_once() {
    let attempt_id = runmat_execution::identity::AttemptId::derive(&[b"reservation-lifecycle"]);
    let resources = request(60);
    let mut state = WorkerState {
        bundle: None,
        materialized_project: None,
        bundle_digest: None,
        attempts: Default::default(),
        allocated: ResourceAllocation::default(),
        values: Default::default(),
        objects: super::super::object_transfer::RemoteObjectStore::new().unwrap(),
        draining: false,
        bundle_cache: None,
    };
    reserve_attempt_capacity(&inventory(), &mut state.allocated, &resources).unwrap();
    let (_finished_sender, finished) = tokio::sync::watch::channel(false);
    state.attempts.insert(
        attempt_id,
        ActiveAttempt {
            task: tokio::spawn(async {}),
            resources,
            cancellation: Default::default(),
            cooperative: true,
            finished,
        },
    );

    assert!(state.finish_attempt(attempt_id).is_some());
    assert_eq!(state.allocated, ResourceAllocation::default());
    assert!(state.finish_attempt(attempt_id).is_none());
    assert_eq!(state.allocated, ResourceAllocation::default());
}
