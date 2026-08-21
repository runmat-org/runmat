use futures::executor::block_on;
use runmat_package_cache::backend::conformance::{verify_backend_contract, MemoryBackend};
use runmat_package_cache::{BackendCommit, CommitOutcome};

#[test]
fn memory_backend_obeys_portable_contract() {
    block_on(verify_backend_contract(MemoryBackend::new)).unwrap();
}

#[test]
fn browser_wire_outcomes_are_stable_tagged_objects() {
    assert_eq!(
        serde_json::to_value(CommitOutcome::Committed(BackendCommit { revision: 7 })).unwrap(),
        serde_json::json!({ "outcome": "committed", "revision": 7 })
    );
    assert_eq!(
        serde_json::to_value(CommitOutcome::Conflict { actual_revision: 9 }).unwrap(),
        serde_json::json!({ "outcome": "conflict", "actual_revision": 9 })
    );
}
