use super::*;
use crate::driver::{DriverCommand, DriverConfig};
use runmat_execution::ExecutionScopeId;

#[test]
fn borrowed_checkpoint_preserves_exact_owned_wire_and_cold_authority() {
    let mut driver = Driver::new(DriverConfig::default(), 7).unwrap();
    let scope = ExecutionScopeId::derive(&[b"borrowed-checkpoint"]);
    driver
        .handle(DriverCommand::RegisterScope {
            scope_id: scope,
            parent: None,
        })
        .unwrap();
    let borrowed = DriverCheckpoint::new(11, driver.snapshot_ref());
    borrowed.validate().unwrap();
    assert!(std::ptr::eq(borrowed.snapshot, driver.snapshot_ref()));
    let bytes = serde_json::to_vec(&borrowed).unwrap();
    let owned = DriverCheckpoint::new(11, driver.snapshot());
    assert_eq!(bytes, serde_json::to_vec(&owned).unwrap());
    let cold: DriverCheckpoint = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cold, owned);
    let continued = cold.continue_under_exclusive_ownership().unwrap();
    assert_eq!(continued.snapshot_ref(), driver.snapshot_ref());
    let replacement: DriverCheckpoint = serde_json::from_slice(&bytes).unwrap();
    let replaced = replacement.restore_at_fence(8).unwrap();
    assert_eq!(replaced.snapshot_ref().driver_fence, 8);
    assert!(replacement_fence_rejects(&bytes, 7));
}

fn replacement_fence_rejects(bytes: &[u8], fence: u64) -> bool {
    serde_json::from_slice::<DriverCheckpoint>(bytes)
        .unwrap()
        .restore_at_fence(fence)
        .is_err()
}

#[test]
fn borrowed_checkpoint_cannot_bypass_original_history_validation() {
    let driver = Driver::new(DriverConfig::default(), 1).unwrap();
    let mut original = driver.snapshot();
    original.next_event_sequence += 1;
    assert!(DriverCheckpoint::new(1, &original).validate().is_err());
    let mut foreign = DriverCheckpoint::new(1, driver.snapshot_ref());
    foreign.schema_version += 1;
    assert!(foreign.validate().is_err());
}
