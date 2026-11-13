mod common;
use common::*;
use runmat_execution::{task::RetryPolicy, EnvironmentSnapshot};
use runmat_execution_runner::DriverCommand;

#[test]
fn original_environment_reservation_is_checked_before_live_and_cold_admission() {
    let mut fixture = fixture(1, 1);
    let mut original = task(
        "original-environment",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    original.request.environment =
        EnvironmentSnapshot::new(vec![("RUNMAT_ORIGINAL".into(), "x".repeat(2048))]).unwrap();
    let required = original
        .request
        .environment
        .storage_profile()
        .unwrap()
        .minimum_memory_bytes()
        .unwrap();
    original.request.resources.memory_bytes = required - 1;
    assert!(original.retry_resource_envelope().is_err());
    assert!(fixture
        .driver
        .handle(DriverCommand::Submit(Box::new(original.clone())))
        .is_err());
    assert!(fixture.driver.snapshot().tasks.is_empty());
    original.request.resources.memory_bytes = required;
    original.retry_resource_envelope().unwrap();
    fixture
        .driver
        .handle(DriverCommand::Submit(Box::new(original.clone())))
        .unwrap();
    let bytes = serde_json::to_vec(&fixture.driver.snapshot()).unwrap();
    let mut cold: runmat_execution_runner::DriverSnapshot = serde_json::from_slice(&bytes).unwrap();
    cold.validate().unwrap();
    cold.tasks
        .get_mut(&original.request.id)
        .unwrap()
        .submission
        .request
        .resources
        .memory_bytes = required - 1;
    assert!(cold.validate().is_err());
}
