use super::*;
use runmat_execution_runner::{Driver, DriverCommand, DriverConfig};
use runmat_filesystem::PublicationDurability;
use std::fs;

#[test]
fn original_snapshot_is_durably_replaced_and_invalid_state_never_overwrites_it() {
    let directory = tempfile::tempdir().unwrap();
    let store = CheckpointStore::new(directory.path().join("checkpoints")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&store.root).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }
    let mut driver = Driver::new(DriverConfig::default(), 1).unwrap();
    driver
        .handle(DriverCommand::Tick { now_millis: 7 })
        .unwrap();
    let first = driver.snapshot();
    assert_eq!(
        store.write(&first).unwrap().durability,
        PublicationDurability::Durable
    );
    driver
        .handle(DriverCommand::Tick { now_millis: 17 })
        .unwrap();
    let original = driver.snapshot();
    assert_eq!(
        store.write(&original).unwrap().durability,
        PublicationDurability::Durable
    );
    let bytes = fs::read(store.root.join("driver.json")).unwrap();
    let cold: DriverSnapshot = serde_json::from_slice(&bytes).unwrap();
    cold.validate().unwrap();
    assert_eq!(cold, original);
    let mut invalid = original;
    invalid.schema_version = 0;
    assert!(store.write(&invalid).is_err());
    assert_eq!(fs::read(store.root.join("driver.json")).unwrap(), bytes);
    assert_eq!(fs::read_dir(&store.root).unwrap().count(), 1);
}

#[test]
fn broken_namespace_is_an_unpublished_error_instead_of_an_execution_verdict() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("checkpoints");
    let store = CheckpointStore::new(root.clone()).unwrap();
    fs::remove_dir(&root).unwrap();
    fs::write(&root, b"injected namespace failure").unwrap();
    let driver = Driver::new(DriverConfig::default(), 1).unwrap();
    let error = store.write(&driver.snapshot()).unwrap_err();
    assert!(error.to_string().contains("did not publish"));
    assert_eq!(fs::read(root).unwrap(), b"injected namespace failure");
}

#[test]
fn original_driver_snapshot_publication_is_safe_inside_an_entered_executor() {
    let directory = tempfile::tempdir().unwrap();
    let store = CheckpointStore::new(directory.path().join("checkpoints")).unwrap();
    let driver = Driver::new(DriverConfig::default(), 1).unwrap();
    let original = driver.snapshot();
    let outcome = futures::executor::block_on(async { store.write(&original) }).unwrap();
    assert_eq!(outcome.durability, PublicationDurability::Durable);
    let cold: DriverSnapshot =
        serde_json::from_slice(&fs::read(store.root.join("driver.json")).unwrap()).unwrap();
    cold.validate().unwrap();
    assert_eq!(cold, original);
}
