use crate::supervisor::{tests::test_config, LocalSupervisor};

#[test]
fn journal_borrows_original_inventory_with_exact_cold_wire() {
    use super::*;
    let driver = Driver::new(DriverConfig::default(), 9).unwrap();
    let config_identity = Digest::sha256(b"original-profile");
    let root_scope = ExecutionScopeId::derive(&[b"original-authority"]);
    let pool_id = PoolId::derive(&[b"original-pool"]);
    let contexts = BTreeMap::from([(Digest::sha256(b"context"), root_scope)]);
    let routes = BTreeMap::from([(
        TaskId::derive(&[b"original-task"]),
        JobId::derive(&[b"original-job"]),
    )]);
    let borrowed = Journal {
        schema_version: 1,
        config_identity,
        root_scope,
        pool_id,
        contexts: &contexts,
        routes: &routes,
        checkpoint: DriverCheckpoint::new(1, driver.snapshot_ref()),
    };
    let owned = Journal {
        schema_version: 1,
        config_identity,
        root_scope,
        pool_id,
        contexts: contexts.clone(),
        routes: routes.clone(),
        checkpoint: DriverCheckpoint::new(1, driver.snapshot()),
    };
    let bytes = serde_json::to_vec(&borrowed).unwrap();
    assert_eq!(bytes, serde_json::to_vec(&owned).unwrap());
    let cold: Journal = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cold.contexts, contexts);
    assert_eq!(cold.routes, routes);
    assert_eq!(cold.checkpoint, owned.checkpoint);
}

#[tokio::test]
async fn unreadable_journal_preserves_original_authority_instead_of_bootstrapping() {
    use std::os::unix::fs::symlink;
    let temp = tempfile::tempdir().unwrap();
    let config = test_config(&temp);
    let supervisor = LocalSupervisor::open(config.clone()).unwrap();
    let original = supervisor
        .execution_context(Some("retained-authority"))
        .await
        .unwrap();
    drop(supervisor);
    let journal = config.paths.root.join("scheduler.json");
    let saved = config.paths.root.join("original-scheduler.json");
    let bytes = std::fs::read(&journal).unwrap();
    std::fs::rename(&journal, &saved).unwrap();
    // exists() hides ELOOP as false. Read errors must be distinguished from
    // absent authority, before constructing any fresh recovery state.
    symlink("scheduler.json", &journal).unwrap();
    assert!(LocalSupervisor::open(config.clone()).is_err());
    assert_eq!(
        std::fs::read_link(&journal).unwrap(),
        std::path::Path::new("scheduler.json")
    );
    assert_eq!(std::fs::read(&saved).unwrap(), bytes);
    std::fs::remove_file(&journal).unwrap();
    std::fs::rename(saved, journal).unwrap();
    let reopened = LocalSupervisor::open(config).unwrap();
    assert_eq!(
        reopened
            .execution_context(Some("retained-authority"))
            .await
            .unwrap(),
        original
    );
}
