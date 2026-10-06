use runmat_value::{CellArray, HandleRef, Tensor, TransientValueError, Value};

use super::values::RootedWorkspaceEntry;
use super::*;

fn assigned(name: &str, value: Value) -> WorkspaceDelta {
    let mut delta = WorkspaceDelta::new();
    delta.assign(name, value).unwrap();
    delta
}

fn removed(name: &str) -> WorkspaceDelta {
    WorkspaceDelta::from_changes(vec![WorkspaceChange::remove(name)])
}

#[test]
fn ordered_delta_applies_clear_and_later_assignment_in_order() {
    let workspace = BaseWorkspaceOwner::new();
    workspace
        .commit(
            WorkspaceRevision::INITIAL,
            assigned("before", Value::Num(1.0)),
        )
        .unwrap();

    let base = workspace.revision().unwrap();
    let mut delta = WorkspaceDelta::new();
    delta.assign("discarded", Value::Num(2.0)).unwrap();
    delta.clear();
    delta.assign("after", Value::Num(3.0)).unwrap();
    let commit = workspace.commit(base, delta).unwrap();

    assert_eq!(commit.revision().get(), 2);
    assert_eq!(commit.change_count(), 2);
    let snapshot = workspace.snapshot().unwrap();
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot.get("after"), Some(&Value::Num(3.0)));
}

#[test]
fn projections_merge_unrelated_changes_and_later_same_name_write_wins() {
    let workspace = BaseWorkspaceOwner::new();
    let mut first = workspace.begin_projection().unwrap();
    let mut second = workspace.begin_projection().unwrap();

    first.assign("left", Value::Num(1.0)).unwrap();
    first.assign("shared", Value::Num(10.0)).unwrap();
    assert!(!first.publish().unwrap().merged_concurrent_changes());

    second.assign("right", Value::Num(2.0)).unwrap();
    second.assign("shared", Value::Num(20.0)).unwrap();
    let commit = second.publish().unwrap();
    assert!(commit.merged_concurrent_changes());
    assert_eq!(second.lookup("left"), Some(&Value::Num(1.0)));
    assert_eq!(second.lookup("right"), Some(&Value::Num(2.0)));
    assert_eq!(second.lookup("shared"), Some(&Value::Num(20.0)));
}

#[test]
fn dirty_projection_must_publish_before_reconciliation() {
    let workspace = BaseWorkspaceOwner::new();
    let mut projection = workspace.begin_projection().unwrap();
    projection.assign("local", Value::Num(1.0)).unwrap();
    workspace
        .commit(
            WorkspaceRevision::INITIAL,
            assigned("external", Value::Num(2.0)),
        )
        .unwrap();

    assert!(matches!(
        projection.reconcile(),
        Err(BaseWorkspaceError::PendingChanges)
    ));
    let publish = projection.publish().unwrap();
    assert!(publish.merged_concurrent_changes());
    assert_eq!(projection.lookup("external"), Some(&Value::Num(2.0)));
    assert_eq!(projection.lookup("local"), Some(&Value::Num(1.0)));
}

#[test]
fn clean_projection_reconciles_incrementally() {
    let workspace = BaseWorkspaceOwner::new();
    let mut projection = workspace.begin_projection().unwrap();
    workspace
        .commit(
            WorkspaceRevision::INITIAL,
            assigned("external", Value::Num(4.0)),
        )
        .unwrap();

    let reconciliation = projection.reconcile().unwrap();
    assert!(matches!(
        reconciliation,
        BaseWorkspaceReconciliation::Delta {
            from: WorkspaceRevision::INITIAL,
            to,
            ..
        } if to.get() == 1
    ));
    assert_eq!(projection.lookup("external"), Some(&Value::Num(4.0)));
}

#[test]
fn bounded_history_returns_a_rooted_reset_snapshot() {
    let workspace = BaseWorkspaceOwner::for_test(1);
    let mut projection = workspace.begin_projection().unwrap();
    workspace
        .commit(
            WorkspaceRevision::INITIAL,
            assigned("first", Value::Num(1.0)),
        )
        .unwrap();
    workspace
        .commit(
            WorkspaceRevision::from_raw(1),
            assigned("second", Value::Num(2.0)),
        )
        .unwrap();

    assert!(matches!(
        projection.reconcile().unwrap(),
        BaseWorkspaceReconciliation::Reset(_)
    ));
    assert_eq!(projection.lookup("first"), Some(&Value::Num(1.0)));
    assert_eq!(projection.lookup("second"), Some(&Value::Num(2.0)));
}

#[test]
fn revision_exhaustion_is_transactional() {
    let workspace = BaseWorkspaceOwner::new();
    let exhausted = WorkspaceRevision::from_raw(u64::MAX);
    workspace.set_revision_for_test(exhausted);

    assert!(matches!(
        workspace.commit(exhausted, assigned("x", Value::Num(1.0))),
        Err(BaseWorkspaceError::RevisionExhausted)
    ));
    let snapshot = workspace.snapshot().unwrap();
    assert_eq!(snapshot.revision(), exhausted);
    assert!(snapshot.is_empty());
}

#[test]
fn transient_values_are_rejected_before_state_changes() {
    let workspace = BaseWorkspaceOwner::new();
    let transient =
        Value::Cell(CellArray::new(vec![Value::OutputList(vec![Value::Num(1.0)])], 1, 1).unwrap());
    let mut delta = WorkspaceDelta::new();
    assert!(matches!(
        delta.assign("bad", transient),
        Err(BaseWorkspaceError::TransientValue(
            TransientValueError::LegacyOutputList
        ))
    ));
    assert!(workspace.snapshot().unwrap().is_empty());
}

#[test]
fn unchanged_tensor_entries_share_the_same_immutable_value() {
    let tensor = Value::Tensor(Tensor::new(vec![1.0; 1024], vec![32, 32]).unwrap());
    let entry = RootedWorkspaceEntry::new(tensor).unwrap();
    let snapshot_entry = entry.clone();
    assert!(entry.shares_value_with(&snapshot_entry));
}

#[test]
fn close_revokes_owner_and_projection_access() {
    let workspace = BaseWorkspaceOwner::new();
    let mut projection = workspace.begin_projection().unwrap();
    projection.assign("pending", Value::Num(1.0)).unwrap();
    assert!(workspace.close().unwrap());
    assert!(!workspace.close().unwrap());
    assert!(workspace.is_closed().unwrap());
    assert!(matches!(
        workspace.snapshot(),
        Err(BaseWorkspaceError::Closed)
    ));
    assert!(matches!(
        projection.publish(),
        Err(BaseWorkspaceError::Closed)
    ));
}

#[cfg(not(target_arch = "wasm32"))]
fn rooted_handle(label: &str) -> (runmat_gc::ExplicitRoot, runmat_gc::GcHandle, Value) {
    let initial = runmat_gc::gc_allocate_rooted(Value::String(label.into())).unwrap();
    let handle = initial.handle();
    let value = Value::HandleObject(HandleRef {
        class_name: "WorkspaceHandle".into(),
        target: handle,
        valid: true,
    });
    (initial, handle, value)
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn canonical_remove_releases_deleted_capture_immediately() {
    runmat_gc::gc_test_context(|| {
        let (initial, handle, value) = rooted_handle("canonical");
        let workspace = BaseWorkspaceOwner::new();
        drop(workspace.apply(assigned("held", value)).unwrap());
        initial.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());

        drop(workspace.apply(removed("held")).unwrap());
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn commit_receipt_does_not_retain_an_intermediate_removed_assignment() {
    runmat_gc::gc_test_context(|| {
        let (initial, handle, value) = rooted_handle("intermediate");
        let workspace = BaseWorkspaceOwner::new();
        let mut delta = WorkspaceDelta::new();
        delta.assign("temporary", value).unwrap();
        delta.remove("temporary");
        let receipt = workspace.apply(delta).unwrap();
        initial.unroot().unwrap();

        assert_eq!(receipt.change_count(), 1);
        assert_eq!(
            receipt.changes().changes()[0].kind(),
            WorkspaceChangeKind::Remove
        );
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn projection_clear_releases_its_capture_without_whole_workspace_reroot() {
    runmat_gc::gc_test_context(|| {
        let (initial, handle, value) = rooted_handle("projection");
        let workspace = BaseWorkspaceOwner::new();
        drop(workspace.apply(assigned("held", value)).unwrap());
        let mut projection = workspace.begin_projection().unwrap();
        initial.unroot().unwrap();
        drop(workspace.apply(removed("held")).unwrap());

        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        projection.clear();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn outstanding_snapshot_retains_only_its_immutable_entry_lifetime() {
    runmat_gc::gc_test_context(|| {
        let (initial, handle, value) = rooted_handle("snapshot");
        let workspace = BaseWorkspaceOwner::new();
        drop(workspace.apply(assigned("held", value)).unwrap());
        let snapshot = workspace.snapshot().unwrap();
        initial.unroot().unwrap();
        drop(workspace.apply(removed("held")).unwrap());

        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        drop(snapshot);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
