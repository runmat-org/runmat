use crate::context::RuntimeWorkspaceService;
use crate::workspace::GlobalBindingTarget;
#[cfg(not(target_arch = "wasm32"))]
use runmat_value::HandleRef;
use runmat_value::{CellArray, Tensor, TransientValueError, Value};
#[cfg(not(target_arch = "wasm32"))]
use std::rc::Rc;

use super::values::RootedWorkspaceEntry;
use super::*;

mod foreign_retirement;

fn assigned(name: &str, value: Value) -> WorkspaceDelta {
    let mut delta = WorkspaceDelta::new();
    delta.assign(name, value).unwrap();
    delta
}

#[test]
fn exact_detach_retires_only_selected_old_binding_before_reentry() {
    let workspace = BaseWorkspaceOwner::new();
    workspace.apply(assigned("x", Value::Num(1.0))).unwrap();
    workspace.apply(assigned("y", Value::Num(7.0))).unwrap();
    let retired = workspace
        .detach_selected(&crate::workspace::WorkspaceDetachSelection::Names(vec![
            "x".into(),
        ]))
        .unwrap();
    assert_eq!(retired.entries(), &[("x".into(), Value::Num(1.0))]);
    assert_eq!(workspace.lookup("x").unwrap(), None);
    workspace.apply(assigned("x", Value::Num(2.0))).unwrap();
    assert_eq!(workspace.lookup("x").unwrap(), Some(Value::Num(2.0)));
    assert_eq!(workspace.lookup("y").unwrap(), Some(Value::Num(7.0)));
}

#[test]
fn exact_except_selection_preserves_names_created_after_retirement() {
    let workspace = BaseWorkspaceOwner::new();
    for (name, value) in [("a", 1.0), ("b", 2.0), ("c", 3.0)] {
        workspace.apply(assigned(name, Value::Num(value))).unwrap();
    }
    let retired = workspace
        .detach_selected(&crate::workspace::WorkspaceDetachSelection::Except(vec![
            "b".into(),
        ]))
        .unwrap();
    assert_eq!(retired.entries().len(), 2);
    workspace.apply(assigned("new", Value::Num(4.0))).unwrap();
    let snapshot = workspace.snapshot().unwrap();
    assert_eq!(snapshot.get("b"), Some(&Value::Num(2.0)));
    assert_eq!(snapshot.get("new"), Some(&Value::Num(4.0)));
    assert_eq!(snapshot.get("a"), None);
    assert_eq!(snapshot.get("c"), None);
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
fn global_binding_and_shadow_removal_share_one_canonical_revision() {
    let workspace = BaseWorkspaceOwner::new();
    workspace.apply(assigned("g", Value::Num(1.0))).unwrap();
    let revision = workspace.revision().unwrap();
    let mut delta = WorkspaceDelta::new();
    delta.bind_global("g");
    let commit = workspace.commit(revision, delta).unwrap();

    assert!(matches!(
        commit.changes().changes()[0].data(),
        super::delta::WorkspaceChangeData::BindGlobal {
            target: GlobalBindingTarget::Named(name),
            ..
        } if name.as_str() == "g"
    ));

    let snapshot = workspace.snapshot().unwrap();
    assert!(snapshot.is_global("g"));
    assert_eq!(snapshot.get("g"), None);

    let mut clear_and_bind = WorkspaceDelta::new();
    clear_and_bind.clear();
    clear_and_bind.bind_global_target("g", GlobalBindingTarget::LegacySlot(9));
    let commit = workspace.apply(clear_and_bind).unwrap();
    assert_eq!(commit.change_count(), 2);
    assert!(matches!(
        commit.changes().changes()[1].data(),
        super::delta::WorkspaceChangeData::BindGlobal {
            target: GlobalBindingTarget::LegacySlot(9),
            ..
        }
    ));
    assert_eq!(
        workspace.snapshot().unwrap().global_target("g"),
        Some(&GlobalBindingTarget::LegacySlot(9))
    );
}

#[test]
fn stale_assignment_cannot_resurface_after_global_binding_is_cleared() {
    let workspace = BaseWorkspaceOwner::new();
    workspace.apply(assigned("g", Value::Num(1.0))).unwrap();
    let mut stale = workspace.begin_projection().unwrap();

    let mut bind = WorkspaceDelta::new();
    bind.bind_global("g");
    workspace.apply(bind).unwrap();
    stale.assign("g", Value::Num(99.0)).unwrap();
    stale.publish().unwrap();

    let mut unbind = WorkspaceDelta::new();
    unbind.unbind_global("g");
    workspace.apply(unbind).unwrap();
    let snapshot = workspace.snapshot().unwrap();
    assert!(!snapshot.is_global("g"));
    assert_eq!(snapshot.get("g"), None);
}

#[test]
fn explicit_local_assignment_atomically_reclassifies_a_global_name() {
    let workspace = BaseWorkspaceOwner::new();
    let service = BaseWorkspaceService::new(workspace.clone());
    let mut bind = WorkspaceDelta::new();
    bind.bind_global("g");
    workspace.apply(bind).unwrap();

    service.assign_base_local("g", Value::Num(7.0)).unwrap();
    let snapshot = workspace.snapshot().unwrap();
    assert!(!snapshot.is_global("g"));
    assert_eq!(snapshot.get("g"), Some(&Value::Num(7.0)));
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
fn stable_capture_and_older_projection_publish_one_authoritative_binding() {
    let context = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let owner = BaseWorkspaceOwner::new();
    owner.apply(assigned("x", Value::Num(1.0))).unwrap();
    owner.apply(assigned("unrelated", Value::Num(7.0))).unwrap();
    let mut projection = owner.begin_projection().unwrap();
    let capture = owner.capture_stable_with_context(&context).unwrap();
    assert_eq!(capture.get("x"), Some(&Value::Num(1.0)));

    projection.assign("x", Value::Num(2.0)).unwrap();
    projection.publish().unwrap();
    let snapshot = owner.snapshot().unwrap();
    assert_eq!(snapshot.len(), 2);
    assert_eq!(snapshot.iter().filter(|(name, _)| *name == "x").count(), 1);
    assert_eq!(snapshot.get("x"), Some(&Value::Num(2.0)));
    assert_eq!(snapshot.get("unrelated"), Some(&Value::Num(7.0)));
}

#[test]
fn first_projection_mutation_captures_complete_canonical_prefix_before_publication() {
    let owner = BaseWorkspaceOwner::new();
    owner.apply(assigned("x", Value::Num(1.0))).unwrap();
    owner.apply(assigned("y", Value::Num(7.0))).unwrap();
    let mut projection = owner.begin_projection().unwrap();
    projection.assign("x", Value::Num(2.0)).unwrap();
    owner.apply(assigned("z", Value::Num(9.0))).unwrap();

    projection.publish().unwrap();
    let snapshot = owner.snapshot().unwrap();
    assert_eq!(snapshot.len(), 3);
    for (name, value) in [("x", 2.0), ("y", 7.0), ("z", 9.0)] {
        assert_eq!(
            snapshot.iter().filter(|(found, _)| *found == name).count(),
            1
        );
        assert_eq!(snapshot.get(name), Some(&Value::Num(value)));
        assert_eq!(projection.lookup(name), Some(&Value::Num(value)));
    }
    assert_eq!(projection.iter().count(), 3);
}

#[test]
fn clean_projection_adopts_same_revision_physical_capture_before_global_change() {
    let context = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let owner = BaseWorkspaceOwner::new();
    owner.apply(assigned("g", Value::Num(1.0))).unwrap();
    let mut projection = owner.begin_projection().unwrap();
    owner.capture_stable_with_context(&context).unwrap();

    // Direct application can already install a managed canonical generation;
    // the physical reconciliation variant is not the public namespace result.
    projection.reconcile().unwrap();
    assert_eq!(
        projection.iter().filter(|(name, _)| *name == "g").count(),
        1
    );
    projection.bind_global("g").unwrap();
    projection.publish().unwrap();
    let snapshot = owner.snapshot().unwrap();
    assert!(snapshot.is_global("g"));
    assert_eq!(snapshot.get("g"), None);
    assert_eq!(snapshot.len(), 0);

    let mut later = owner.begin_projection().unwrap();
    later.unbind_global("g").unwrap();
    later.publish().unwrap();
    assert_eq!(owner.snapshot().unwrap().get("g"), None);
}

#[test]
fn captured_generation_reconciliation_keeps_unrelated_names_once_after_delta() {
    let context = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let owner = BaseWorkspaceOwner::new();
    owner.apply(assigned("x", Value::Num(1.0))).unwrap();
    owner.apply(assigned("y", Value::Num(2.0))).unwrap();
    let mut projection = owner.begin_projection().unwrap();
    owner.capture_stable_with_context(&context).unwrap();
    let mut writer = owner.begin_projection().unwrap();
    writer.assign("x", Value::Num(3.0)).unwrap();
    writer.publish().unwrap();

    projection.reconcile().unwrap();
    assert_eq!(projection.iter().count(), 2);
    assert_eq!(projection.lookup("x"), Some(&Value::Num(3.0)));
    assert_eq!(projection.lookup("y"), Some(&Value::Num(2.0)));
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
        Value::Cell(CellArray::new(vec![Value::output_list(vec![Value::Num(1.0)])], 1, 1).unwrap());
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
fn rooting_failure_keeps_its_typed_gc_source() {
    use std::error::Error;

    let error = base_workspace_error_to_runtime(BaseWorkspaceError::from(
        runmat_gc::GcError::RootRegistrationFailed("root table unavailable".into()),
    ));

    assert_eq!(error.identifier(), Some("RunMat:Workspace:Rooting"));
    let source = error.source().expect("typed GC source");
    assert!(matches!(
        source.downcast_ref::<runmat_gc::GcError>(),
        Some(runmat_gc::GcError::RootRegistrationFailed(message))
            if message == "root table unavailable"
    ));
}

#[test]
fn canonical_service_keeps_direct_and_nested_transient_value_sources() {
    use crate::context::RuntimeWorkspaceService;
    use std::error::Error;

    let service = BaseWorkspaceService::new(BaseWorkspaceOwner::new());
    let direct = Value::output_list(vec![Value::Num(1.0)]);
    let nested = Value::Cell(CellArray::new(vec![direct.clone()], 1, 1).unwrap());

    for (name, value) in [("direct", direct), ("nested", nested)] {
        let error = service.assign(name, value).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:TransientSequenceNotStorable")
        );
        let source = error.source().expect("typed transient-value source");
        assert!(matches!(
            source.downcast_ref::<TransientValueError>(),
            Some(TransientValueError::LegacyOutputList)
        ));
        assert_eq!(service.try_lookup(name).unwrap(), None);
    }
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

#[test]
fn close_revokes_effect_authority_but_retains_the_local_projection_view() {
    let workspace = BaseWorkspaceOwner::new();
    workspace
        .apply(assigned("admitted", Value::Num(7.0)))
        .unwrap();
    let projection = workspace.begin_projection().unwrap();

    workspace.close().unwrap();

    assert_eq!(projection.lookup("admitted"), Some(&Value::Num(7.0)));
    assert_eq!(projection.iter().count(), 1);
    assert!(matches!(
        projection.ensure_effect_authority(),
        Err(BaseWorkspaceError::Closed)
    ));
}

#[test]
fn canonical_service_keeps_base_globals_and_persistents_as_separate_authorities() {
    use crate::context::{RuntimeContext, RuntimeServicePorts, RuntimeWorkspaceService};
    use crate::execution::RuntimeExecutionService;
    use std::rc::Rc;

    let owner = BaseWorkspaceOwner::new();
    owner.apply(assigned("base", Value::Num(1.0))).unwrap();
    let service = Rc::new(BaseWorkspaceService::new(owner));
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_workspace(service.clone()));
    let _scope = crate::context::RuntimeContextGuard::enter(runtime);
    crate::workspace::session::store_global_named("global", Value::Num(2.0))
        .expect("install the independent global binding");
    crate::workspace::session::store_persistent_named("f", "persistent", Value::Num(3.0));

    assert_eq!(service.try_lookup("base").unwrap(), Some(Value::Num(1.0)));
    assert_eq!(service.try_lookup("global").unwrap(), None);
    assert_eq!(service.try_global_names().unwrap(), vec!["global"]);
    assert_eq!(service.try_snapshot().unwrap().len(), 1);
}

#[test]
fn checked_facade_does_not_fall_back_when_canonical_owner_is_closed() {
    use crate::context::{RuntimeContext, RuntimeServicePorts};
    use crate::execution::RuntimeExecutionService;
    use std::rc::Rc;

    fn legacy_lookup(name: &str) -> Option<Value> {
        (name == "shadow").then_some(Value::Num(99.0))
    }
    fn empty_snapshot() -> Vec<(String, Value)> {
        Vec::new()
    }
    fn empty_globals() -> Vec<String> {
        Vec::new()
    }

    let _workspace_guard = crate::workspace::test_guard();
    let owner = BaseWorkspaceOwner::new();
    let service = Rc::new(BaseWorkspaceService::new(owner.clone()));
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_workspace(service));
    let _scope = crate::context::RuntimeContextGuard::enter(runtime);
    crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
        lookup: legacy_lookup,
        snapshot: empty_snapshot,
        globals: empty_globals,
        assign: None,
        clear: None,
        remove: None,
    });
    owner.close().unwrap();

    let error = crate::workspace::try_lookup("shadow").unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
    assert_eq!(
        crate::workspace::try_snapshot()
            .expect_err("closed owner must not become an empty snapshot")
            .identifier(),
        Some("RunMat:Workspace:Closed")
    );
    assert_eq!(
        crate::workspace::try_assign("x", Value::Num(1.0))
            .expect_err("closed owner must reject mutations")
            .identifier(),
        Some("RunMat:Workspace:Closed")
    );
}

#[test]
fn checked_facade_preserves_canonical_borrow_conflicts() {
    use crate::context::{RuntimeContext, RuntimeServicePorts};
    use crate::execution::RuntimeExecutionService;
    use std::rc::Rc;

    let owner = BaseWorkspaceOwner::new();
    let service = Rc::new(BaseWorkspaceService::new(owner.clone()));
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_workspace(service));
    let _scope = crate::context::RuntimeContextGuard::enter(runtime);
    let _exclusive = owner.state.borrow_mut();

    let error = crate::workspace::try_snapshot().unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Workspace:Busy"));
}

#[test]
fn caller_workspace_service_does_not_gain_base_workspace_authority() {
    use crate::context::{RuntimeContext, RuntimeServicePorts};
    use crate::execution::RuntimeExecutionService;
    use std::rc::Rc;

    #[derive(Default)]
    struct CallerWorkspace;

    impl crate::context::RuntimeWorkspaceService for CallerWorkspace {
        fn lookup(&self, _name: &str) -> Option<Value> {
            None
        }

        fn snapshot(&self) -> Vec<(String, Value)> {
            Vec::new()
        }

        fn global_names(&self) -> Vec<String> {
            Vec::new()
        }

        fn assign(&self, _name: &str, _value: Value) -> Result<(), crate::RuntimeError> {
            Ok(())
        }

        fn clear(&self) -> Result<(), crate::RuntimeError> {
            Ok(())
        }

        fn remove(&self, _name: &str) -> Result<(), crate::RuntimeError> {
            Ok(())
        }
    }

    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default().with_workspace(Rc::new(CallerWorkspace)),
    );
    let _scope = crate::context::RuntimeContextGuard::enter(runtime);
    let error = crate::workspace::base_workspace_owner().unwrap_err();
    assert_eq!(
        error.identifier(),
        Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
    );
    let detach =
        crate::workspace::try_detach_selected(&crate::workspace::WorkspaceDetachSelection::All)
            .unwrap_err();
    assert_eq!(
        detach.identifier(),
        Some("RunMat:Workspace:ExactDetachUnavailable")
    );
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
        // An unrelated later write must not keep the deleted binding alive in
        // an implementation-only overlay predecessor or captured base.
        drop(
            workspace
                .apply(assigned("unrelated", Value::Num(7.0)))
                .unwrap(),
        );
        initial.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());

        drop(workspace.apply(removed("held")).unwrap());
        assert_eq!(
            workspace.lookup("unrelated").unwrap(),
            Some(Value::Num(7.0))
        );
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn stable_capture_keeps_legacy_handle_root_until_last_generation_retires() {
    runmat_gc::gc_test_context(|| {
        let context = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let (initial, handle, value) = rooted_handle("captured");
        let workspace = BaseWorkspaceOwner::new();
        drop(workspace.apply(assigned("held", value)).unwrap());
        let capture = workspace.capture_stable_with_context(&context).unwrap();
        assert!(capture.get("held").is_some());
        initial.unroot().unwrap();
        drop(workspace.apply(removed("held")).unwrap());
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        drop(capture);
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
fn unpublished_ordered_delta_retains_intermediate_values_until_dropped() {
    runmat_gc::gc_test_context(|| {
        let (initial, handle, value) = rooted_handle("pending");
        let workspace = BaseWorkspaceOwner::new();
        let mut projection = workspace.begin_projection().unwrap();
        projection.assign("temporary", value).unwrap();
        projection.remove("temporary").unwrap();
        initial.unroot().unwrap();

        assert!(projection.lookup("temporary").is_none());
        assert!(projection.has_pending_changes());
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());

        drop(projection);
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
        projection.clear().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn projection_service_publication_effect_releases_replaced_and_cleared_captures() {
    runmat_gc::gc_test_context(|| {
        use crate::context::RuntimeWorkspaceService;
        let workspace = BaseWorkspaceOwner::new();
        let service = BaseWorkspaceProjectionService::new(workspace.begin_projection().unwrap());

        let (first_root, first_handle, first_value) = rooted_handle("published-first");
        service.assign("held", first_value).unwrap();
        service.publish().unwrap();
        first_root.unroot().unwrap();

        let (second_root, second_handle, second_value) = rooted_handle("published-second");
        service.assign("held", second_value).unwrap();
        service.publish().unwrap();
        second_root.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&first_handle).is_err(),
            "coalescing a later publication releases the replaced receipt root"
        );
        assert!(runmat_gc::gc_clone_value(&second_handle).is_ok());

        service.clear().unwrap();
        service.publish().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&second_handle).is_err(),
            "a published clear releases the coalesced receipt root"
        );

        // Interleaved executor aliases must retire before publication, while
        // a real captured generation retains its original rooted values.
        let context = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let owner = BaseWorkspaceOwner::new();
        let service = BaseWorkspaceProjectionService::new_for_context(
            owner.begin_projection().unwrap(),
            &context,
        );
        let (initial, handle, value) = rooted_handle("interleaved-aliases");
        service
            .assign_executor_slot_ref(&context, 0, "a", &value)
            .unwrap();
        service
            .assign_executor_slot_ref(&context, 1, "b", &value)
            .unwrap();
        service
            .assign_executor_slot_ref(&context, 2, "noise", &Value::Num(1.0))
            .unwrap();
        initial.unroot().unwrap();
        let captured = service
            .capture_workspace_generation_with_context(&context)
            .unwrap()
            .unwrap();
        service.remove_with_context(&context, "a").unwrap();
        service
            .assign_executor_slot_ref(&context, 2, "noise", &Value::Num(2.0))
            .unwrap();
        service.remove_with_context(&context, "b").unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_ok(),
            "real snapshot remains rooted"
        );
        drop(captured);
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_err(),
            "hidden executor stores retire before publication"
        );
        service.publish_with_context(&context).unwrap();
        assert!(owner.lookup("a").unwrap().is_none());
        assert!(owner.lookup("b").unwrap().is_none());
        assert_eq!(owner.lookup("noise").unwrap(), Some(Value::Num(2.0)));
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn outstanding_snapshot_retains_only_its_immutable_entry_lifetime() {
    runmat_gc::gc_test_context(|| {
        let (initial, handle, value) = rooted_handle("snapshot");
        let workspace = BaseWorkspaceOwner::new();
        drop(workspace.apply(assigned("held", value)).unwrap());
        drop(
            workspace
                .apply(assigned("unrelated", Value::Num(7.0)))
                .unwrap(),
        );
        let snapshot = workspace.snapshot().unwrap();
        initial.unroot().unwrap();
        drop(workspace.apply(removed("held")).unwrap());
        assert!(snapshot.get("held").is_some());
        assert_eq!(workspace.lookup("held").unwrap(), None);
        assert_eq!(
            workspace.lookup("unrelated").unwrap(),
            Some(Value::Num(7.0))
        );

        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        drop(snapshot);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
