use super::*;
use runmat_runtime::context::RuntimeWorkspaceService;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn source_store_and_effect_import_keep_distinct_keys_under_one_active_source() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    session.active_source_identity = Some(crate::abi::SourceIdentity::PathAndContentHash {
        path: "active.m".into(),
        hash: "revision".into(),
    });
    let projection = runmat_runtime::workspace::BaseWorkspaceProjectionService::new(
        session
            .base_workspace
            .begin_projection()
            .expect("projection"),
    )
    .with_publication_observer(
        session.workspace_publication_observer(&HashMap::from([("requested".into(), 2)])),
    );
    projection
        .assign_executor_slot(2, "requested", Value::Num(7.0))
        .expect("source store");
    projection.publish().expect("committed source store");
    projection
        .assign("loaded", Value::Num(9.0))
        .expect("effect import");
    projection.publish().expect("committed effect import");
    drop(projection);
    session
        .sync_workspace_binding_metadata()
        .expect("canonical ingress");
    assert!(matches!(
        &session.workspace_bindings.borrow()["requested"].key,
        crate::abi::WorkspaceBindingKey::SourceBinding { binding, .. } if binding.0 == "requested"
    ));
    assert!(matches!(
        &session.workspace_bindings.borrow()["loaded"].key,
        crate::abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "loaded"
    ));
    assert_eq!(session.lowering_workspace_bindings()["requested"], 2);
    assert_eq!(
        session.logical_workspace_value("requested").unwrap(),
        Some(Value::Num(7.0))
    );
    assert_eq!(
        session.logical_workspace_value("loaded").unwrap(),
        Some(Value::Num(9.0))
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn published_source_origin_survives_abandoned_invocation_and_next_source_ingress() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let original = crate::abi::SourceIdentity::PathAndContentHash {
        path: "first.m".into(),
        hash: "first-revision".into(),
    };
    session.active_source_identity = Some(original.clone());
    let projection = runmat_runtime::workspace::BaseWorkspaceProjectionService::new(
        session
            .base_workspace
            .begin_projection()
            .expect("projection"),
    )
    .with_publication_observer(
        session.workspace_publication_observer(&HashMap::from([("retained".into(), 3)])),
    );
    projection
        .assign_executor_slot(3, "retained", Value::Num(13.0))
        .expect("source store");
    projection.publish().expect("publication before failure");
    drop(projection);
    session.active_source_identity = Some(crate::abi::SourceIdentity::PathAndContentHash {
        path: "later.m".into(),
        hash: "later-revision".into(),
    });
    session
        .sync_workspace_binding_metadata()
        .expect("next request ingress");
    assert!(matches!(
        &session.workspace_bindings.borrow()["retained"].key,
        crate::abi::WorkspaceBindingKey::SourceBinding { source, binding, .. }
            if source == &original && binding.0 == "retained"
    ));
    assert_eq!(
        session.logical_workspace_value("retained").unwrap(),
        Some(Value::Num(13.0))
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn canonical_ingress_retains_live_slots_and_admits_effect_only_names() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let mut delta = runmat_runtime::workspace::WorkspaceDelta::new();
    delta.assign("old", Value::Num(7.0)).expect("old value");
    delta
        .assign("loaded", Value::Num(9.0))
        .expect("imported value");
    session
        .base_workspace
        .apply(delta)
        .expect("ordered publication");
    session.bind_workspace_slot("old".into(), 3);
    session.bind_workspace_slot("retired".into(), 8);
    session.active_source_identity = Some(crate::abi::SourceIdentity::PathAndContentHash {
        path: "source.m".into(),
        hash: "revision".into(),
    });
    session
        .sync_workspace_binding_metadata()
        .expect("canonical ingress");
    let slots = session.lowering_workspace_bindings();
    assert_eq!(slots.get("old"), Some(&3));
    assert_eq!(slots.get("loaded"), Some(&4));
    assert!(!slots.contains_key("retired"));
    assert!(matches!(
        &session.workspace_bindings.borrow()["loaded"].key,
        crate::abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "loaded"
    ));
    assert_eq!(
        session.logical_workspace_value("old").unwrap(),
        Some(Value::Num(7.0))
    );
    assert_eq!(
        session.logical_workspace_value("loaded").unwrap(),
        Some(Value::Num(9.0))
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn global_declaration_and_retirement_use_durable_namespace() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    {
        let _selected = session.runtime_context.enter();
        runmat_runtime::workspace::session::declare_global_named("g");
    }
    let mut delta = runmat_runtime::workspace::WorkspaceDelta::new();
    delta.bind_global("g");
    session.base_workspace.apply(delta).expect("global binding");
    session.active_source_identity = Some(crate::abi::SourceIdentity::PathAndContentHash {
        path: "active.m".into(),
        hash: "revision".into(),
    });
    session
        .sync_workspace_binding_metadata()
        .expect("global ingress");
    assert!(matches!(
        &session.workspace_bindings.borrow()["g"].key,
        crate::abi::WorkspaceBindingKey::Global { .. }
    ));
    let mut delta = runmat_runtime::workspace::WorkspaceDelta::new();
    delta.unbind_global("g");
    delta.assign("g", Value::Num(11.0)).expect("new local");
    session
        .base_workspace
        .apply(delta)
        .expect("global retirement");
    session
        .sync_workspace_binding_metadata()
        .expect("local ingress");
    assert!(matches!(
        &session.workspace_bindings.borrow()["g"].key,
        crate::abi::WorkspaceBindingKey::Interactive { .. }
    ));
    assert_eq!(
        session.logical_workspace_value("g").unwrap(),
        Some(Value::Num(11.0))
    );
    session
        .reconcile_workspace_global_binding_keys(&HashMap::new(), &HashSet::new())
        .expect("effect local reconciliation");
    assert!(matches!(
        &session.workspace_bindings.borrow()["g"].key,
        crate::abi::WorkspaceBindingKey::Interactive { .. }
    ));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn completed_register_metadata_cannot_restore_cleared_values() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let mut delta = runmat_runtime::workspace::WorkspaceDelta::new();
    delta
        .assign("live", Value::Num(12.0))
        .expect("canonical value");
    session.base_workspace.apply(delta).expect("publish value");
    session.refresh_workspace_binding_slots(
        &HashMap::from([
            ("live".into(), (2, Value::Num(-1.0))),
            ("cleared".into(), (3, Value::Num(99.0))),
        ]),
        &HashSet::new(),
    );
    session
        .sync_workspace_binding_metadata()
        .expect("retire stale slots");
    assert_eq!(session.lowering_workspace_bindings().get("live"), Some(&2));
    assert!(!session
        .lowering_workspace_bindings()
        .contains_key("cleared"));
    assert_eq!(
        session.logical_workspace_value("live").unwrap(),
        Some(Value::Num(12.0))
    );
    assert_eq!(session.logical_workspace_value("cleared").unwrap(), None);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn duplicate_live_slot_metadata_is_remapped_without_changing_values() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let mut delta = runmat_runtime::workspace::WorkspaceDelta::new();
    delta.assign("a", Value::Num(1.0)).expect("a");
    delta.assign("b", Value::Num(2.0)).expect("b");
    session.base_workspace.apply(delta).expect("publish both");
    session.bind_workspace_slot("a".into(), 5);
    session.bind_workspace_slot("b".into(), 5);
    session
        .sync_workspace_binding_metadata()
        .expect("checked remap");
    assert_eq!(session.lowering_workspace_bindings().get("a"), Some(&5));
    assert_eq!(session.lowering_workspace_bindings().get("b"), Some(&6));
    assert_eq!(
        session.logical_workspace_value("a").unwrap(),
        Some(Value::Num(1.0))
    );
    assert_eq!(
        session.logical_workspace_value("b").unwrap(),
        Some(Value::Num(2.0))
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn exhausted_live_slot_is_rejected_without_mutating_binding_metadata() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    let mut delta = runmat_runtime::workspace::WorkspaceDelta::new();
    delta
        .assign("live", Value::Num(1.0))
        .expect("canonical value");
    session.base_workspace.apply(delta).expect("publication");
    session.bind_workspace_slot("live".into(), usize::MAX);
    let before = session.workspace_bindings.borrow().clone();
    let error = session
        .sync_workspace_binding_metadata()
        .expect_err("unrepresentable live slot must fail");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:WorkspaceBindingSlots")
    );
    assert_eq!(session.workspace_bindings.borrow().len(), before.len());
    for (name, prior) in before {
        let bindings = session.workspace_bindings.borrow();
        let current = &bindings[&name];
        assert_eq!(current.slot, prior.slot);
        assert_eq!(current.key, prior.key);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn invalid_source_slot_reports_committed_value_without_claiming_source_origin() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    session.active_source_identity = Some(crate::abi::SourceIdentity::PathAndContentHash {
        path: "active.m".into(),
        hash: "revision".into(),
    });
    let projection = runmat_runtime::workspace::BaseWorkspaceProjectionService::new(
        session
            .base_workspace
            .begin_projection()
            .expect("projection"),
    )
    .with_publication_observer(session.workspace_publication_observer(&HashMap::new()));
    projection
        .assign_executor_slot(1, "committed", Value::Num(17.0))
        .expect("staged source store");
    let error = projection
        .publish()
        .expect_err("unknown compiled slot must not claim a source binding");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:WorkspaceBindingSlots")
    );
    assert_eq!(
        session.logical_workspace_value("committed").unwrap(),
        Some(Value::Num(17.0))
    );
    assert!(!session
        .workspace_bindings
        .borrow()
        .contains_key("committed"));
    drop(projection);
    session
        .sync_workspace_binding_metadata()
        .expect("recover canonical value without fabricated source origin");
    assert!(matches!(
        &session.workspace_bindings.borrow()["committed"].key,
        crate::abi::WorkspaceBindingKey::Interactive { .. }
    ));
}
