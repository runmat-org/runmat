use super::*;
use runmat_runtime::context::RuntimeBaseWorkspaceService;
use runmat_runtime::context::RuntimeWorkspaceService;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn invocation_base_write_prepares_the_same_observed_binding_before_publication() {
    let session = RunMatSession::with_options(false, false).expect("session");
    let runtime = &session.runtime_context;
    let projection = std::rc::Rc::new(
        runmat_runtime::workspace::BaseWorkspaceProjectionService::new_for_context(
            session
                .base_workspace
                .begin_projection()
                .expect("projection"),
            runtime,
        )
        .with_publication_observer(
            session
                .workspace_publication_observer_with_context(runtime)
                .expect("admitted observer"),
        ),
    );
    let base = runmat_runtime::workspace::BaseWorkspaceService::new(session.base_workspace.clone())
        .with_invocation_projection(projection.clone())
        .expect("same canonical owner");
    base.assign_logical_ref_with_context(
        &runtime.session_variable_authority(),
        runtime,
        "effect_name",
        &Value::Num(17.0),
    )
    .expect("selected base mutation");
    assert_eq!(
        session.logical_workspace_value("effect_name").unwrap(),
        Some(Value::Num(17.0))
    );
    assert!(matches!(
        &session.workspace_bindings.borrow()["effect_name"].key,
        crate::abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "effect_name"
    ));
    let foreign = runmat_runtime::workspace::BaseWorkspaceService::new(
        runmat_runtime::workspace::BaseWorkspaceOwner::new(),
    );
    assert!(foreign.with_invocation_projection(projection).is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn invocation_base_write_rejects_unrepresentable_binding_before_canonical_mutation() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    session.bind_workspace_slot("prior".into(), usize::MAX);
    let runtime = &session.runtime_context;
    let projection = std::rc::Rc::new(
        runmat_runtime::workspace::BaseWorkspaceProjectionService::new_for_context(
            session
                .base_workspace
                .begin_projection()
                .expect("projection"),
            runtime,
        )
        .with_publication_observer(
            session
                .workspace_publication_observer_with_context(runtime)
                .expect("admitted observer"),
        ),
    );
    let base = runmat_runtime::workspace::BaseWorkspaceService::new(session.base_workspace.clone())
        .with_invocation_projection(projection)
        .expect("same canonical owner");
    let error = base
        .assign_logical_ref_with_context(
            &runtime.session_variable_authority(),
            runtime,
            "effect_name",
            &Value::Num(17.0),
        )
        .expect_err("observer must reject exhausted binding space before mutation");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:WorkspaceBindingSlots")
    );
    assert_eq!(
        session.logical_workspace_value("effect_name").unwrap(),
        None
    );
    assert!(!session
        .workspace_bindings
        .borrow()
        .contains_key("effect_name"));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn source_store_and_effect_import_keep_distinct_keys_under_one_active_source() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    session.active_source_identity = Some(crate::abi::SourceIdentity::PathAndContentHash {
        path: "active.m".into(),
        hash: "revision".into(),
    });
    let projection = runmat_runtime::workspace::BaseWorkspaceProjectionService::new_for_context(
        session
            .base_workspace
            .begin_projection()
            .expect("projection"),
        &session.runtime_context,
    )
    .with_publication_observer(
        session
            .workspace_publication_observer_with_context(&session.runtime_context)
            .expect("admitted publication observer"),
    );
    projection
        .assign_executor_slot_ref(&session.runtime_context, 2, "requested", &Value::Num(7.0))
        .expect("source store");
    projection
        .assign_executor_slot_ref(&session.runtime_context, 10, "appended", &Value::Num(8.0))
        .expect("source slot appended after observer construction");
    projection
        .publish_with_context(&session.runtime_context)
        .expect("committed source store");
    assert!(matches!(
        &session.workspace_bindings.borrow()["appended"].key,
        crate::abi::WorkspaceBindingKey::SourceBinding { binding, .. } if binding.0 == "appended"
    ));
    projection
        .assign_with_context(&session.runtime_context, "loaded", Value::Num(9.0))
        .expect("effect import");
    projection
        .publish_with_context(&session.runtime_context)
        .expect("committed effect import");
    assert!(matches!(
        &session.workspace_bindings.borrow()["loaded"].key,
        crate::abi::WorkspaceBindingKey::Interactive { name, .. } if name.0 == "loaded"
    ));
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
    let binding_name = match &session.workspace_bindings.borrow()["requested"].key {
        crate::abi::WorkspaceBindingKey::SourceBinding { binding, .. } => binding.0.as_ptr(),
        _ => panic!("source publication retains its source binding"),
    };
    session
        .refresh_workspace_binding_slots([("requested", 5)], |_, _| Ok(None))
        .expect("slot refresh");
    let refreshed = session.workspace_bindings.borrow();
    assert_eq!(refreshed["requested"].slot, 5);
    match &refreshed["requested"].key {
        crate::abi::WorkspaceBindingKey::SourceBinding { binding, .. } => {
            assert_eq!(binding.0.as_ptr(), binding_name);
        }
        _ => panic!("slot refresh must not replace the source origin"),
    }
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
    let projection = runmat_runtime::workspace::BaseWorkspaceProjectionService::new_for_context(
        session
            .base_workspace
            .begin_projection()
            .expect("projection"),
        &session.runtime_context,
    )
    .with_publication_observer(
        session
            .workspace_publication_observer_with_context(&session.runtime_context)
            .expect("admitted publication observer"),
    );
    projection
        .assign_executor_slot_ref(&session.runtime_context, 3, "retained", &Value::Num(13.0))
        .expect("source store");
    projection
        .publish_with_context(&session.runtime_context)
        .expect("publication before failure");
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
        runmat_runtime::workspace::session::declare_global_named("g")
            .expect("session global declaration");
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
    let authority = session.runtime_context.session_variable_authority();
    for (name, value) in [
        ("first_target", 21.0),
        ("next_target", 22.0),
        ("var_7", 99.0),
    ] {
        authority
            .store_global_with_context(&session.runtime_context, name, Value::Num(value))
            .unwrap();
    }
    authority
        .store_global_slot_with_context(&session.runtime_context, 7, Value::Num(23.0))
        .unwrap();
    for (target, expected) in [
        (
            runmat_runtime::workspace::GlobalBindingTarget::Named(
                runmat_runtime::workspace::WorkspaceName::unbounded("first_target"),
            ),
            21.0,
        ),
        (
            runmat_runtime::workspace::GlobalBindingTarget::Named(
                runmat_runtime::workspace::WorkspaceName::unbounded("next_target"),
            ),
            22.0,
        ),
        (
            runmat_runtime::workspace::GlobalBindingTarget::LegacySlot(7),
            23.0,
        ),
    ] {
        let mut retarget = runmat_runtime::workspace::WorkspaceDelta::new();
        retarget.bind_global_target("g", target.clone());
        session.base_workspace.apply(retarget).unwrap();
        session.sync_workspace_binding_metadata().unwrap();
        assert!(super::global_key::matches_session_global_target(
            &session.workspace_bindings.borrow()["g"].key,
            session.workspace_handle().0,
            &target,
        ));
        assert_ne!(session.workspace_bindings.borrow()["g"].slot, 7);
        assert_eq!(
            session.logical_workspace_value("g").unwrap(),
            Some(Value::Num(expected))
        );
        assert_eq!(
            session.logical_workspace_values().unwrap().get("g"),
            Some(&Value::Num(expected))
        );
        assert!(session.logical_workspace_names().unwrap().contains("g"));
    }
    assert_eq!(
        authority.global_value("var_7").unwrap(),
        Some(Value::Num(99.0))
    );
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
    // A real invocation admits canonical binding metadata before compilation;
    // completion may refresh that slot but cannot create a binding from registers.
    session
        .sync_workspace_binding_metadata()
        .expect("admit the live binding at invocation ingress");
    session
        .refresh_workspace_binding_slots([("live", 2), ("cleared", 3)], |_, _| Ok(None))
        .expect("slot refresh");
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
fn imported_binding_metadata_uses_final_assignment_for_repeated_name() {
    let session = RunMatSession::with_options(false, false).expect("session");
    let entries = vec![
        ("same".to_owned(), Value::Num(1.0)),
        ("other".to_owned(), Value::Num(3.0)),
        ("same".to_owned(), Value::Num(2.0)),
    ];
    let (prepared, _lease) = session
        .prepare_import_binding_metadata(&entries)
        .expect("admitted import metadata");
    assert_eq!(prepared.len(), 2);
    assert_eq!(prepared["same"].slot, 2);
    assert_eq!(prepared["other"].slot, 1);
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
    let before = session.workspace_bindings.borrow()["live"].key.clone();
    let error = session
        .sync_workspace_binding_metadata()
        .expect_err("unrepresentable live slot must fail");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:WorkspaceBindingSlots")
    );
    let bindings = session.workspace_bindings.borrow();
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings["live"].slot, usize::MAX);
    assert_eq!(bindings["live"].key, before);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unrepresentable_source_slot_rejects_publication_before_canonical_state_changes() {
    let mut session = RunMatSession::with_options(false, false).expect("session");
    session.active_source_identity = Some(crate::abi::SourceIdentity::PathAndContentHash {
        path: "active.m".into(),
        hash: "revision".into(),
    });
    let projection = runmat_runtime::workspace::BaseWorkspaceProjectionService::new_for_context(
        session
            .base_workspace
            .begin_projection()
            .expect("projection"),
        &session.runtime_context,
    )
    .with_publication_observer(
        session
            .workspace_publication_observer_with_context(&session.runtime_context)
            .expect("admitted publication observer"),
    );
    let error = projection
        .assign_executor_slot_ref(
            &session.runtime_context,
            usize::MAX,
            "committed",
            &Value::Num(17.0),
        )
        .expect_err("unrepresentable slot must fail before the source store");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:WorkspaceBindingSlots")
    );
    assert_eq!(session.logical_workspace_value("committed").unwrap(), None);
    assert!(!session
        .workspace_bindings
        .borrow()
        .contains_key("committed"));
    drop(projection);
    session
        .sync_workspace_binding_metadata()
        .expect("reconcile unchanged canonical state");
    assert!(!session
        .workspace_bindings
        .borrow()
        .contains_key("committed"));
}
