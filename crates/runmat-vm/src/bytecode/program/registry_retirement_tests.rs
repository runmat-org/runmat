use super::{FunctionBytecode, FunctionRegistry};
use runmat_hir::{FunctionId, SourceId};
use std::collections::HashMap;

fn function(id: usize, name: &str, source: usize) -> FunctionBytecode {
    FunctionBytecode {
        function: FunctionId(id),
        display_name: name.into(),
        source_id: Some(SourceId(source)),
        ..Default::default()
    }
}

fn registry() -> FunctionRegistry {
    FunctionRegistry::new(HashMap::from([
        (FunctionId(1), function(1, "first", 5)),
        (FunctionId(2), function(2, "second", 5)),
        (FunctionId(3), function(3, "unrelated", 6)),
    ]))
}

#[test]
fn execution_snapshot_clone_is_shared_until_publication_mutates() {
    let original = registry();
    let mut publication = original.clone();
    assert!(std::sync::Arc::ptr_eq(&original.data, &publication.data));
    publication.insert_replacing_name(function(4, "new", 8));
    assert!(!std::sync::Arc::ptr_eq(&original.data, &publication.data));
    assert_eq!(original.resolve_name("new"), None);
    assert_eq!(publication.resolve_name("new"), Some(FunctionId(4)));
    let serialized = serde_json::to_value(&publication).expect("registry record");
    assert!(serialized.get("functions").is_some());
    assert!(serialized.get("data").is_none());
}

#[test]
fn function_execution_uses_shared_registry_without_cloning_legacy_body_map() {
    let registry = registry();
    let function = registry.get(FunctionId(1)).unwrap();
    let bytecode = function.execution_bytecode(&registry);
    assert!(bytecode.bound_functions.is_empty());
    assert!(std::sync::Arc::ptr_eq(
        &registry.data,
        &bytecode.function_registry.data
    ));
    assert!(bytecode.function_registry.get(FunctionId(2)).is_some());
}

#[test]
fn source_retirement_removes_all_target_aliases_and_preserves_rebound_names() {
    let mut registry = registry();
    registry.names.insert("first_alias".into(), FunctionId(1));
    registry.names.insert("second_alias".into(), FunctionId(2));
    registry.names.insert("rebound_alias".into(), FunctionId(1));
    registry.names.insert("rebound_alias".into(), FunctionId(3));
    registry.names.insert("first".into(), FunctionId(3));
    let removed = registry.remove_source(SourceId(5));
    assert_eq!(removed.len(), 2);
    for name in ["first_alias", "second_alias", "second"] {
        assert_eq!(registry.resolve_name(name), None);
    }
    for name in ["first", "rebound_alias", "unrelated"] {
        assert_eq!(registry.resolve_name(name), Some(FunctionId(3)));
    }
    assert_eq!(registry.functions.len(), 1);
    assert!(registry.functions_for_source(SourceId(5)).is_empty());
    assert_eq!(registry.functions_for_source(SourceId(6)), &[FunctionId(3)]);
}

#[test]
fn replacement_retires_old_aliases_without_removing_new_name() {
    let mut registry = registry();
    registry.names.insert("old_alias".into(), FunctionId(1));
    registry.names.insert("rebound_alias".into(), FunctionId(3));
    registry.insert_replacing_name(function(4, "first", 7));
    assert_eq!(registry.resolve_name("first"), Some(FunctionId(4)));
    assert_eq!(registry.resolve_name("old_alias"), None);
    assert_eq!(registry.resolve_name("rebound_alias"), Some(FunctionId(3)));
    assert!(registry.get(FunctionId(1)).is_none());
    assert_eq!(registry.functions_for_source(SourceId(5)), &[FunctionId(2)]);
    assert_eq!(registry.functions_for_source(SourceId(7)), &[FunctionId(4)]);
}

#[test]
fn same_identity_refresh_preserves_aliases_and_updates_source_membership() {
    let mut registry = registry();
    registry.names.insert("live_alias".into(), FunctionId(1));
    registry.insert_replacing_name(function(1, "first", 7));
    assert_eq!(registry.resolve_name("first"), Some(FunctionId(1)));
    assert_eq!(registry.resolve_name("live_alias"), Some(FunctionId(1)));
    assert_eq!(registry.functions_for_source(SourceId(5)), &[FunctionId(2)]);
    assert_eq!(registry.functions_for_source(SourceId(7)), &[FunctionId(1)]);
    registry.remove(FunctionId(1)).unwrap();
    assert_eq!(registry.resolve_name("first"), None);
    assert_eq!(registry.resolve_name("live_alias"), None);
    assert!(registry.functions_for_source(SourceId(7)).is_empty());
    assert!(registry.get(FunctionId(3)).is_some());
}

#[test]
fn leased_old_body_survives_name_republication_without_becoming_name_visible() {
    let mut registry = registry();
    let old_owner = registry
        .get(FunctionId(1))
        .unwrap()
        .definition_owner
        .clone();
    let lease = runmat_value::FunctionDefinitionOwner::issue(&old_owner, 1);
    registry.insert_replacing_name(function(4, "first", 7));
    assert_eq!(registry.resolve_name("first"), Some(FunctionId(4)));
    assert!(registry.get(FunctionId(1)).is_none());
    assert_eq!(
        registry
            .get_owned(FunctionId(1), &lease)
            .unwrap()
            .display_name,
        "first"
    );
    let wrong = runmat_value::FunctionDefinitionOwner::issue(
        &registry.get(FunctionId(4)).unwrap().definition_owner,
        4,
    );
    assert!(registry.get_owned(FunctionId(1), &wrong).is_none());
    drop(lease);
    registry.insert_replacing_name(function(5, "second", 8));
    assert!(registry.retired.get(&FunctionId(1)).is_none());
}

#[test]
fn reused_numeric_id_keeps_old_lease_on_old_body() {
    let mut registry = registry();
    let lease = runmat_value::FunctionDefinitionOwner::issue(
        &registry.get(FunctionId(1)).unwrap().definition_owner,
        1,
    );
    registry.insert_replacing_name(function(1, "first", 7));
    let new_body = registry.get(FunctionId(1)).unwrap();
    let old_body = registry.get_owned(FunctionId(1), &lease).unwrap();
    assert!(!std::sync::Arc::ptr_eq(
        &new_body.definition_owner,
        &old_body.definition_owner
    ));
    assert_eq!(old_body.source_id, Some(SourceId(5)));
    assert_eq!(new_body.source_id, Some(SourceId(7)));
}

#[test]
fn unchanged_origin_refresh_keeps_existing_definition_lease() {
    let mut registry = registry();
    let origin = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("unchanged program"),
        function: runmat_types::ProgramFunctionId(1),
    };
    registry
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .binding_origin = Some(origin);
    let lease = runmat_value::FunctionDefinitionOwner::issue(
        &registry.get(FunctionId(1)).unwrap().definition_owner,
        1,
    );
    let mut republished = function(1, "first", 5);
    republished.binding_origin = Some(origin);
    registry.insert_replacing_name(republished);
    assert!(registry.get_owned(FunctionId(1), &lease).is_some());
}

#[test]
fn old_caller_view_retains_old_helper_without_name_fallback() {
    let mut registry = registry();
    registry
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .instructions = vec![super::Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)];
    let owners = registry
        .definition_dependency_owners(FunctionId(1))
        .unwrap();
    let session = std::sync::Arc::new(());
    let authority = runmat_value::BoundFunctionAuthority::local(
        std::sync::Arc::downgrade(&session),
        runmat_value::FunctionDefinitionOwner::issue(&owners[0].1, owners[0].0 .0),
        owners[1..]
            .iter()
            .map(|(id, owner)| runmat_value::FunctionDefinitionOwner::issue(owner, id.0))
            .collect(),
        runmat_value::LocalExport::Ephemeral,
    );
    registry.remove_source(SourceId(5));
    registry.insert_replacing_name(function(4, "first", 7));
    registry.insert_replacing_name(function(5, "second", 7));
    let old = registry
        .owned_view(FunctionId(1), &authority)
        .expect("old owned definition and helper");
    assert!(std::sync::Arc::ptr_eq(&old.data, &registry.data));
    assert!(old.get(FunctionId(1)).is_some());
    assert!(old.get(FunctionId(2)).is_some());
    assert!(old.static_binding(FunctionId(2)).is_some());
    let helper_binding = old.static_binding(FunctionId(2)).unwrap();
    assert!(matches!(
        &helper_binding.authority,
        runmat_value::BoundFunctionAuthority::Local { dependencies, .. }
            if dependencies.iter().any(|lease| lease.function() == 2 && lease.matches(&old.get(FunctionId(2)).unwrap().definition_owner))
    ));
    assert!(old.static_binding(FunctionId(5)).is_some());
    assert!(!old
        .static_binding(FunctionId(5))
        .unwrap()
        .authority
        .admits_local(&std::sync::Arc::downgrade(&session)));
    assert_eq!(old.current_view().get(FunctionId(1)), None);
    assert_eq!(
        old.current_view()
            .get(FunctionId(4))
            .map(|f| f.display_name.as_str()),
        Some("first")
    );
    assert_eq!(old.resolve_name("first"), Some(FunctionId(4)));
    assert_eq!(old.resolve_name("second"), Some(FunctionId(5)));
    assert!(registry.get(FunctionId(1)).is_none());
}

#[test]
fn reused_id_static_edge_and_dynamic_name_have_distinct_owners() {
    let mut registry = registry();
    registry
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .private_owner_scope = "old/private".into();
    registry
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .instructions = vec![super::Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)];
    let owners = registry
        .definition_dependency_owners(FunctionId(1))
        .unwrap();
    let issuer = std::sync::Arc::new(());
    let authority = runmat_value::BoundFunctionAuthority::local(
        std::sync::Arc::downgrade(&issuer),
        runmat_value::FunctionDefinitionOwner::issue(&owners[0].1, 1),
        owners[1..]
            .iter()
            .map(|(id, owner)| runmat_value::FunctionDefinitionOwner::issue(owner, id.0))
            .collect(),
        runmat_value::LocalExport::Ephemeral,
    );
    registry.insert_replacing_name(function(1, "first", 7));
    registry.insert_replacing_name(function(2, "second", 7));
    let view = registry.owned_view(FunctionId(1), &authority).unwrap();
    assert_eq!(view.resolve_name("second"), Some(FunctionId(2)));
    assert_eq!(
        view.get(FunctionId(2)).unwrap().source_id,
        Some(SourceId(5))
    );
    assert_eq!(
        view.current_view().get(FunctionId(2)).unwrap().source_id,
        Some(SourceId(7))
    );
    assert_eq!(
        view.get(FunctionId(1)).unwrap().private_owner_scope,
        "old/private"
    );
    assert_eq!(
        view.current_view()
            .get(FunctionId(1))
            .unwrap()
            .private_owner_scope,
        ""
    );
    assert!(view.static_binding(FunctionId(2)).is_some());
}
