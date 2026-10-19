use super::{BytecodeSourceRef, FunctionBytecode, FunctionRegistry};
use crate::FunctionSourceProjectionError;
use runmat_hir::{FunctionId, SourceId};
use std::collections::HashMap;

thread_local! {
    static ISSUED: (runmat_runtime::context::RuntimeContext, std::cell::RefCell<Vec<runmat_runtime::source_context::RuntimeSourceLease>>) = (
        runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        )),
        std::cell::RefCell::new(Vec::new()),
    );
}

fn source_ref(source: usize) -> runmat_runtime::source_context::RuntimeSourceRef {
    ISSUED.with(|(runtime, leases)| {
        let mut leases = leases.borrow_mut();
        while leases.len() <= source {
            let ordinal = leases.len();
            leases.push(
                runmat_runtime::source_context::lease_source_context_in_context(
                    runtime,
                    format!("source-{ordinal}.m"),
                    None,
                    format!("source {ordinal}"),
                )
                .expect("test source owner"),
            );
        }
        leases[source].source_ref()
    })
}

fn runtime_ref(source: usize) -> BytecodeSourceRef {
    BytecodeSourceRef::Runtime(source_ref(source))
}

fn function(id: usize, name: &str, source: usize) -> FunctionBytecode {
    FunctionBytecode {
        function: FunctionId(id),
        display_name: name.into(),
        source_id: Some(runtime_ref(source)),
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
fn program_source_members_use_selected_runtime_projection() {
    use runmat_runtime::source_context::ProgramSourceRecord;
    use runmat_types::ProgramSourceId;

    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_program_sources([ProgramSourceRecord {
        ordinal: ProgramSourceId(0),
        name: std::sync::Arc::from("locals.m"),
        fullpath_name: None,
        text: std::sync::Arc::from("primary and local functions"),
    }])
    .expect("selected source");
    let program = Some(BytecodeSourceRef::Program(ProgramSourceId(0)));
    let source = runtime.program_source_id(ProgramSourceId(0)).unwrap();
    let registry = FunctionRegistry::new(HashMap::from([
        (
            FunctionId(2),
            FunctionBytecode {
                function: FunctionId(2),
                display_name: "child_local".into(),
                source_id: program,
                source_role: runmat_types::FunctionSourceRole::Local,
                ..Default::default()
            },
        ),
        (
            FunctionId(1),
            FunctionBytecode {
                function: FunctionId(1),
                display_name: "primary_file".into(),
                source_id: program,
                source_role: runmat_types::FunctionSourceRole::Primary,
                ..Default::default()
            },
        ),
    ]));

    let group = registry.execution_source_group_facts_in(&runtime, source);
    assert_eq!(group.len(), 2);
    assert_eq!(group[0].name, "primary_file");
    assert_eq!(group[0].role, runmat_types::FunctionSourceRole::Primary);
    assert_eq!(group[1].name, "child_local");
    assert_eq!(group[1].role, runmat_types::FunctionSourceRole::Local);
    assert!(group.iter().all(|member| member.source_id == source));
    let owners = group[0]
        .definition_owners
        .as_ref()
        .expect("complete program source graph");
    assert_eq!(owners.len(), 2);
    assert!(std::sync::Arc::ptr_eq(
        owners,
        group[1].definition_owners.as_ref().unwrap()
    ));
}

#[test]
fn source_group_refuses_foreign_and_retired_issued_owners() {
    let first = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let second = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let foreign = runmat_runtime::source_context::lease_source_context_in_context(
        &first,
        "foreign.m",
        None,
        "function x = foreign; end",
    )
    .expect("foreign owner");
    let local = runmat_runtime::source_context::lease_source_context_in_context(
        &second,
        "local.m",
        None,
        "function x = local; end",
    )
    .expect("local owner");
    assert_eq!(foreign.source_id(), local.source_id());
    let registry = FunctionRegistry::new(HashMap::from([
        (
            FunctionId(1),
            FunctionBytecode {
                function: FunctionId(1),
                display_name: "foreign".into(),
                source_id: Some(BytecodeSourceRef::Runtime(foreign.source_ref())),
                ..Default::default()
            },
        ),
        (
            FunctionId(2),
            FunctionBytecode {
                function: FunctionId(2),
                display_name: "local".into(),
                source_id: Some(BytecodeSourceRef::Runtime(local.source_ref())),
                ..Default::default()
            },
        ),
    ]));
    assert!(registry
        .execution_source_group(&second, foreign.source_ref())
        .is_empty());
    let group = registry.execution_source_group_facts_in(&second, local.source_id());
    assert_eq!(group.len(), 1);
    assert_eq!(group[0].name, "local");
    assert_eq!(group[0].function, 2);
    assert_eq!(group[0].source_id, local.source_id());
    assert!(group[0]
        .definition_owners
        .as_ref()
        .is_some_and(|owners| !owners.is_empty()));

    let retired = local.source_ref();
    drop(local);
    assert!(registry.execution_source_group(&second, retired).is_empty());
    assert!(registry
        .execution_source_group_facts_in(&second, retired.source_id())
        .is_empty());
}

#[test]
fn retained_program_source_group_uses_only_captured_definition_owners() {
    use runmat_runtime::source_context::ProgramSourceRecord;
    use runmat_types::ProgramSourceId;

    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_program_sources([ProgramSourceRecord {
        ordinal: ProgramSourceId(0),
        name: std::sync::Arc::from("retained.m"),
        fullpath_name: None,
        text: std::sync::Arc::from("primary and local functions"),
    }])
    .expect("selected source");
    let source = runtime.program_source_id(ProgramSourceId(0)).unwrap();
    let program = Some(BytecodeSourceRef::Program(ProgramSourceId(0)));
    let mut registry = FunctionRegistry::new(HashMap::from([
        (
            FunctionId(1),
            FunctionBytecode {
                function: FunctionId(1),
                display_name: "old_primary".into(),
                source_id: program,
                source_file: Some("/old/retained.m".into()),
                source_role: runmat_types::FunctionSourceRole::Primary,
                ..Default::default()
            },
        ),
        (
            FunctionId(2),
            FunctionBytecode {
                function: FunctionId(2),
                display_name: "old_local".into(),
                source_id: program,
                source_file: Some("/old/retained.m".into()),
                source_role: runmat_types::FunctionSourceRole::Local,
                ..Default::default()
            },
        ),
    ]));
    let (authority, _issuer) = local_authority(&registry, FunctionId(1));
    registry.remove(FunctionId(1)).expect("retire primary");
    registry.remove(FunctionId(2)).expect("retire local");
    for (id, name) in [(1, "new_primary"), (2, "new_local")] {
        registry.insert_replacing_name(FunctionBytecode {
            function: FunctionId(id),
            display_name: name.into(),
            source_id: program,
            ..Default::default()
        });
    }

    let selected = registry
        .owned_view(FunctionId(1), &authority)
        .expect("retained program owner view");
    let old = selected.execution_source_group_facts_in(&runtime, source);
    assert_eq!(old.len(), 2);
    assert_eq!(old[0].name, "old_primary");
    assert_eq!(old[1].name, "old_local");
    assert!(old
        .iter()
        .all(|info| info.source_file.as_deref() == Some("/old/retained.m")));
    let owners = old[0].definition_owners.as_ref().expect("retained graph");
    assert_eq!(owners.len(), 2);
    assert!(std::sync::Arc::ptr_eq(
        owners,
        old[1].definition_owners.as_ref().unwrap()
    ));
    let current = registry.execution_source_group_facts_in(&runtime, source);
    assert_eq!(current.len(), 2);
    assert_eq!(current[0].name, "new_primary");
    assert_eq!(current[1].name, "new_local");
}

fn local_authority(
    registry: &FunctionRegistry,
    function: FunctionId,
) -> (runmat_value::BoundFunctionAuthority, std::sync::Arc<()>) {
    let owners = registry
        .definition_dependency_owners(function)
        .expect("complete retained definition graph");
    let issuer = std::sync::Arc::new(());
    let authority = runmat_value::BoundFunctionAuthority::local(
        std::sync::Arc::downgrade(&issuer),
        runmat_value::FunctionDefinitionOwner::issue(&owners[0].1, owners[0].0 .0),
        owners[1..]
            .iter()
            .map(|(id, owner)| runmat_value::FunctionDefinitionOwner::issue(owner, id.0))
            .collect(),
        runmat_value::LocalExport::Ephemeral,
    );
    (authority, issuer)
}

#[test]
fn retained_definition_without_published_id_remains_selected_for_graph_facts() {
    let origin = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("retained-graph"),
        function: runmat_types::ProgramFunctionId(1),
    };
    let mut old = function(1, "old_graph_body", 5);
    old.binding_origin = Some(origin);
    let mut registry = FunctionRegistry::new(HashMap::from([(FunctionId(1), old)]));
    let (authority, _issuer) = local_authority(&registry, FunctionId(1));
    registry.remove(FunctionId(1)).expect("retired leased body");
    assert!(registry.functions.is_empty());
    let selected = registry
        .owned_view(FunctionId(1), &authority)
        .expect("exact retained execution view");
    let bodies = selected.selected_execution_bodies();
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies[0].display_name, "old_graph_body");
    assert_eq!(bodies[0].binding_origin, Some(origin));
    ISSUED.with(|(runtime, _)| {
        let fact = selected
            .execution_source_fact_for_origin_in(runtime, origin)
            .expect("retained origin source fact");
        assert_eq!(fact.name, "old_graph_body");
        assert_eq!(fact.binding_origin, Some(origin));
        assert!(registry
            .execution_source_fact_for_origin_in(runtime, origin)
            .is_none());
    });
}

#[test]
fn source_projection_preserves_aliases_and_rebuilds_current_membership_atomically() {
    let mut registry = registry();
    registry.names.insert("alias".into(), FunctionId(1));
    let unchanged = registry.clone();
    assert_eq!(
        registry.project_current_source_ids(|source| {
            (source == runtime_ref(5)).then_some(runtime_ref(1))
        }),
        Err(FunctionSourceProjectionError::Missing(runtime_ref(6)))
    );
    assert_eq!(registry.source_functions, unchanged.source_functions);
    assert_eq!(registry.names, unchanged.names);
    assert_eq!(
        registry.get(FunctionId(3)).unwrap().source_id,
        Some(runtime_ref(6))
    );

    registry
        .project_current_source_ids(|source| {
            if source == runtime_ref(5) {
                Some(runtime_ref(1))
            } else if source == runtime_ref(6) {
                Some(runtime_ref(2))
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(registry.resolve_name("alias"), Some(FunctionId(1)));
    assert_eq!(
        registry.functions_for_source(source_ref(1)),
        &[FunctionId(1), FunctionId(2)]
    );
    assert_eq!(
        registry.functions_for_source(source_ref(2)),
        &[FunctionId(3)]
    );
    assert!(registry.functions_for_source(source_ref(5)).is_empty());
}

#[test]
fn mixed_retained_runtime_and_program_sources_project_without_losing_old_owner() {
    let mut program = function(1, "program", 0);
    program.source_id = Some(BytecodeSourceRef::Program(runmat_types::ProgramSourceId(0)));
    let retained = function(2, "retained", 7);
    let mut registry = FunctionRegistry::new(HashMap::from([
        (FunctionId(1), program),
        (FunctionId(2), retained),
    ]));
    let before = registry.source_functions.clone();
    assert_eq!(
        registry.project_current_source_ids(|source| match source {
            BytecodeSourceRef::Program(_) => None,
            runtime => Some(runtime),
        }),
        Err(FunctionSourceProjectionError::Missing(
            BytecodeSourceRef::Program(runmat_types::ProgramSourceId(0))
        ))
    );
    assert_eq!(registry.source_functions, before);
    registry
        .project_current_source_ids(|source| match source {
            BytecodeSourceRef::Program(runmat_types::ProgramSourceId(0)) => Some(runtime_ref(11)),
            runtime => Some(runtime),
        })
        .expect("selected program binding");
    assert_eq!(
        registry.functions_for_source(source_ref(11)),
        &[FunctionId(1)]
    );
    assert_eq!(
        registry.functions_for_source(source_ref(7)),
        &[FunctionId(2)]
    );
}

#[test]
fn retained_source_group_keeps_original_path_and_locals_after_id_reuse() {
    let mut registry = registry();
    registry
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .source_role = runmat_types::FunctionSourceRole::Primary;
    registry
        .functions
        .get_mut(&FunctionId(2))
        .unwrap()
        .source_role = runmat_types::FunctionSourceRole::Local;
    for id in [FunctionId(1), FunctionId(2)] {
        registry.functions.get_mut(&id).unwrap().source_file = Some("/old/first.m".into());
    }
    let owners = registry
        .definition_dependency_owners(FunctionId(1))
        .expect("same-source local owner graph");
    let (root_id, root_owner) = &owners[0];
    let root = runmat_value::FunctionDefinitionOwner::issue(root_owner, root_id.0);
    let dependencies = owners
        .iter()
        .skip(1)
        .map(|(id, owner)| runmat_value::FunctionDefinitionOwner::issue(owner, id.0))
        .collect();
    let authority = runmat_value::BoundFunctionAuthority::local(
        std::sync::Weak::new(),
        root,
        dependencies,
        runmat_value::LocalExport::Ephemeral,
    );
    registry.remove_source(source_ref(5));
    registry.insert_replacing_name(function(1, "replacement", 7));
    let old = registry
        .owned_view(FunctionId(1), &authority)
        .expect("retained old source view");
    let group = old
        .execution_source_group_facts_in(&ISSUED.with(|(runtime, _)| runtime.clone()), SourceId(5));
    assert_eq!(group.len(), 2);
    assert_eq!(group[0].role, runmat_types::FunctionSourceRole::Primary);
    assert_eq!(group[1].role, runmat_types::FunctionSourceRole::Local);
    assert!(std::sync::Arc::ptr_eq(
        group[0].definition_owners.as_ref().unwrap(),
        group[1].definition_owners.as_ref().unwrap()
    ));
    assert!(group
        .iter()
        .all(|info| info.source_file.as_deref() == Some("/old/first.m")));
    assert!(old
        .execution_source_fact_in(&ISSUED.with(|(runtime, _)| runtime.clone()), FunctionId(1))
        .is_some());
    assert!(registry
        .execution_source_group_facts_in(&ISSUED.with(|(runtime, _)| runtime.clone()), SourceId(5))
        .is_empty());
    assert_eq!(registry.resolve_name("replacement"), Some(FunctionId(1)));
}

#[test]
fn selected_origin_binding_ignores_absent_or_foreign_ambient_resolver() {
    let runtime = ISSUED.with(|(runtime, _)| runtime.clone());
    let origin = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256(b"retired-selected-source"),
        function: runmat_types::ProgramFunctionId(1),
    };
    let mut original = registry();
    original
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .binding_origin = Some(origin);
    let original = std::rc::Rc::new(original);
    let selected = original.execution_source_resolver_in(&runtime);
    let mut replacement = original.as_ref().clone();
    replacement.insert_replacing_name(function(1, "replacement", 7));
    assert_ne!(
        replacement
            .get(FunctionId(1))
            .and_then(|body| body.binding_origin),
        Some(origin)
    );
    let resolve = || {
        runmat_runtime::user_functions::resolve_binding_origin_binding_in_context(
            &selected, origin, &runtime,
        )
        .expect("selected older body remains bound")
    };
    assert_eq!(resolve().function, 1);
    let foreign = std::rc::Rc::new(runmat_runtime::user_functions::ExecutionSourceFacts::new(
        std::rc::Rc::new(|_| None),
        std::rc::Rc::new(|_| Vec::new()),
        std::rc::Rc::new(|_| None),
        std::rc::Rc::new(|_| None),
    ));
    let _foreign_guard =
        runmat_runtime::user_functions::install_execution_source_resolver(Some(foreign))
            .expect("call-port scope");
    assert_eq!(resolve().function, 1);
}

#[test]
fn execution_snapshot_clone_is_shared_until_publication_mutates() {
    let original = registry();
    let mut publication = original.clone();
    assert!(std::rc::Rc::ptr_eq(&original.data, &publication.data));
    publication.insert_replacing_name(function(4, "new", 8));
    assert!(!std::rc::Rc::ptr_eq(&original.data, &publication.data));
    assert_eq!(original.resolve_name("new"), None);
    assert_eq!(publication.resolve_name("new"), Some(FunctionId(4)));
    let error = serde_json::to_value(&publication)
        .expect_err("live session source authority must not enter portable bytecode");
    assert!(error
        .to_string()
        .contains("session source authority cannot enter portable bytecode"));
}

#[test]
fn portable_registry_record_keeps_flat_wire_shape() {
    let function = FunctionBytecode {
        function: FunctionId(1),
        display_name: "portable".into(),
        source_id: Some(BytecodeSourceRef::Program(runmat_types::ProgramSourceId(0))),
        ..Default::default()
    };
    let registry = FunctionRegistry::new(HashMap::from([(function.function, function)]));
    let serialized = serde_json::to_value(&registry).expect("portable registry record");
    assert!(serialized.get("functions").is_some());
    assert!(serialized.get("data").is_none());
    let restored: FunctionRegistry =
        serde_json::from_value(serialized).expect("portable roundtrip");
    assert_eq!(restored.resolve_name("portable"), Some(FunctionId(1)));
    assert_eq!(
        restored.get(FunctionId(1)).unwrap().source_id,
        Some(BytecodeSourceRef::Program(runmat_types::ProgramSourceId(0)))
    );
}

#[test]
fn function_execution_uses_shared_registry_without_cloning_legacy_body_map() {
    let registry = registry();
    let function = registry.get(FunctionId(1)).unwrap();
    let bytecode = function.execution_bytecode(&registry);
    assert!(bytecode.bound_functions.is_empty());
    assert!(std::rc::Rc::ptr_eq(
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
    let removed = registry.remove_source(source_ref(5));
    assert_eq!(removed.len(), 2);
    for name in ["first_alias", "second_alias", "second"] {
        assert_eq!(registry.resolve_name(name), None);
    }
    for name in ["first", "rebound_alias", "unrelated"] {
        assert_eq!(registry.resolve_name(name), Some(FunctionId(3)));
    }
    assert_eq!(registry.functions.len(), 1);
    assert!(registry.functions_for_source(source_ref(5)).is_empty());
    assert_eq!(
        registry.functions_for_source(source_ref(6)),
        &[FunctionId(3)]
    );
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
    assert_eq!(
        registry.functions_for_source(source_ref(5)),
        &[FunctionId(2)]
    );
    assert_eq!(
        registry.functions_for_source(source_ref(7)),
        &[FunctionId(4)]
    );
}

#[test]
fn same_identity_refresh_preserves_aliases_and_updates_source_membership() {
    let mut registry = registry();
    registry.names.insert("live_alias".into(), FunctionId(1));
    registry.insert_replacing_name(function(1, "first", 7));
    assert_eq!(registry.resolve_name("first"), Some(FunctionId(1)));
    assert_eq!(registry.resolve_name("live_alias"), Some(FunctionId(1)));
    assert_eq!(
        registry.functions_for_source(source_ref(5)),
        &[FunctionId(2)]
    );
    assert_eq!(
        registry.functions_for_source(source_ref(7)),
        &[FunctionId(1)]
    );
    registry.remove(FunctionId(1)).unwrap();
    assert_eq!(registry.resolve_name("first"), None);
    assert_eq!(registry.resolve_name("live_alias"), None);
    assert!(registry.functions_for_source(source_ref(7)).is_empty());
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
    assert!(!registry.retired.contains_key(&FunctionId(1)));
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
    assert_eq!(old_body.source_id, Some(runtime_ref(5)));
    assert_eq!(new_body.source_id, Some(runtime_ref(7)));
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
fn changed_origin_clone_cannot_publish_under_retired_definition_lease() {
    let mut registry = registry();
    let old_origin = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("original product"),
        function: runmat_types::ProgramFunctionId(1),
    };
    let new_origin = runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("replacement product"),
        function: runmat_types::ProgramFunctionId(1),
    };
    registry
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .binding_origin = Some(old_origin);
    let old = registry.get(FunctionId(1)).unwrap().clone();
    let old_lease = runmat_value::FunctionDefinitionOwner::issue(&old.definition_owner, 1);
    registry.insert_replacing_name(FunctionBytecode {
        binding_origin: Some(new_origin),
        ..old
    });

    let current = registry.get(FunctionId(1)).unwrap();
    assert_eq!(current.binding_origin, Some(new_origin));
    assert!(!old_lease.matches(&current.definition_owner));
    assert_eq!(
        registry
            .get_owned(FunctionId(1), &old_lease)
            .and_then(|body| body.binding_origin),
        Some(old_origin)
    );
}

#[test]
fn changed_origin_preserves_distinct_preissued_incoming_owner() {
    let mut registry = registry();
    let old_lease = runmat_value::FunctionDefinitionOwner::issue(
        &registry.get(FunctionId(1)).unwrap().definition_owner,
        1,
    );
    let mut incoming = function(1, "first", 7);
    incoming.binding_origin = Some(runmat_execution::FunctionBindingOrigin {
        program: runmat_execution::Digest::sha256("independently issued product"),
        function: runmat_types::ProgramFunctionId(1),
    });
    let incoming_lease =
        runmat_value::FunctionDefinitionOwner::issue(&incoming.definition_owner, 1);
    registry.insert_replacing_name(incoming);

    assert!(registry.get_owned(FunctionId(1), &incoming_lease).is_some());
    assert_eq!(
        registry
            .get_owned(FunctionId(1), &old_lease)
            .unwrap()
            .source_id,
        Some(runtime_ref(5))
    );
}

#[test]
fn removed_then_reinserted_body_cannot_reuse_the_retained_owner() {
    let mut registry = registry();
    let old_owner = std::sync::Arc::clone(&registry.get(FunctionId(1)).unwrap().definition_owner);
    let old_lease = runmat_value::FunctionDefinitionOwner::issue(&old_owner, 1);
    let old = registry.remove(FunctionId(1)).expect("remove old body");
    registry.insert_replacing_name(FunctionBytecode {
        binding_origin: Some(runmat_execution::FunctionBindingOrigin {
            program: runmat_execution::Digest::sha256("new product after removal"),
            function: runmat_types::ProgramFunctionId(1),
        }),
        ..old
    });

    assert!(!old_lease.matches(&registry.get(FunctionId(1)).unwrap().definition_owner));
    assert!(registry.get_owned(FunctionId(1), &old_lease).is_some());
}

#[test]
fn copied_owner_cannot_cross_function_id_or_name_replacement() {
    let original = function(1, "first", 5);
    let old_owner = std::sync::Arc::clone(&original.definition_owner);
    let mut registry = FunctionRegistry::new(HashMap::from([(FunctionId(1), original.clone())]));
    let old_lease = runmat_value::FunctionDefinitionOwner::issue(&old_owner, 1);
    registry.insert_replacing_name(FunctionBytecode {
        function: FunctionId(4),
        binding_origin: Some(runmat_execution::FunctionBindingOrigin {
            program: runmat_execution::Digest::sha256("different function product"),
            function: runmat_types::ProgramFunctionId(4),
        }),
        ..original
    });

    assert_eq!(registry.resolve_name("first"), Some(FunctionId(4)));
    assert!(!old_lease.matches(&registry.get(FunctionId(4)).unwrap().definition_owner));
    assert!(registry.get_owned(FunctionId(1), &old_lease).is_some());

    let original = function(1, "first", 5);
    let mut distinct_names =
        FunctionRegistry::new(HashMap::from([(FunctionId(1), original.clone())]));
    distinct_names.insert_replacing_name(FunctionBytecode {
        function: FunctionId(5),
        display_name: "unrelated".into(),
        ..original
    });
    assert!(!std::sync::Arc::ptr_eq(
        &distinct_names.get(FunctionId(1)).unwrap().definition_owner,
        &distinct_names.get(FunctionId(5)).unwrap().definition_owner
    ));
}

#[test]
fn failed_candidate_can_retain_only_escaped_exact_bodies() {
    let mut published = registry();
    let candidate = FunctionRegistry::new(HashMap::from([
        (FunctionId(7), function(7, "escaped", 7)),
        (FunctionId(8), function(8, "unescaped", 8)),
    ]));
    let escaped = runmat_value::FunctionDefinitionOwner::issue(
        &candidate.get(FunctionId(7)).unwrap().definition_owner,
        7,
    );

    let programs = published.retain_live_owned_definitions_from(&candidate);

    assert!(published.get(FunctionId(7)).is_none());
    assert!(published.resolve_name("escaped").is_none());
    assert!(published.get_owned(FunctionId(7), &escaped).is_some());
    assert!(!published.retired.contains_key(&FunctionId(8)));
    assert!(programs.is_empty(), "fixture has no portable origins");
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
    registry.remove_source(source_ref(5));
    registry.insert_replacing_name(function(4, "first", 7));
    registry.insert_replacing_name(function(5, "second", 7));
    let old = registry
        .owned_view(FunctionId(1), &authority)
        .expect("old owned definition and helper");
    assert!(std::rc::Rc::ptr_eq(&old.data, &registry.data));
    assert!(old.get(FunctionId(1)).is_some());
    assert!(old.get(FunctionId(2)).is_some());
    assert!(old.static_binding(FunctionId(2)).is_some());
    let helper_binding = old.static_binding(FunctionId(2)).unwrap();
    let nested = old
        .owned_view(FunctionId(2), &helper_binding.authority)
        .expect("shared retained graph selects helper");
    assert!(std::rc::Rc::ptr_eq(
        old.static_overlay.as_ref().unwrap(),
        nested.static_overlay.as_ref().unwrap()
    ));
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
    assert!(old.current_view().get(FunctionId(1)).is_none());
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
        Some(runtime_ref(5))
    );
    assert_eq!(
        view.current_view().get(FunctionId(2)).unwrap().source_id,
        Some(runtime_ref(7))
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

#[test]
fn retained_execution_stack_uses_exact_old_bodies_in_both_directions() {
    let mut retained_registry = registry();
    retained_registry
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .instructions = vec![super::Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)];
    let (authority, _issuer) = local_authority(&retained_registry, FunctionId(1));

    let mut replacement_entry = function(1, "first", 7);
    replacement_entry.instructions =
        vec![super::Instr::CallBuiltinMulti("javaObject".into(), 1, 1)];
    retained_registry.insert_replacing_name(replacement_entry);
    retained_registry.insert_replacing_name(function(2, "second", 7));

    let retained = retained_registry
        .owned_view(FunctionId(1), &authority)
        .unwrap();
    assert_eq!(
        retained.execution_stack(FunctionId(1)),
        runmat_types::ExecutionStackRequirement::Any
    );
    assert_eq!(
        retained_registry.execution_stack(FunctionId(1)),
        runmat_types::ExecutionStackRequirement::Process
    );

    let mut inverse = registry();
    inverse
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .instructions = vec![super::Instr::CallBuiltinMulti("javaObject".into(), 1, 1)];
    let (foreign_authority, _foreign_issuer) = local_authority(&inverse, FunctionId(1));
    inverse.insert_replacing_name(function(1, "first", 8));
    let retained_foreign = inverse
        .owned_view(FunctionId(1), &foreign_authority)
        .unwrap();
    assert_eq!(
        retained_foreign.execution_stack(FunctionId(1)),
        runmat_types::ExecutionStackRequirement::Process
    );
    assert_eq!(
        inverse.execution_stack(FunctionId(1)),
        runmat_types::ExecutionStackRequirement::Any
    );
}

#[test]
fn retained_execution_stack_terminates_cycles_and_rejects_missing_targets() {
    let mut cyclic = registry();
    cyclic
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .instructions = vec![super::Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)];
    cyclic
        .functions
        .get_mut(&FunctionId(2))
        .unwrap()
        .instructions = vec![super::Instr::CallSemanticFunctionMulti(FunctionId(1), 0, 1)];
    let (cycle_authority, _cycle_issuer) = local_authority(&cyclic, FunctionId(1));
    cyclic.insert_replacing_name(function(1, "first", 7));
    cyclic.insert_replacing_name(function(2, "second", 7));
    let cycle = cyclic.owned_view(FunctionId(1), &cycle_authority).unwrap();
    assert_eq!(
        cycle.execution_stack(FunctionId(1)),
        runmat_types::ExecutionStackRequirement::Any
    );

    let mut missing = registry();
    missing
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .instructions = vec![super::Instr::CallSemanticFunctionMulti(
        FunctionId(99),
        0,
        1,
    )];
    let owner = std::sync::Arc::clone(&missing.functions[&FunctionId(1)].definition_owner);
    let issuer = std::sync::Arc::new(());
    let authority = runmat_value::BoundFunctionAuthority::local(
        std::sync::Arc::downgrade(&issuer),
        runmat_value::FunctionDefinitionOwner::issue(&owner, 1),
        Vec::new(),
        runmat_value::LocalExport::Ephemeral,
    );
    missing.insert_replacing_name(function(1, "first", 7));
    let missing = missing.owned_view(FunctionId(1), &authority).unwrap();
    assert_eq!(
        missing.execution_stack(FunctionId(1)),
        runmat_types::ExecutionStackRequirement::Process
    );
}

#[test]
fn retained_execution_stack_cache_is_owned_by_the_exact_overlay() {
    let mut registry = registry();
    let (authority, _issuer) = local_authority(&registry, FunctionId(1));
    registry.insert_replacing_name(function(1, "first", 7));
    let retained = registry.owned_view(FunctionId(1), &authority).unwrap();
    assert_eq!(
        retained.execution_stack(FunctionId(1)),
        runmat_types::ExecutionStackRequirement::Any
    );

    let retained_clone = retained.clone();
    assert!(std::rc::Rc::ptr_eq(
        retained.static_overlay.as_ref().unwrap(),
        retained_clone.static_overlay.as_ref().unwrap()
    ));
    let mut current = retained.current_view();
    let mut foreign = function(1, "first", 8);
    foreign.instructions = vec![super::Instr::CallBuiltinMulti("javaObject".into(), 1, 1)];
    current.insert_replacing_name(foreign);
    assert_eq!(
        current.execution_stack(FunctionId(1)),
        runmat_types::ExecutionStackRequirement::Process
    );
    assert_eq!(
        retained_clone.execution_stack(FunctionId(1)),
        runmat_types::ExecutionStackRequirement::Any
    );
}

#[test]
fn retained_execution_stack_does_not_borrow_published_method_targets() {
    let mut registry = registry();
    registry
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .instructions = vec![super::Instr::CallFunctionMulti {
        identity: runmat_types::CallableIdentity::Method(runmat_types::MethodId(
            "Example.pure".into(),
        )),
        fallback_policy: runmat_types::CallableFallbackPolicy::ObjectDispatch,
        arg_count: 0,
        out_count: 0,
    }];
    let (authority, _issuer) = local_authority(&registry, FunctionId(1));

    let mut published_registration = function(1, "first", 7);
    published_registration.instructions = vec![super::Instr::RegisterClass {
        name: "Example".into(),
        super_class: None,
        is_sealed: false,
        is_abstract: false,
        properties: Vec::new(),
        methods: vec![crate::bytecode::instr::BytecodeClassMethod {
            name: "pure".into(),
            function_name: "Example.pure".into(),
            role: runmat_types::ClassMethodRole::Ordinary,
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: runmat_types::MemberAccess::Public,
        }],
        enumerations: Vec::new(),
        events: Vec::new(),
    }];
    registry.insert_replacing_name(published_registration);
    registry.insert_replacing_name(function(2, "Example.pure", 7));

    let retained = registry.owned_view(FunctionId(1), &authority).unwrap();
    assert_eq!(
        retained.execution_stack(FunctionId(1)),
        runmat_types::ExecutionStackRequirement::Process,
        "published method metadata must not certify a retained dynamic target"
    );
}
