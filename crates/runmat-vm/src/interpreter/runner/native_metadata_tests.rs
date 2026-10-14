use super::*;
use crate::bytecode::program::{FunctionBytecode, FunctionRegistry};
use runmat_hir::FunctionId;

fn native_class(name: &str, parent: Option<&str>) -> Instr {
    Instr::RegisterClass {
        name: runmat_types::ClassIdentity::new(name).expect("class identity"),
        super_class: parent
            .map(|name| runmat_types::ClassIdentity::new(name).expect("superclass identity")),
        is_sealed: false,
        is_abstract: false,
        properties: Vec::new(),
        methods: Vec::new(),
        events: Vec::new(),
        enumerations: Vec::new(),
    }
}

fn bound_function(id: usize, instructions: Vec<Instr>) -> FunctionBytecode {
    FunctionBytecode {
        function: FunctionId(id),
        instructions,
        ..FunctionBytecode::default()
    }
}

#[test]
fn native_metadata_publishes_typed_events_and_rejects_conflicting_duplicate() {
    let mut parent = native_class("VmNativeEventParent", Some("handle"));
    let event = runmat_types::EventDeclaration {
        name: "Changed".into(),
        attributes: runmat_types::EventAttributes {
            is_hidden: true,
            listen_access: runmat_types::EventAccess::Public,
            notify_access: runmat_types::EventAccess::Protected,
        },
        declared_attributes: Vec::new(),
        span: Default::default(),
    };
    if let Instr::RegisterClass { events, .. } = &mut parent {
        events.push(event.clone());
    }
    let child = native_class("VmNativeEventChild", Some("VmNativeEventParent"));
    let bytecode = Bytecode::with_instructions(vec![child, parent.clone()], 0);
    prepare_native_execution_metadata(&bytecode).expect("typed event publication");
    let (found, owner) = runmat_runtime::class_registry::lookup_event(
        &runmat_types::ClassIdentity::from("VmNativeEventChild"),
        &"Changed".into(),
    )
    .expect("inherited event");
    assert_eq!(
        owner,
        runmat_types::ClassIdentity::from("VmNativeEventParent")
    );
    assert_eq!(found, event);

    let mut changed = parent;
    if let Instr::RegisterClass { events, .. } = &mut changed {
        events[0].attributes.is_hidden = false;
    }
    let mut duplicate = Bytecode::with_instructions(vec![changed], 0);
    duplicate.function_registry = FunctionRegistry::new(HashMap::from([(
        FunctionId(9),
        bound_function(9, bytecode.instructions.clone()),
    )]));
    let error = prepare_native_execution_metadata(&duplicate).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:ClassMetadataConflict"));
}

#[test]
fn native_metadata_finds_nonfirst_function_class_without_script_entrypoint() {
    let mut bytecode = Bytecode::empty();
    bytecode.function_registry = FunctionRegistry::new(HashMap::from([
        (FunctionId(1), bound_function(1, Vec::new())),
        (
            FunctionId(2),
            bound_function(2, vec![native_class("VmNativeSecondOnly", None)]),
        ),
    ]));
    bytecode.bound_functions.insert(
        FunctionId(3),
        bound_function(3, vec![native_class("VmNativeBoundOnly", None)]),
    );
    let runmat_runtime::class_registry::ClassRegistryRevision::Current(before) =
        runmat_runtime::class_registry::class_registry_generation()
    else {
        panic!("metadata publication must observe the current registry");
    };
    prepare_native_execution_metadata(&bytecode).expect("publish bound metadata");
    assert!(bytecode.instructions.is_empty());
    assert_eq!(
        runmat_runtime::class_registry::class_registry_generation(),
        runmat_runtime::class_registry::ClassRegistryRevision::Current(before + 2)
    );
    let identity = runmat_types::ClassIdentity::new("VmNativeSecondOnly").unwrap();
    assert!(runmat_runtime::class_registry::get_class(&identity).is_some());
    let bound_only = runmat_types::ClassIdentity::new("VmNativeBoundOnly").unwrap();
    assert!(runmat_runtime::class_registry::get_class(&bound_only).is_some());
}

#[test]
fn metadata_scan_uses_a_separate_admitted_registry_component() {
    let bytecode = Bytecode::empty();
    let registry = FunctionRegistry::new(HashMap::from([(
        FunctionId(17),
        bound_function(
            17,
            vec![native_class("VmAdmittedComponentClass", Some("handle"))],
        ),
    )]));
    let class = runmat_types::ClassIdentity::new("VmAdmittedComponentClass").unwrap();
    assert!(bytecode.function_registry.functions.is_empty());
    assert!(bytecode.bound_functions.is_empty());
    super::native_metadata::prepare_execution_metadata(
        &bytecode,
        registry
            .functions
            .iter()
            .map(|(id, function)| (id.0, 0, function)),
    )
    .expect("install class from the separate admitted registry");
    assert_eq!(
        runmat_runtime::class_registry::get_class(&class)
            .expect("admitted registry class")
            .parent,
        Some(runmat_types::standard::HANDLE.owned())
    );
}

#[test]
fn native_metadata_merges_partial_top_and_bound_classes_once_in_dependency_order() {
    let mut bytecode =
        Bytecode::with_instructions(vec![native_class("VmNativePartialTop", None)], 0);
    bytecode.function_registry = FunctionRegistry::new(HashMap::from([
        (
            FunctionId(1),
            bound_function(
                1,
                vec![
                    native_class("VmNativePartialChild", Some("VmNativePartialParent")),
                    native_class("VmNativePartialTop", None),
                ],
            ),
        ),
        (
            FunctionId(2),
            bound_function(2, vec![native_class("VmNativePartialParent", None)]),
        ),
    ]));
    let runmat_runtime::class_registry::ClassRegistryRevision::Current(before) =
        runmat_runtime::class_registry::class_registry_generation()
    else {
        panic!("metadata publication must observe the current registry");
    };
    prepare_native_execution_metadata(&bytecode).expect("publish merged metadata");
    assert_eq!(
        runmat_runtime::class_registry::class_registry_generation(),
        runmat_runtime::class_registry::ClassRegistryRevision::Current(before + 3),
        "duplicate top-level declaration must not be registered twice"
    );
    let child = runmat_types::ClassIdentity::new("VmNativePartialChild").unwrap();
    let parent = runmat_types::ClassIdentity::new("VmNativePartialParent").unwrap();
    assert_eq!(
        runmat_runtime::class_registry::get_class(&child)
            .expect("child registered")
            .parent,
        Some(parent.clone())
    );
    assert!(runmat_runtime::class_registry::get_class(&parent).is_some());
}

#[test]
fn native_metadata_rejects_conflicting_duplicate_before_publication() {
    let mut bytecode =
        Bytecode::with_instructions(vec![native_class("VmNativeConflictTop", None)], 0);
    let mut conflicting = native_class("VmNativeConflictTop", None);
    if let Instr::RegisterClass { is_sealed, .. } = &mut conflicting {
        *is_sealed = true;
    }
    bytecode.function_registry = FunctionRegistry::new(HashMap::from([(
        FunctionId(1),
        bound_function(1, vec![conflicting]),
    )]));
    let before = runmat_runtime::class_registry::class_registry_generation();
    let error = prepare_native_execution_metadata(&bytecode).expect_err("conflicting class");
    assert_eq!(error.identifier(), Some("RunMat:ClassMetadataConflict"));
    assert_eq!(
        runmat_runtime::class_registry::class_registry_generation(),
        before
    );
}

#[test]
fn native_metadata_rejects_superclass_cycle_before_publication() {
    let mut bytecode = Bytecode::empty();
    bytecode.function_registry = FunctionRegistry::new(HashMap::from([
        (
            FunctionId(1),
            bound_function(
                1,
                vec![native_class("VmNativeCycleA", Some("VmNativeCycleB"))],
            ),
        ),
        (
            FunctionId(2),
            bound_function(
                2,
                vec![native_class("VmNativeCycleB", Some("VmNativeCycleA"))],
            ),
        ),
    ]));
    let before = runmat_runtime::class_registry::class_registry_generation();
    let error = prepare_native_execution_metadata(&bytecode).expect_err("cycle");
    assert_eq!(error.identifier(), Some("RunMat:ClassInheritanceCycle"));
    assert_eq!(
        runmat_runtime::class_registry::class_registry_generation(),
        before
    );
}

#[test]
fn native_metadata_batch_does_not_publish_earlier_class_before_later_sealed_parent_failure() {
    let mut bytecode = Bytecode::empty();
    bytecode.function_registry = FunctionRegistry::new(HashMap::from([
        (
            FunctionId(1),
            bound_function(1, vec![native_class("VmNativeAtomicEarlier", None)]),
        ),
        (
            FunctionId(2),
            bound_function(
                2,
                vec![native_class(
                    "VmNativeAtomicLater",
                    Some("VmNativeAtomicSealed"),
                )],
            ),
        ),
        (
            FunctionId(3),
            bound_function(3, {
                let mut parent = native_class("VmNativeAtomicSealed", None);
                if let Instr::RegisterClass { is_sealed, .. } = &mut parent {
                    *is_sealed = true;
                }
                vec![parent]
            }),
        ),
    ]));
    let before = runmat_runtime::class_registry::class_registry_generation();
    let error = prepare_native_execution_metadata(&bytecode).expect_err("sealed parent");
    assert_eq!(error.identifier(), Some("RunMat:ClassSealed"));
    assert_eq!(
        runmat_runtime::class_registry::class_registry_generation(),
        before
    );
    assert!(
        runmat_runtime::class_registry::get_class(&runmat_types::ClassIdentity::from(
            "VmNativeAtomicEarlier"
        ))
        .is_none()
    );
}
