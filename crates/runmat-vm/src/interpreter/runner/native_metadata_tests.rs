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
    let before = runmat_runtime::class_registry::class_registry_generation();
    prepare_native_execution_metadata(&bytecode).expect("publish bound metadata");
    assert!(bytecode.instructions.is_empty());
    assert_eq!(
        runmat_runtime::class_registry::class_registry_generation(),
        before + 2
    );
    let identity = runmat_types::ClassIdentity::new("VmNativeSecondOnly").unwrap();
    assert!(runmat_runtime::class_registry::get_class(&identity).is_some());
    let bound_only = runmat_types::ClassIdentity::new("VmNativeBoundOnly").unwrap();
    assert!(runmat_runtime::class_registry::get_class(&bound_only).is_some());
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
    let before = runmat_runtime::class_registry::class_registry_generation();
    prepare_native_execution_metadata(&bytecode).expect("publish merged metadata");
    assert_eq!(
        runmat_runtime::class_registry::class_registry_generation(),
        before + 3,
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
