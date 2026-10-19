use super::*;
use std::{collections::HashMap, rc::Rc};

fn declaration(name: &str, parent: Option<&str>) -> CheckedClassRegistration {
    CheckedClassRegistration {
        class: RuntimeClass {
            name: ClassIdentity::new(name).expect("class"),
            parent: parent.map(|name| ClassIdentity::new(name).expect("parent")),
            properties: HashMap::new(),
            methods: HashMap::new(),
            events: HashMap::new(),
        },
        method_bindings: HashMap::new(),
        is_sealed: false,
        is_abstract: false,
        enumerations: Vec::new(),
    }
}

#[test]
fn open_orphan_does_not_poison_unrelated_closed_class() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _scope = runtime.enter();
    register_open_class_declarations(vec![declaration("OpenOrphan", Some("AbsentOpenParent"))])
        .expect("declaration is not executable admission");
    register_checked_classes(vec![declaration("ClosedUnrelated", None)])
        .expect("unrelated selected class remains executable");
    assert!(super::super::get_class(&ClassIdentity::new("ClosedUnrelated").unwrap()).is_some());
    assert!(
        register_checked_classes(vec![declaration("OpenOrphan", Some("AbsentOpenParent"))])
            .is_err()
    );
}

#[test]
fn closed_child_cannot_inherit_an_incomplete_memoized_open_parent() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _scope = runtime.enter();
    register_open_class_declarations(vec![declaration("OpenAncestor", Some("MissingAncestor"))])
        .expect("source declaration remains open");
    let generation = super::super::class_registry_generation();
    assert!(
        register_checked_classes(vec![declaration("ClosedDescendant", Some("OpenAncestor"),)])
            .is_err()
    );
    assert_eq!(super::super::class_registry_generation(), generation);
    assert!(super::super::get_class(&ClassIdentity::new("ClosedDescendant").unwrap()).is_none());
    register_checked_classes(vec![declaration("IndependentClosed", None)])
        .expect("unrelated closed product remains admissible");
}

#[test]
fn closing_open_orphan_into_cycle_rejects_atomically() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _scope = runtime.enter();
    register_open_class_declarations(vec![declaration("OpenCycleA", Some("OpenCycleB"))])
        .expect("unresolved declaration");
    let generation = super::super::class_registry_generation();
    assert!(
        register_open_class_declarations(vec![declaration("OpenCycleB", Some("OpenCycleA"))])
            .is_err()
    );
    assert_eq!(super::super::class_registry_generation(), generation);
    assert!(super::super::get_class(&ClassIdentity::new("OpenCycleB").unwrap()).is_none());
}

#[test]
fn later_sealed_parent_cannot_invalidate_an_open_child() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _scope = runtime.enter();
    register_open_class_declarations(vec![declaration(
        "OpenSealedChild",
        Some("LateSealedParent"),
    )])
    .expect("unresolved child declaration");
    let generation = super::super::class_registry_generation();
    let mut parent = declaration("LateSealedParent", None);
    parent.is_sealed = true;
    assert!(register_open_class_declarations(vec![parent]).is_err());
    assert_eq!(super::super::class_registry_generation(), generation);
    assert!(super::super::get_class(&ClassIdentity::new("LateSealedParent").unwrap()).is_none());
}
