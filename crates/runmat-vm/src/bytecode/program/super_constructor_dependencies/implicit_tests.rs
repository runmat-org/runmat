use std::collections::HashMap;

use runmat_execution::{Digest, FunctionBindingOrigin};
use runmat_hir::FunctionId;
use runmat_types::{
    ClassIdentity, ClassMethodOwner, ClassMethodRole, MethodName, ProgramFunctionId,
};

use super::{FunctionRegistry, Instr, SuperConstructorDependencyError, SuperConstructorIndex};
use crate::bytecode::FunctionBytecode;

fn origin(program: &str, function: u32) -> FunctionBindingOrigin {
    FunctionBindingOrigin {
        program: Digest::sha256(program),
        function: ProgramFunctionId(function),
    }
}

fn declaration(class: &ClassIdentity, parent: Option<ClassIdentity>) -> Instr {
    Instr::RegisterClass {
        name: class.clone(),
        super_class: parent,
        is_sealed: false,
        is_abstract: false,
        properties: Vec::new(),
        methods: Vec::new(),
        events: Vec::new(),
        enumerations: Vec::new(),
    }
}

#[test]
fn repeated_same_product_declarations_are_one_target_but_conflicts_remain_errors() {
    let base = ClassIdentity::new("Base").expect("base");
    let parent = ClassIdentity::new("handle").expect("parent");
    let body = |id, declared_parent| FunctionBytecode {
        function: FunctionId(id),
        binding_origin: Some(origin(
            "base-product",
            u32::try_from(id).expect("small test function ID"),
        )),
        instructions: vec![declaration(&base, declared_parent)],
        ..Default::default()
    };
    let first = body(2, Some(parent.clone()));
    let second = body(3, Some(parent));
    let registry = FunctionRegistry::new(HashMap::from([
        (FunctionId(2), first.clone()),
        (FunctionId(3), second),
    ]));
    assert!(SuperConstructorIndex::new(&registry)
        .declaration(&base, None)
        .expect("identical declarations share one product")
        .is_some());

    let conflicting = FunctionRegistry::new(HashMap::from([
        (FunctionId(2), first),
        (FunctionId(3), body(3, None)),
    ]));
    assert!(matches!(
        SuperConstructorIndex::new(&conflicting).declaration(&base, None),
        Err(SuperConstructorDependencyError::ConflictingDeclaration)
    ));
}

#[test]
fn separately_admitted_no_body_parent_retains_its_declaration() {
    let derived = ClassIdentity::new("Derived").expect("derived");
    let base = ClassIdentity::new("Base").expect("base");
    let caller = FunctionBytecode {
        function: FunctionId(1),
        binding_origin: Some(origin("derived", 1)),
        class_method_owner: Some(ClassMethodOwner {
            declaring_class: derived.clone(),
            method: MethodName::from("Derived"),
            is_static: false,
            role: ClassMethodRole::Constructor,
        }),
        instructions: vec![
            declaration(&derived, Some(base.clone())),
            Instr::CallSuperConstructorMulti {
                current_class: derived,
                super_class: base.clone(),
                arg_count: 0,
                out_count: 1,
            },
        ],
        ..Default::default()
    };
    let parent = FunctionBytecode {
        function: FunctionId(2),
        binding_origin: Some(origin("base", 2)),
        instructions: vec![declaration(&base, None)],
        ..Default::default()
    };
    let registry = FunctionRegistry::new(HashMap::from([
        (FunctionId(1), caller.clone()),
        (FunctionId(2), parent),
    ]));
    assert_eq!(
        registry
            .definition_dependency_owners(FunctionId(1))
            .unwrap()
            .len(),
        2
    );
    let implicit = registry
        .retained_implicit_super_constructor_targets_for(FunctionId(1))
        .expect("exact linked implicit target");
    assert_eq!(implicit.len(), 1);
    assert_eq!(implicit[0].parent_declaration.as_ref().unwrap().class, base);

    let missing = FunctionRegistry::new(HashMap::from([(FunctionId(1), caller)]));
    assert!(matches!(
        missing.retained_implicit_super_constructor_targets_for(FunctionId(1)),
        Err(SuperConstructorDependencyError::MissingDeclaration)
    ));
}

#[test]
fn intrinsic_implicit_parent_needs_no_user_declaration() {
    let derived = ClassIdentity::new("Derived").expect("derived");
    let handle = ClassIdentity::new("handle").expect("handle");
    let caller = FunctionBytecode {
        function: FunctionId(1),
        binding_origin: Some(origin("derived", 1)),
        class_method_owner: Some(ClassMethodOwner {
            declaring_class: derived.clone(),
            method: MethodName::from("Derived"),
            is_static: false,
            role: ClassMethodRole::Constructor,
        }),
        instructions: vec![
            declaration(&derived, Some(handle.clone())),
            Instr::CallSuperConstructorMulti {
                current_class: derived,
                super_class: handle,
                arg_count: 0,
                out_count: 1,
            },
        ],
        ..Default::default()
    };
    let registry = FunctionRegistry::new(HashMap::from([(FunctionId(1), caller)]));
    let implicit = registry
        .retained_implicit_super_constructor_targets_for(FunctionId(1))
        .expect("intrinsic parent");
    assert_eq!(implicit.len(), 1);
    assert!(implicit[0].parent_declaration.is_none());
}
