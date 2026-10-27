use super::*;
use crate::FunctionRegistry;
use runmat_runtime::execution::value_codec::{GraphCaptureProof, GraphNestedZeroCopy};
use runmat_types::{
    ClassIdentity, ClassMethodOwner, ClassMethodRole, FunctionId, FunctionSourceRole, MethodName,
};

fn compiled_nested(source: &str) -> (FunctionRegistry, FunctionId) {
    let ast = runmat_parser::parse(source).expect("parse nested program");
    let hir =
        runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower nested HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower nested MIR");
    let bodies = crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir)
        .expect("compile nested bytecode");
    let nested = bodies
        .values()
        .find(|body| body.source_role == FunctionSourceRole::Nested)
        .expect("nested compiled body")
        .function;
    (FunctionRegistry::new(bodies), nested)
}

#[test]
fn compiled_capture_free_nested_body_needs_no_activation() {
    let (registry, nested) = compiled_nested(
        "function h = outer()\nfunction y = inner(x)\ny = x + 1;\nend\nh = @inner;\nend\n",
    );
    let body = registry.get(nested).unwrap();
    assert!(body.capture_slots.is_empty());
    assert!(body.inherited_storage_slots.is_empty());
    let parent = nearest_activation_parent(
        body.lexical_parent,
        &registry,
        GraphProgramUnitId(0),
        None,
        &[],
    )
    .unwrap();
    assert!(parent.is_none());
    assert!(matches!(
        capture_proof(body, parent),
        GraphCaptureProof::NestedShared {
            vm_copied: 0,
            zero_copy: GraphNestedZeroCopy::CanonicalStorage,
        }
    ));
}

#[test]
fn compiled_canonical_storage_only_nested_body_needs_no_activation() {
    let (registry, nested) = compiled_nested(
        "function h = outer()\nglobal shared\nfunction y = inner()\ny = shared;\nend\nh = @inner;\nend\n",
    );
    let body = registry.get(nested).unwrap();
    assert!(!body.inherited_storage_slots.is_empty());
    assert!(body.capture_slots.iter().all(|slot| body
        .inherited_storage_slots
        .iter()
        .any(|storage| storage.slot == *slot)));
    let parent = nearest_activation_parent(
        body.lexical_parent,
        &registry,
        GraphProgramUnitId(0),
        None,
        &[],
    )
    .unwrap();
    assert!(parent.is_none());
    assert!(matches!(
        capture_proof(body, parent),
        GraphCaptureProof::NestedShared {
            zero_copy: GraphNestedZeroCopy::CanonicalStorage,
            ..
        }
    ));
    let mut mixed = body.clone();
    mixed.capture_slots.push(usize::MAX);
    assert!(matches!(
        capture_proof(&mixed, parent),
        GraphCaptureProof::NestedShared {
            zero_copy: GraphNestedZeroCopy::RequiresCopies,
            ..
        }
    ));
}

#[test]
fn typed_class_method_role_selects_public_constructor_handle_metadata() {
    let class = ClassIdentity::new("pkg.RevisionDerived").unwrap();
    let mut body = crate::FunctionBytecode {
        display_name: "pkg.RevisionDerived.RevisionDerived".into(),
        source_role: FunctionSourceRole::Method,
        class_method_owner: Some(ClassMethodOwner {
            declaring_class: class.clone(),
            method: MethodName::from("RevisionDerived"),
            is_static: false,
            role: ClassMethodRole::Constructor,
        }),
        ..Default::default()
    };
    assert_eq!(
        super::facts::graph_callable_name(&body),
        "pkg.RevisionDerived"
    );
    assert!(matches!(
        capture_proof(&body, None),
        GraphCaptureProof::Exact(0)
    ));

    body.class_method_owner.as_mut().unwrap().role = ClassMethodRole::Ordinary;
    assert_eq!(
        super::facts::graph_callable_name(&body),
        "pkg.RevisionDerived.RevisionDerived"
    );
    assert!(matches!(
        capture_proof(&body, None),
        GraphCaptureProof::InstanceMethodReceiver
    ));

    body.class_method_owner.as_mut().unwrap().is_static = true;
    assert!(matches!(
        capture_proof(&body, None),
        GraphCaptureProof::Exact(0)
    ));
}
