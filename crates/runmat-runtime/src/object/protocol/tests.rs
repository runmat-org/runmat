use futures::executor::block_on;
use runmat_types::{ObjectIndexingContext, SequenceUse};
use runmat_value::{CellArray, StructValue, Tensor, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use super::*;
use crate::object::indexing::{ObjectSubscript, ObjectSubscriptPath};
use crate::sequence::{ResolveValueSequence, SequenceResolutionContext, ValueSequence};

fn single(sequence: ValueSequence) -> Value {
    sequence
        .resolve(
            SequenceUse::RequireSingle,
            SequenceResolutionContext::default(),
        )
        .expect("single sequence")
        .pop()
        .expect("one value")
}

fn read(path: ObjectSubscriptPath, base: Value) -> Value {
    single(
        block_on(read_subscript_path_sequence_with_access(
            base,
            path,
            SubscriptReadRequest {
                sequence_use: SequenceUse::RequireSingle,
                sequence_context: SequenceResolutionContext::default(),
                indexing_context: ObjectIndexingContext::Expression,
                access: ObjectAccessContext::default(),
            },
        ))
        .expect("path read"),
    )
}

fn brace_path(index: crate::object::indexing::ObjectIndexComponent) -> ObjectSubscriptPath {
    ObjectSubscriptPath::new(vec![ObjectSubscript::braces(
        crate::object::indexing::ObjectIndexSelector::IndexValues {
            components: vec![index],
        },
    )])
    .expect("brace path")
}

#[test]
fn brace_read_and_argument_expansion_keep_distinct_noncell_errors() {
    let path = brace_path(Value::Num(1.0).into());
    for base in [Value::Num(10.0), Value::ClassRef("Example".into())] {
        let error = block_on(read_subscript_path_with_access(
            base,
            path.clone(),
            ObjectAccessContext::default(),
        ))
        .expect_err("non-cell brace read");
        assert_eq!(error.identifier(), Some("RunMat:CellIndexingOnNonCell"));
    }
    let callable = block_on(read_subscript_path_with_access(
        Value::FunctionHandle("sin".into()),
        path,
        ObjectAccessContext::default(),
    ))
    .expect_err("callable brace selector");
    assert_eq!(
        callable.identifier(),
        Some("RunMat:UnsupportedFunctionHandleSelector")
    );
    let expansion = block_on(crate::call::arguments::expand_brace_values(
        Value::Num(10.0),
        &[Value::Num(1.0)],
        None,
    ))
    .expect_err("argument expansion remains separate");
    assert_eq!(
        expansion.identifier(),
        Some("RunMat:CellExpansionOnNonCell")
    );
}

#[test]
fn intermediate_brace_read_requires_exactly_one_cell_value() {
    let path = ObjectSubscriptPath::new(vec![
        ObjectSubscript::braces(crate::object::indexing::ObjectIndexSelector::IndexValues {
            components: vec![crate::object::indexing::ObjectIndexComponent::Colon],
        }),
        ObjectSubscript::member("payload"),
    ])
    .expect("chained brace path");
    for values in [vec![], vec![Value::Num(1.0), Value::Num(2.0)]] {
        let length = values.len();
        let cell = CellArray::new_with_shape(values, vec![1, length]).expect("cell cardinality");
        let error = block_on(read_subscript_path_with_access(
            Value::Cell(cell),
            path.clone(),
            ObjectAccessContext::default(),
        ))
        .expect_err("zero or many values cannot supply a chained receiver");
        assert_eq!(
            error.identifier(),
            Some("RunMat:CommaSeparatedListRequiresSingleValue")
        );
    }
    let mut member = StructValue::new();
    member.insert("payload", Value::Num(7.0));
    let single =
        CellArray::new_with_shape(vec![Value::Struct(member)], vec![1, 1]).expect("single cell");
    assert_eq!(read(path, Value::Cell(single)), Value::Num(7.0));
}

#[test]
fn composite_brace_read_uses_exact_selected_runtime_lease_and_lab_selector() {
    use crate::execution::RuntimeExecutionServices;
    use runmat_execution::{CompositeHandle, CompositeId, GangHandle, GangId, PoolRequest};
    use runmat_types::{
        LabCount, NumericClass, NumericDomain, NumericFact, ParallelRegionId, ProgramFunctionId,
        RegionId, ValueFact, ValueKindFact,
    };

    let execution = std::rc::Rc::new(crate::execution::RuntimeExecutionService::new());
    let pool = execution
        .ensure_pool(PoolRequest::automatic())
        .expect("selected pool");
    let runtime = crate::context::RuntimeContext::new(execution);
    let handle = CompositeHandle {
        id: CompositeId::derive(&[b"brace-read-composite"]),
        owner_region: ParallelRegionId(RegionId {
            function: ProgramFunctionId(1),
            ordinal: 1,
        }),
        scope_id: pool.handle.scope_id,
        generation: 1,
        gang: GangHandle {
            id: GangId::derive(&[b"brace-read-gang"]),
            scope_id: pool.handle.scope_id,
            generation: 1,
            pool: pool.handle,
            labs: LabCount(1),
        },
        value: ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    };
    for (selector, expected) in [
        (Value::Num(2.0), "RunMat:CompositeIndexOutOfBounds"),
        (Value::Num(1.0), "RunMat:RuntimeCapabilityUnavailable"),
    ] {
        let error = block_on(read_subscript_path_sequence_with_access_in_context(
            &runtime,
            Value::Composite(Box::new(handle.clone())),
            brace_path(selector.into()),
            SubscriptReadRequest::require_single(ObjectAccessContext::default()),
        ))
        .expect_err("Composite access must retain exact selector and selected service checks");
        assert_eq!(error.identifier(), Some(expected));
    }
}

#[test]
fn typed_dotted_invoke_distinguishes_callable_member_from_member_indexing() {
    let mut callable = StructValue::new();
    callable.insert("operation", Value::FunctionHandle("sqrt".into()));
    let path = ObjectSubscriptPath::new(
        ObjectSubscript::dotted_invoke("operation", vec![Value::Num(9.0)]).to_vec(),
    )
    .expect("dotted invoke path");
    assert_eq!(read(path, Value::Struct(callable)), Value::Num(3.0));

    let mut indexed = StructValue::new();
    indexed.insert(
        "samples",
        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).expect("tensor")),
    );
    let path = ObjectSubscriptPath::new(
        ObjectSubscript::dotted_invoke("samples", vec![Value::Num(2.0)]).to_vec(),
    )
    .expect("dotted invoke path");
    assert_eq!(read(path, Value::Struct(indexed)), Value::Num(20.0));
}

#[test]
fn semantic_fallback_failure_does_not_replay_the_invocation() {
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _activation = runtime.enter();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, _, _| {
            assert_eq!(function, 937);
            observed.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Err(crate::runtime_error::semantic_error(
                    "UndefinedFunction",
                    "the invoked function itself failed",
                ))
            })
        },
    )));
    let path = ObjectSubscriptPath::new(
        ObjectSubscript::dotted_invoke_components_with_fallback(
            "bump",
            vec![Value::Num(3.0).into()],
            Some(runmat_types::CallableIdentity::BoundFunction(
                runmat_types::FunctionId(937),
            )),
        )
        .to_vec(),
    )
    .expect("typed dotted invocation");
    let error = block_on(read_subscript_path_sequence_with_access(
        Value::Object(runmat_value::ObjectInstance::new("NoMethodFixture")),
        path,
        SubscriptReadRequest::require_single(ObjectAccessContext::default()),
    ))
    .expect_err("callee failure must propagate");
    assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn owned_path_suffix_uses_selected_session_after_first_step() {
    let wrong = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _wrong_activation = wrong.enter();
    let selected = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let expected_session = selected.session_lease();
    let selected_setup = selected.enter();
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "OwnedSuffixSession.subsref").then_some(943)
        })));
    let class = runmat_types::ClassIdentity::from("OwnedSuffixSession");
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: class.clone(),
        parent: None,
        properties: Default::default(),
        methods: std::collections::HashMap::from([(
            runmat_types::MethodName::from("subsref"),
            crate::class_registry::RuntimeMethod {
                name: "subsref".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: runmat_types::MemberAccess::Public,
                function_name: "OwnedSuffixSession.subsref".into(),
                implicit_class_argument: None,
            },
        )]),
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, _arguments, requested| {
            assert_eq!(function, 943);
            assert_eq!(requested, 1);
            let active = crate::context::legacy::active().expect("selected suffix scope");
            assert!(std::sync::Weak::ptr_eq(
                &active.session_lease(),
                &expected_session
            ));
            observed.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                ValueSequence::single(Value::Num(7.0))
                    .map_err(crate::sequence::sequence_error_to_runtime)
            })
        },
    )));
    drop(selected_setup);
    let path = ObjectSubscriptPath::new(vec![
        ObjectSubscript::member("child"),
        ObjectSubscript::member("p"),
    ])
    .expect("owned nested path");
    let child = Value::Object(runmat_value::ObjectInstance::new(class));
    let result = block_on(execute_owned_subsref_in_context(
        &selected,
        Value::Struct(StructValue::new()),
        path,
        move |_base, first| async move {
            assert_eq!(
                first.kind(),
                crate::object::indexing::ObjectIndexKind::Member
            );
            Ok(child)
        },
        None,
    ))
    .expect("owned suffix read");
    assert_eq!(result, Value::Num(7.0));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn object_array_without_member_keeps_whole_array_receiver_for_semantic_fallback() {
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _activation = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, arguments, requested| {
            assert_eq!(function, 938);
            assert_eq!(requested, 1);
            assert_eq!(arguments.len(), 2);
            assert!(matches!(&arguments[0], Value::ObjectArray(array) if array.len() == 2));
            Box::pin(async {
                crate::sequence::ValueSequence::single(Value::Num(5.0))
                    .map_err(crate::sequence::sequence_error_to_runtime)
            })
        },
    )));
    let array = runmat_value::ObjectArray::from_objects(
        "ArrayMethodFixture",
        vec![
            runmat_value::ObjectInstance::new("ArrayMethodFixture"),
            runmat_value::ObjectInstance::new("ArrayMethodFixture"),
        ],
        vec![1, 2],
    )
    .expect("homogeneous objects");
    let path = ObjectSubscriptPath::new(
        ObjectSubscript::dotted_invoke_components_with_fallback(
            "bump",
            vec![Value::Num(3.0).into()],
            Some(runmat_types::CallableIdentity::BoundFunction(
                runmat_types::FunctionId(938),
            )),
        )
        .to_vec(),
    )
    .expect("typed dotted invocation");
    assert_eq!(read(path, Value::ObjectArray(array)), Value::Num(5.0));
}

#[test]
fn ordinary_adjacent_member_and_parentheses_are_not_method_provenance() {
    let path = ObjectSubscriptPath::new(vec![
        ObjectSubscript::member("samples"),
        ObjectSubscript::parentheses(crate::object::indexing::ObjectIndexSelector::IndexValues {
            components: vec![Value::Num(1.0).into()],
        }),
    ])
    .expect("ordinary path");
    assert!(path
        .steps()
        .iter()
        .all(|step| step.origin() == crate::object::indexing::ObjectSubscriptOrigin::Ordinary));
}

#[test]
fn dotted_property_colon_is_an_index_selector_not_a_call_argument() {
    let mut indexed = StructValue::new();
    indexed.insert(
        "samples",
        Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap()),
    );
    let path = ObjectSubscriptPath::new(
        ObjectSubscript::dotted_invoke_components(
            "samples",
            vec![crate::object::indexing::ObjectIndexComponent::Colon],
        )
        .to_vec(),
    )
    .unwrap();
    assert_eq!(
        read(path, Value::Struct(indexed)),
        Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![4, 1]).unwrap())
    );
}

#[test]
fn resolved_callable_still_rejects_a_standalone_colon_argument() {
    let context = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    context.set_error_namespace("RunMat");
    let _scope = crate::context::RuntimeContextGuard::enter(context);
    let mut callable = StructValue::new();
    callable.insert("operation", Value::FunctionHandle("sqrt".into()));
    let path = ObjectSubscriptPath::new(
        ObjectSubscript::dotted_invoke_components(
            "operation",
            vec![crate::object::indexing::ObjectIndexComponent::Colon],
        )
        .to_vec(),
    )
    .unwrap();
    let error = block_on(read_subscript_path_with_access(
        Value::Struct(callable),
        path,
        ObjectAccessContext::default(),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:InvalidDottedInvokeArgument")
    );
}
