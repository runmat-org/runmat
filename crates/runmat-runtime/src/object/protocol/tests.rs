use futures::executor::block_on;
use runmat_types::{ObjectIndexingContext, SequenceUse};
use runmat_value::{CellArray, StructValue, Tensor, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use super::*;
use crate::object::indexing::{ObjectSubscript, ObjectSubscriptPath};
use crate::sequence::{ResolveValueSequence, SequenceResolutionContext};

fn single(sequence: crate::execution::RootedValueSequence) -> crate::execution::RootedValue {
    sequence
        .try_map_owned(
            |sequence| {
                Ok(sequence
                    .resolve(
                        SequenceUse::RequireSingle,
                        SequenceResolutionContext::default(),
                    )
                    .expect("single sequence")
                    .try_into_first()
                    .expect("unique single-output storage")
                    .expect("one value"))
            },
            |error| error.into_runtime_error(),
        )
        .expect("root single output")
}

fn read(path: ObjectSubscriptPath, base: Value) -> crate::execution::RootedValue {
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
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![index]),
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
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                crate::object::indexing::ObjectIndexComponent::Colon,
            ]),
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
    let mut member = runmat_value::UnboundedStructValueBuilder::new();
    member.insert("payload", Value::Num(7.0));
    let single = CellArray::new_with_shape(vec![Value::Struct(member.finish())], vec![1, 1])
        .expect("single cell");
    assert_eq!(read(path, Value::Cell(single)), Value::Num(7.0));
}

#[test]
fn callable_selected_from_cell_uses_parentheses_as_one_invocation() {
    use crate::object::indexing::{ObjectIndexComponent, ObjectIndexSelector};

    let path = |arguments: Vec<ObjectIndexComponent>| {
        ObjectSubscriptPath::new(vec![
            ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
                components: runmat_value::ExactArrayStorage::shared_unbounded(vec![Value::Num(
                    1.0,
                )
                .into()]),
            }),
            ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
                components: runmat_value::ExactArrayStorage::shared_unbounded(arguments),
            }),
        ])
        .expect("cell callable path")
    };
    let cell = |value| Value::Cell(CellArray::new(vec![value], 1, 1).expect("one cell"));

    assert_eq!(
        read(
            path(vec![Value::Num(9.0).into()]),
            cell(Value::FunctionHandle("sqrt".into())),
        ),
        Value::Num(3.0)
    );
    assert_eq!(
        read(
            path(vec![Value::Num(9.0).into()]),
            cell(Value::Closure(runmat_value::Closure::new(
                "sqrt".into(),
                None,
                runmat_types::FunctionSourceRole::Unknown,
                None,
                Vec::new(),
                None,
            ))),
        ),
        Value::Num(3.0)
    );

    let indexed_result = ObjectSubscriptPath::new(vec![
        ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                Value::Num(1.0).into()
            ]),
        }),
        ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![Value::Tensor(
                Tensor::new(vec![1.0, 4.0], vec![1, 2]).expect("input tensor"),
            )
            .into()]),
        }),
        ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                Value::Num(2.0).into()
            ]),
        }),
    ])
    .expect("call result index path");
    assert_eq!(
        read(indexed_result, cell(Value::FunctionHandle("deal".into()))),
        Value::Num(4.0)
    );

    let sequence = block_on(read_subscript_path_sequence_with_access(
        cell(Value::FunctionHandle("deal".into())),
        path(vec![Value::Num(7.0).into()]),
        SubscriptReadRequest {
            sequence_use: SequenceUse::SelectPrefix { count: 2 },
            sequence_context: SequenceResolutionContext::default(),
            indexing_context: ObjectIndexingContext::Expression,
            access: ObjectAccessContext::default(),
        },
    ))
    .expect("selected callable output count");
    let values = sequence.with_owned_sequence(|sequence| {
        sequence.resolve(
            SequenceUse::SelectPrefix { count: 2 },
            SequenceResolutionContext::default(),
        )
    });
    assert_eq!(
        values.expect("two outputs").as_slice(),
        &[Value::Num(7.0), Value::Num(7.0)]
    );

    let colon = block_on(read_subscript_path_with_access(
        cell(Value::FunctionHandle("sqrt".into())),
        path(vec![ObjectIndexComponent::Colon]),
        ObjectAccessContext::default(),
    ))
    .expect_err("colon is an index selector, not a callable argument");
    assert_eq!(
        colon.identifier(),
        Some("RunMat:UnsupportedFunctionHandleSelector")
    );

    let unknown_arity = block_on(read_subscript_path_sequence_with_access(
        cell(Value::FunctionHandle("sqrt".into())),
        path(vec![Value::Num(9.0).into()]),
        SubscriptReadRequest {
            sequence_use: SequenceUse::ExpandAll,
            sequence_context: SequenceResolutionContext::default(),
            indexing_context: ObjectIndexingContext::Expression,
            access: ObjectAccessContext::default(),
        },
    ))
    .expect_err("expanded invocation requires a known output count");
    assert_eq!(
        unknown_arity.identifier(),
        Some("RunMat:SequenceOutputContextUnavailable")
    );
}

#[test]
fn selected_symbolic_and_foreign_parentheses_keep_their_existing_owners() {
    use crate::object::indexing::{ObjectIndexComponent, ObjectIndexSelector};
    use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
    use runmat_value::{ForeignRef, ForeignResourceKey, SymbolicExpr};

    let path = |components: Vec<ObjectIndexComponent>| {
        ObjectSubscriptPath::new(vec![
            ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
                components: runmat_value::ExactArrayStorage::shared_unbounded(vec![Value::Num(
                    1.0,
                )
                .into()]),
            }),
            ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
                components: components.into(),
            }),
        ])
        .expect("selected item path")
    };
    let cell = |value| Value::Cell(CellArray::new(vec![value], 1, 1).expect("one cell"));
    let symbolic = || {
        cell(Value::Symbolic(SymbolicExpr::function_reference(
            "f",
            vec!["x".into()],
        )))
    };
    assert_eq!(
        read(path(vec![Value::Num(2.0).into()]), symbolic()),
        Value::Symbolic(SymbolicExpr::function_call(
            "f",
            vec![SymbolicExpr::constant(2.0)],
        ))
    );
    let colon = block_on(read_subscript_path_with_access(
        symbolic(),
        path(vec![ObjectIndexComponent::Colon]),
        ObjectAccessContext::default(),
    ))
    .expect_err("symbolic call rejects colon");
    assert_eq!(
        colon.identifier(),
        Some("RunMat:UnsupportedSymbolicFunctionSelector")
    );
    let arity = block_on(read_subscript_path_with_access(
        symbolic(),
        path(Vec::new()),
        ObjectAccessContext::default(),
    ))
    .expect_err("symbolic function arity remains checked");
    assert_eq!(arity.identifier(), Some("RunMat:SymbolicFunctionArity"));

    let reference = ForeignRef::detached(
        ForeignResourceKey {
            host_identity: "missing-foreign-host".into(),
            handle: 1,
            generation: 1,
        },
        ForeignTypeIdentity {
            family: "python".into(),
            name: "object".into(),
            version: 1,
        },
        ForeignOwnership::Borrowed,
        ForeignAffinity::AnyThread,
        ForeignLifetime::Session,
    );
    let direct = block_on(crate::foreign::index_foreign_resource(
        reference.clone(),
        vec![Value::Num(1.0)],
    ))
    .expect_err("detached foreign owner is unavailable");
    let chained = block_on(read_subscript_path_with_access(
        cell(Value::Foreign(reference)),
        path(vec![Value::Num(1.0).into()]),
        ObjectAccessContext::default(),
    ))
    .expect_err("chained foreign read reaches the same owner");
    assert_eq!(chained.identifier(), direct.identifier());
    assert_eq!(chained.message(), direct.message());
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
            Value::Composite(Box::new(handle.clone()).into()),
            brace_path(selector.into()),
            SubscriptReadRequest::require_single(ObjectAccessContext::default()),
        ))
        .expect_err("Composite access must retain exact selector and selected service checks");
        assert_eq!(error.identifier(), Some(expected));
    }
}

#[test]
fn typed_dotted_invoke_distinguishes_callable_member_from_member_indexing() {
    let mut callable = runmat_value::UnboundedStructValueBuilder::new();
    callable.insert("operation", Value::FunctionHandle("sqrt".into()));
    let path = ObjectSubscriptPath::new(
        ObjectSubscript::dotted_invoke("operation", vec![Value::Num(9.0)]).to_vec(),
    )
    .expect("dotted invoke path");
    assert_eq!(
        read(path, Value::Struct(callable.finish())),
        Value::Num(3.0)
    );

    let mut indexed = runmat_value::UnboundedStructValueBuilder::new();
    indexed.insert(
        "samples",
        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).expect("tensor")),
    );
    let path = ObjectSubscriptPath::new(
        ObjectSubscript::dotted_invoke("samples", vec![Value::Num(2.0)]).to_vec(),
    )
    .expect("dotted invoke path");
    assert_eq!(
        read(path, Value::Struct(indexed.finish())),
        Value::Num(20.0)
    );
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
    )))
    .expect("call-port scope");
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
        })))
        .expect("call-port scope");
    let class = runmat_types::ClassIdentity::from("OwnedSuffixSession");
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
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
                role: runmat_types::ClassMethodRole::Ordinary,
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
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(7.0))
            })
        },
    )))
    .expect("call-port scope");
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
            crate::execution::RootedValue::new(child).map_err(|error| error.into_runtime_error())
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
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(5.0))
            })
        },
    )))
    .expect("call-port scope");
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
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                Value::Num(1.0).into()
            ]),
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
    let mut indexed = runmat_value::UnboundedStructValueBuilder::new();
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
        read(path, Value::Struct(indexed.finish())),
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
    let mut callable = runmat_value::UnboundedStructValueBuilder::new();
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
        Value::Struct(callable.finish()),
        path,
        ObjectAccessContext::default(),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:InvalidDottedInvokeArgument")
    );
}
