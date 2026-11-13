use super::*;
use crate::object::indexing::{
    ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use futures::executor::block_on;
use runmat_value::{CellArray, Tensor};
fn arguments(values: Vec<Value>) -> crate::call::BuiltinArguments {
    crate::builtins::common::test_support::builtin_arguments(values)
}
fn array(values: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(values, shape).unwrap())
}
fn cell(values: Vec<Value>) -> Value {
    let n = values.len();
    Value::Cell(CellArray::new(values, 1, n).unwrap())
}
fn paren(values: Vec<Value>) -> ObjectSubscript {
    ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
        components: runmat_value::ExactArrayStorage::shared_unbounded(
            values
                .into_iter()
                .map(ObjectIndexComponent::Value)
                .collect(),
        ),
    })
}
fn path(steps: Vec<ObjectSubscript>) -> ObjectSubscriptPath {
    ObjectSubscriptPath::new(steps).unwrap()
}
fn get(runtime: &RuntimeContext, base: &RootedValue, steps: Vec<ObjectSubscript>) -> RootedValue {
    block_on(crate::object::protocol::read_subscript_path_in_context(
        runtime,
        base.value().clone(),
        path(steps),
        None,
    ))
    .unwrap()
}
fn set(
    runtime: &RuntimeContext,
    base: &RootedValue,
    steps: Vec<ObjectSubscript>,
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    block_on(crate::object::protocol::assign_subscript_path_in_context(
        runtime,
        base.value().clone(),
        path(steps),
        vec![rhs],
        None,
    ))
}
fn numbers(value: &RootedValue) -> Vec<f64> {
    match value.value() {
        Value::Num(x) => vec![*x],
        Value::Tensor(x) => x.materialize_f64(),
        _ => panic!("real bound array"),
    }
}
#[test]
fn numeric_named_nd_grammar_and_class_shape_use_the_real_native_definition() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let foreign = crate::user_functions::test_support::exact_fixture_runtime();
    let _ambient = foreign.enter();
    let _foreign_count = crate::output_count::push_output_count(Some(2)).unwrap();
    for (dims, expected) in [
        (vec![], vec![1, 1]),
        (vec![Value::Num(3.)], vec![3, 1]),
        (
            vec![Value::Num(2.), Value::Num(3.), Value::Num(2.)],
            vec![2, 3, 2],
        ),
        (vec![array(vec![2., 3., 2.], vec![1, 3])], vec![2, 3, 2]),
        (
            vec![cell(vec![Value::from("a"), Value::from("b")])],
            vec![1, 2],
        ),
        (
            vec![Value::Cell(
                CellArray::new(vec![Value::from("a"), Value::from("b")], 2, 1).unwrap(),
            )],
            vec![2, 1],
        ),
        (
            vec![Value::StringArray(
                runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap(),
            )],
            vec![1, 2],
        ),
        (
            vec![cell(vec![
                cell(vec![Value::from("a"), Value::from("b")]),
                Value::Num(3.),
            ])],
            vec![2, 3],
        ),
    ] {
        let mut values = vec![Value::from("decision")];
        values.extend(dims);
        let output = block_on(registered(Some(runtime.clone()), arguments(values))).unwrap();
        assert_eq!(
            &*crate::builtins::common::shape::value_dimensions_borrowed(output.value()).unwrap(),
            &expected
        );
        let Value::Object(object) = output.value() else {
            panic!("native variable")
        };
        assert!(object.class_name().is(CLASS));
        assert!(is_reference(object));
        let fact = crate::value_fact::value_fact(output.value());
        assert_eq!(fact.alias, runmat_types::AliasFact::Identity);
        assert_eq!(fact.mutation, runmat_types::MutationFact::HandleSemantics);
    }
    let source = Tensor::new(vec![3., 5.], vec![1, 2]).unwrap();
    let pointer = source.as_f64_slice().unwrap().as_ptr();
    let semi = block_on(registered(
        Some(runtime.clone()),
        arguments(vec![
            Value::from("semi"),
            Value::Num(2.),
            Value::from("Type"),
            Value::from("semi-integer"),
            Value::from("LowerBound"),
            Value::Num(1.),
            Value::from("UpperBound"),
            Value::Tensor(source),
        ]),
    ))
    .unwrap();
    let Value::Object(object) = semi.value() else {
        panic!("native semi variable")
    };
    let mut cooperation = block_on(super::cooperation(Some(&runtime))).unwrap();
    let view = block_on(View::new(object, &mut cooperation)).unwrap();
    assert_eq!(
        view.definition().domain(),
        runmat_value::optimization_variable::OptimizationVariableDomain::SemiInteger
    );
    let runmat_value::optimization_variable::OptimizationBoundView::PerElement(values) = view
        .definition()
        .bound(runmat_value::optimization_variable::OptimizationBoundRole::Upper)
    else {
        panic!("original bound buffer")
    };
    assert_eq!(values.as_ptr(), pointer);
    for dimensions in [Value::Num(0.), Value::Num(1.5), Value::from("not_an_axis")] {
        assert!(block_on(registered(
            Some(runtime.clone()),
            arguments(vec![Value::from("x"), dimensions])
        ))
        .is_err());
    }
    assert!(block_on(registered(
        Some(runtime.clone()),
        arguments(vec![
            Value::from("x"),
            cell(vec![Value::from("a"), Value::from("a")])
        ])
    ))
    .is_err());
    assert!(block_on(registered(
        Some(runtime.clone()),
        arguments(vec![
            Value::from("x"),
            Value::from("Type"),
            Value::from("semi-integer")
        ])
    ))
    .is_err());
}

#[test]
fn reference_property_roles_actual_aliases_atomic_generation_and_gc_roots() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let original = block_on(registered(
            Some(runtime.clone()),
            arguments(vec![
                Value::from("x"),
                Value::Num(3.),
                Value::from("LowerBound"),
                Value::Num(0.),
            ]),
        ))
        .unwrap();
        let subset = get(
            &runtime,
            &original,
            vec![paren(vec![array(vec![1., 3.], vec![1, 2])])],
        );
        let Value::Object(object) = original.value() else {
            panic!("definition view")
        };
        let Some(runmat_value::optimization_variable::OptimizationVariablePayload::Reference(
            selection,
        )) = object.optimization_variable_payload()
        else {
            panic!("native reference")
        };
        let target = selection.definition();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&target, |_| ()).is_ok());
        let updated = set(
            &runtime,
            &original,
            vec![
                paren(vec![array(vec![1., 3.], vec![1, 2])]),
                ObjectSubscript::member("LowerBound"),
            ],
            Value::Num(2.),
        )
        .unwrap();
        assert_eq!(
            &*crate::builtins::common::shape::value_dimensions_borrowed(updated.value()).unwrap(),
            &[3, 1]
        );
        assert_eq!(
            numbers(&get(
                &runtime,
                &original,
                vec![ObjectSubscript::member("LowerBound")]
            )),
            vec![2., 0., 2.]
        );
        let error = set(
            &runtime,
            &subset,
            vec![ObjectSubscript::member("Type")],
            Value::from("integer"),
        )
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:PropertyReadOnly"));
        set(
            &runtime,
            &original,
            vec![ObjectSubscript::member("Type")],
            Value::from("integer"),
        )
        .unwrap();
        let kind = get(&runtime, &subset, vec![ObjectSubscript::member("Type")]);
        assert!(matches!(kind.value(),Value::String(value) if value=="integer"));
        let labels = cell(vec![
            cell(vec![Value::from("a"), Value::from("b"), Value::from("c")]),
            cell(vec![]),
        ]);
        set(
            &runtime,
            &original,
            vec![ObjectSubscript::member("IndexNames")],
            labels,
        )
        .unwrap();
        set(
            &runtime,
            &subset,
            vec![
                paren(vec![Value::from("c")]),
                ObjectSubscript::member("LowerBound"),
            ],
            Value::Num(5.),
        )
        .unwrap();
        assert_eq!(
            numbers(&get(
                &runtime,
                &original,
                vec![ObjectSubscript::member("LowerBound")]
            )),
            vec![2., 0., 5.]
        );
        // An identity selection remains a read-only Reference, not a way to
        // recover definition-level setter authority.
        let all = get(
            &runtime,
            &original,
            vec![ObjectSubscript::parentheses(
                ObjectIndexSelector::IndexValues {
                    components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                        ObjectIndexComponent::Colon,
                    ]),
                },
            )],
        );
        assert_eq!(
            set(
                &runtime,
                &all,
                vec![ObjectSubscript::member("IndexNames")],
                cell(vec![])
            )
            .unwrap_err()
            .identifier(),
            Some("RunMat:PropertyReadOnly")
        );
        let Value::Object(object) = original.value() else {
            unreachable!()
        };
        let mut cooperation = block_on(super::cooperation(Some(&runtime))).unwrap();
        let stale = block_on(View::new(object, &mut cooperation)).unwrap();
        set(
            &runtime,
            &original,
            vec![ObjectSubscript::member("UpperBound")],
            Value::Num(9.),
        )
        .unwrap();
        assert!(block_on(properties::set(
            &stale,
            runmat_builtins::OptimizationVariableProperty::LowerBound,
            &Value::Num(-2.),
            &mut cooperation
        ))
        .is_err());
        assert_eq!(
            numbers(&get(
                &runtime,
                &original,
                vec![ObjectSubscript::member("LowerBound")]
            )),
            vec![2., 0., 5.]
        );
        assert!(crate::execution::value_codec::encode_inline_value(original.value()).is_err());
        drop(stale);
        drop(all);
        drop(kind);
        drop(updated);
        drop(original);
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_with_value(&target, |_| ()).is_ok(),
            "surviving subset must retain the original definition"
        );
        drop(subset);
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_with_value(&target, |_| ()).is_err(),
            "no global variable-name registry may keep it alive"
        );
    });
}

#[test]
fn original_finite_and_cancel_admission_precedes_model_storage() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
        RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Finite(ManagedHostEffectLedger);
    impl RuntimeParallelService for Finite {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let state = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(state.free_bytes()),
                epoch: state.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(1024 * 1024);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger.clone()))),
        );
    let error = block_on(registered(
        Some(runtime.clone()),
        arguments(vec![Value::from("x"), Value::Num(3.)]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(ledger.snapshot().peak_reserved_bytes, 0);
    // Warmth belongs to this exact original session and builtin issuance,
    // not a same-named class or a registration in an ambient foreign context.
    let warm = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    {
        let _scope = warm.enter();
        registration::ensure();
    }
    let warm = warm.with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger.clone()))),
    );
    let original_cooperation = block_on(construction_cooperation(Some(&warm))).unwrap();
    let registration = registration::Fence::prepare(&original_cooperation).unwrap();
    let foreign = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    {
        let _scope = foreign.enter();
        registration::ensure();
    }
    let foreign = foreign.with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger.clone()))),
    );
    let foreign_cooperation = block_on(construction_cooperation(Some(&foreign))).unwrap();
    assert!(
        registration.check(&foreign_cooperation).is_err(),
        "same-named foreign registration is not the original session"
    );
    let output = block_on(registered(
        Some(warm.clone()),
        arguments(vec![
            Value::from("bounded"),
            cell(vec![Value::from("alpha"), Value::from("beta")]),
            Value::from("LowerBound"),
            array(vec![1., 2.], vec![1, 2]),
        ]),
    ))
    .unwrap();
    let Value::Object(object) = output.value() else {
        panic!("variable");
    };
    let Some(runmat_value::optimization_variable::OptimizationVariablePayload::Reference(
        reference,
    )) = object.optimization_variable_payload()
    else {
        panic!("reference");
    };
    let target = reference.definition();
    let owner = object.optimization_native_storage_identity().unwrap();
    let alias = RootedValue::in_context(output.value().clone(), &warm).unwrap();
    assert_eq!(
        match alias.value() {
            Value::Object(object) => object.optimization_native_storage_identity(),
            _ => None,
        },
        Some(owner)
    );
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(output);
    runmat_gc::gc_with_value(&target, |value| {
        let Value::Object(object) = value else {
            panic!("definition");
        };
        let Some(runmat_value::optimization_variable::OptimizationVariablePayload::Definition(
            data,
        )) = object.optimization_variable_payload()
        else {
            panic!("definition payload");
        };
        assert_eq!(data.name(), "bounded");
        assert_eq!(data.element_count(), 2);
        assert_eq!(data.index_names()[1].as_ref().unwrap()[0].as_str(), "alpha");
        assert_eq!(
            data.bound_value(
                runmat_value::optimization_variable::OptimizationBoundRole::Lower,
                1
            ),
            Some(2.)
        );
    })
    .unwrap();
    assert!(
        ledger.snapshot().live_claimed_bytes > 0,
        "alias retains real backing"
    );
    drop(alias);
    let virtual_array = block_on(registered(
        Some(warm.clone()),
        arguments(vec![Value::from("virtual_array"), Value::Num(1_000_000.)]),
    ))
    .unwrap();
    let Value::Object(virtual_object) = virtual_array.value() else {
        panic!("virtual variable");
    };
    let Some(runmat_value::optimization_variable::OptimizationVariablePayload::Reference(
        virtual_reference,
    )) = virtual_object.optimization_variable_payload()
    else {
        panic!("virtual reference");
    };
    assert_eq!(virtual_reference.element_count(), 1_000_000);
    assert!(
        virtual_reference.is_whole(),
        "no element-sized identity index is allocated"
    );
    drop(virtual_array);
    let tiny = ManagedHostEffectLedger::new(1);
    let denied = warm.clone().with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Finite(tiny.clone()))),
    );
    assert!(block_on(registered(
        Some(denied),
        arguments(vec![Value::from("denied"), Value::Num(2.)])
    ))
    .is_err());
    assert_eq!(
        tiny.snapshot().live_reserved_bytes,
        0,
        "denied backing is not published"
    );
    // The generic fence prevents later helpers from acquiring a replacement
    // epoch, including a synchronous allocation before another yield.
    let mut cooperation = block_on(construction_cooperation(Some(&warm))).unwrap();
    let original = ledger.snapshot();
    ledger.set_epoch_for_test(original.epoch + 1);
    let Err(error) = super::super::common::model_storage::array::<usize>(&cooperation, 1) else {
        panic!("stale synchronous allocation succeeded");
    };
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    assert!(block_on(cooperation.completed(0)).is_err());
    assert_eq!(
        ledger.snapshot().live_reserved_bytes,
        original.live_reserved_bytes
    );
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(source.token());
    runtime.set_error_namespace("VariableInvocation");
    source.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(registered(Some(runtime), arguments(vec![Value::from("x")]))).unwrap_err();
    assert_eq!(error.identifier(), Some("VariableInvocation:Cancelled"));
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
}

#[test]
fn operational_gc_failures_retain_the_original_typed_cause() {
    use std::error::Error;
    for source in [
        runmat_gc::GcError::OutOfMemory("definition storage".into()),
        runmat_gc::GcError::InvalidPointer("definition retired".into()),
        runmat_gc::GcError::RootRegistrationFailed("root storage".into()),
        runmat_gc::GcError::SyncError("owner lock".into()),
    ] {
        let expected = source.to_string();
        let error = gc_error("definition operation failed", source);
        assert_eq!(error.identifier(), Some("RunMat:optimvar:GcFailure"));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        let original = error
            .source()
            .unwrap()
            .downcast_ref::<runmat_gc::GcError>()
            .unwrap();
        assert_eq!(original.to_string(), expected);
    }
}

#[test]
fn functional_field_extensions_share_the_native_protocol_and_original_definition() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _scope = runtime.enter();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = block_on(registered(
        Some(runtime.clone()),
        arguments(vec![
            Value::from("x"),
            Value::Num(3.),
            Value::from("LowerBound"),
            Value::Num(0.),
        ]),
    ))
    .unwrap();
    let subset = get(
        &runtime,
        &x,
        vec![paren(vec![array(vec![1., 3.], vec![1, 2])])],
    );
    let updated = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "setfield",
        &[
            subset.value().clone(),
            Value::from("LowerBound"),
            Value::Num(4.),
        ],
    ))
    .unwrap();
    assert_eq!(
        crate::builtins::common::shape::value_dimensions_borrowed(updated.value())
            .unwrap()
            .as_slice(),
        &[2, 1]
    );
    let bounds = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "getfield",
        &[x.value().clone(), Value::from("LowerBound")],
    ))
    .unwrap();
    assert_eq!(numbers(&bounds), vec![4., 0., 4.]);
    // One complete nested assignment enters the native protocol, retaining
    // its original generation while canonical indexed assignment edits it.
    let _ = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "setfield",
        &[
            x.value().clone(),
            Value::from("UpperBound"),
            cell(vec![array(vec![1., 3.], vec![1, 2])]),
            Value::Num(9.),
        ],
    ))
    .unwrap();
    assert_eq!(
        numbers(&get(
            &runtime,
            &x,
            vec![ObjectSubscript::member("UpperBound")]
        )),
        vec![9., f64::INFINITY, 9.]
    );
    let last = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "getfield",
        &[
            x.value().clone(),
            Value::from("UpperBound"),
            cell(vec![Value::from("end")]),
        ],
    ))
    .unwrap();
    assert_eq!(numbers(&last), vec![9.]);
    let _updated_end = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "setfield",
        &[
            x.value().clone(),
            Value::from("UpperBound"),
            cell(vec![Value::from("end")]),
            Value::Num(7.),
        ],
    ))
    .unwrap();
    assert_eq!(
        numbers(&get(
            &runtime,
            &x,
            vec![ObjectSubscript::member("UpperBound")]
        )),
        vec![9., f64::INFINITY, 7.]
    );
    let mut ordinary = ObjectInstance::new("ordinary.test".to_string());
    ordinary.replace_property_unbounded("value".to_string(), Value::Num(7.));
    let ordinary = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "getfield",
        &[Value::Object(ordinary), Value::from("value")],
    ))
    .unwrap();
    assert_eq!(ordinary.value(), &Value::Num(7.));
    #[cfg(target_pointer_width = "64")]
    {
        // A virtual definition and zero expression need only small shape and
        // selected-offset metadata, even when the final offset exceeds u32.
        let extent = u32::MAX as usize + 2;
        let huge = block_on(registered(
            Some(runtime.clone()),
            arguments(vec![
                Value::from("wide"),
                Value::Num(extent as f64),
                Value::from("LowerBound"),
                Value::Num(0.),
            ]),
        ))
        .unwrap();
        let last = get(
            &runtime,
            &huge,
            vec![paren(vec![Value::Int(runmat_value::IntValue::U64(
                extent as u64,
            ))])],
        );
        let Value::Object(selected) = last.value() else {
            panic!("native reference")
        };
        let view = block_on(View::new(
            selected,
            &mut crate::builtins::math::optim::common::ScalarCooperation::for_builtin(
                runmat_builtins::OPTIMVAR_CATALOG_ENTRY.identity,
                Some(&runtime),
            ),
        ))
        .unwrap();
        assert_eq!(view.selection().element_count(), 1);
        assert_eq!(view.selection().definition_index(0), Some(extent - 1));
        assert_eq!(view.definition().element_count(), extent);
        assert_eq!(
            numbers(&get(
                &runtime,
                &last,
                vec![ObjectSubscript::member("LowerBound")]
            )),
            vec![0.]
        );
        let unassigned = block_on(crate::call_builtin_async_in_context(
            &runtime,
            "optimexpr",
            &[Value::Num(extent as f64), Value::Num(1.)],
        ))
        .unwrap();
        let missing = get(
            &runtime,
            &unassigned,
            vec![paren(vec![Value::Int(runmat_value::IntValue::U64(
                extent as u64,
            ))])],
        );
        let error = block_on(crate::call_builtin_async_in_context(
            &runtime,
            "evaluate",
            &[
                missing.value().clone(),
                Value::Struct(runmat_value::StructValue::new()),
            ],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:OptimModel:InvalidInput"));
        let zero = block_on(crate::call_builtin_async_in_context(
            &runtime,
            "zeros",
            &[
                array(vec![extent as f64, 1.], vec![1, 2]),
                Value::from("like"),
                huge.value().clone(),
            ],
        ))
        .unwrap();
        let selected_zero = get(
            &runtime,
            &zero,
            vec![paren(vec![Value::Int(runmat_value::IntValue::U64(
                extent as u64,
            ))])],
        );
        let evaluated = block_on(crate::call_builtin_async_in_context(
            &runtime,
            "evaluate",
            &[
                selected_zero.value().clone(),
                Value::Struct(runmat_value::StructValue::new()),
            ],
        ))
        .unwrap();
        assert_eq!(numbers(&evaluated), vec![0.]);
    }
    let unassigned = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "optimexpr",
        &[Value::Num(1.), Value::Num(3.)],
    ))
    .unwrap();
    // Constructor dimensions are positive; an empty indexed view has no
    // unassigned elements and keeps its remaining axis extent.
    let empty_unassigned = get(
        &runtime,
        &unassigned,
        vec![ObjectSubscript::parentheses(
            ObjectIndexSelector::IndexValues {
                components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                    ObjectIndexComponent::Value(array(Vec::new(), vec![0, 0])),
                    ObjectIndexComponent::Colon,
                ]),
            },
        )],
    );
    let empty_value = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "evaluate",
        &[
            empty_unassigned.value().clone(),
            Value::Struct(runmat_value::StructValue::new()),
        ],
    ))
    .unwrap();
    assert!(numbers(&empty_value).is_empty());
    assert_eq!(
        crate::builtins::common::shape::value_dimensions_borrowed(empty_value.value())
            .unwrap()
            .as_slice(),
        &[0, 3]
    );
    // Native payload presence cannot authorize a replaced nonbuiltin class.
    let class = crate::class_registry::get_class(&CLASS.owned()).unwrap();
    crate::class_registry::register_class(class);
    let error = block_on(crate::call_builtin_async_in_context(
        &runtime,
        "getfield",
        &[x.value().clone(), Value::from("LowerBound")],
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Object:NativeFieldProtocol")
    );
}
