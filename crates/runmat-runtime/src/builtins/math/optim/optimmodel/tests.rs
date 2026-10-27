use super::*;
use crate::object::indexing::{
    ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use futures::executor::block_on;
fn array(values: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(runmat_value::Tensor::new(values, shape).unwrap())
}
async fn call(runtime: &RuntimeContext, name: &str, args: &[Value]) -> RootedValue {
    crate::call_builtin_async_in_context(runtime, name, args)
        .await
        .unwrap()
}
fn path(steps: Vec<ObjectSubscript>) -> ObjectSubscriptPath {
    ObjectSubscriptPath::new(steps).unwrap()
}
fn paren(index: f64) -> ObjectSubscript {
    ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
        components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
            ObjectIndexComponent::Value(Value::Num(index)),
        ]),
    })
}
async fn read(
    runtime: &RuntimeContext,
    value: &RootedValue,
    steps: Vec<ObjectSubscript>,
) -> RootedValue {
    crate::object::protocol::read_subscript_path_in_context(
        runtime,
        value.value().clone(),
        path(steps),
        None,
    )
    .await
    .unwrap()
}
async fn assign(
    runtime: &RuntimeContext,
    value: &RootedValue,
    steps: Vec<ObjectSubscript>,
    rhs: Value,
) -> RootedValue {
    crate::object::protocol::assign_subscript_path_in_context(
        runtime,
        value.value().clone(),
        path(steps),
        vec![rhs],
        None,
    )
    .await
    .unwrap()
}
fn numbers(value: &RootedValue) -> Vec<f64> {
    match value.value() {
        Value::Num(value) => vec![*value],
        Value::Tensor(value) => value.materialize_f64(),
        _ => panic!("double expression result"),
    }
}
#[test]
fn rooted_expression_assignments_evaluate_original_variable_shapes_and_keep_live_aliases() {
    block_on(async {
        let original = crate::user_functions::test_support::exact_fixture_runtime();
        let foreign = crate::user_functions::test_support::exact_fixture_runtime();
        let _ambient = foreign.enter();
        let x = call(
            &original,
            "optimvar",
            &[Value::from("decision"), Value::Num(2.)],
        )
        .await;
        let first = read(&original, &x, vec![paren(1.)]).await;
        let second = read(&original, &x, vec![paren(2.)]).await;
        let twice = call(
            &original,
            "optim.problemdef.OptimizationExpression.mtimes",
            &[Value::Num(2.), first.value().clone()],
        )
        .await;
        let total = call(
            &original,
            "optim.problemdef.OptimizationExpression.plus",
            &[twice.value().clone(), second.value().clone()],
        )
        .await;
        let triple = call(
            &original,
            "optim.problemdef.OptimizationExpression.mtimes",
            &[Value::Num(3.), second.value().clone()],
        )
        .await;
        let difference = call(
            &original,
            "optim.problemdef.OptimizationExpression.minus",
            &[first.value().clone(), triple.value().clone()],
        )
        .await;
        let empty = call(&original, "optimexpr", &[Value::Num(2.)]).await;
        let partial = assign(&original, &empty, vec![paren(1.)], total.value().clone()).await;
        let complete = assign(
            &original,
            &partial,
            vec![paren(2.)],
            difference.value().clone(),
        )
        .await;
        let point = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
            "decision".into(),
            array(vec![4., 1.], vec![2, 1]),
        )]));
        let selected = read(&original, &partial, vec![paren(1.)]).await;
        assert_eq!(
            numbers(
                &call(
                    &original,
                    "evaluate",
                    &[selected.value().clone(), point.clone()]
                )
                .await
            ),
            vec![9.]
        );
        let missing = crate::call_builtin_async_in_context(
            &original,
            "evaluate",
            &[partial.value().clone(), point.clone()],
        )
        .await
        .unwrap_err();
        assert_eq!(missing.identifier(), Some("RunMat:OptimModel:InvalidInput"));
        assert_eq!(
            numbers(
                &call(
                    &original,
                    "evaluate",
                    &[complete.value().clone(), point.clone()]
                )
                .await
            ),
            vec![9., 1.]
        );
        let variables = read(
            &original,
            &complete,
            vec![ObjectSubscript::member("Variables")],
        )
        .await;
        let Value::Struct(fields) = variables.value() else {
            panic!("actual variable definitions")
        };
        let retained =
            RootedValue::in_context(fields.fields.get("decision").unwrap().clone(), &original)
                .unwrap();
        let _updated = assign(
            &original,
            &retained,
            vec![paren(1.), ObjectSubscript::member("LowerBound")],
            Value::Num(7.),
        )
        .await;
        let lower = read(&original, &x, vec![ObjectSubscript::member("LowerBound")]).await;
        assert_eq!(numbers(&lower), vec![7., f64::NEG_INFINITY]);
        let inequality = call(
            &original,
            "optim.problemdef.OptimizationExpression.ge",
            &[x.value().clone(), Value::Num(2.)],
        )
        .await;
        let equality = call(
            &original,
            "optim.problemdef.OptimizationExpression.eq",
            &[x.value().clone(), Value::Num(2.)],
        )
        .await;
        let Value::Object(ge) = inequality.value() else {
            panic!("constraint")
        };
        let Value::Object(eq) = equality.value() else {
            panic!("constraint")
        };
        assert!(ge.class_name().is(standard::OPTIMIZATION_INEQUALITY));
        assert!(eq.class_name().is(standard::OPTIMIZATION_EQUALITY));
        assert_eq!(
            numbers(
                &call(
                    &original,
                    "evaluate",
                    &[inequality.value().clone(), point.clone()]
                )
                .await
            ),
            vec![-2., 1.]
        );
        assert_eq!(
            numbers(&call(&original, "evaluate", &[equality.value().clone(), point]).await),
            vec![2., 1.]
        );
        let other = call(
            &original,
            "optimvar",
            &[Value::from("decision"), Value::Num(2.)],
        )
        .await;
        let conflicting = call(
            &original,
            "optim.problemdef.OptimizationExpression.ge",
            &[x.value().clone(), other.value().clone()],
        )
        .await;
        let conflicting_point =
            Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
                "decision".into(),
                array(vec![4., 1.], vec![2, 1]),
            )]));
        let failure = crate::call_builtin_async_in_context(
            &original,
            "evaluate",
            &[conflicting.value().clone(), conflicting_point],
        )
        .await
        .unwrap_err();
        assert_eq!(failure.identifier(), Some("RunMat:OptimModel:InvalidInput"));
        assert!(failure.message().contains("different original variables"));
        let bad = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
            "decision".into(),
            array(vec![4., 1.], vec![1, 2]),
        )]));
        let failure = crate::call_builtin_async_in_context(
            &original,
            "evaluate",
            &[complete.value().clone(), bad],
        )
        .await
        .unwrap_err();
        assert_eq!(failure.identifier(), Some("RunMat:OptimModel:InvalidInput"));
    });
}
#[test]
fn canonical_numeric_functions_and_named_expression_assignment_share_real_shape_owners() {
    block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let x = call(&runtime, "optimvar", &[Value::from("x"), Value::Num(2.)]).await;
        let sum = call(
            &runtime,
            "optim.problemdef.OptimizationExpression.sum",
            &[x.value().clone()],
        )
        .await;
        let exp = call(
            &runtime,
            "optim.problemdef.OptimizationExpression.exp",
            &[x.value().clone()],
        )
        .await;
        let point = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
            "x".into(),
            Value::Tensor(
                runmat_value::Tensor::from_numeric_storage(
                    runmat_value::NumericStorage::F32(vec![0., 1.]),
                    vec![2, 1],
                )
                .unwrap(),
            ),
        )]));
        assert_eq!(
            numbers(&call(&runtime, "evaluate", &[sum.value().clone(), point.clone()]).await),
            vec![1.]
        );
        let values = numbers(&call(&runtime, "evaluate", &[exp.value().clone(), point]).await);
        assert_eq!(values[0], 1.);
        assert!((values[1] - std::f64::consts::E).abs() < 1e-14);
        let names = Value::Cell(
            runmat_value::CellArray::new(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap(),
        );
        let expr = call(&runtime, "optimexpr", &[names]).await;
        let selector = ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                ObjectIndexComponent::Value(Value::from("b")),
            ]),
        });
        let expr = assign(&runtime, &expr, vec![selector], Value::Num(5.)).await;
        let b = ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
            components: runmat_value::ExactArrayStorage::shared_unbounded(vec![
                ObjectIndexComponent::Value(Value::from("b")),
            ]),
        });
        let selected = read(&runtime, &expr, vec![b]).await;
        let empty = Value::Struct(runmat_value::StructValue::new());
        assert_eq!(
            numbers(&call(&runtime, "evaluate", &[selected.value().clone(), empty]).await),
            vec![5.]
        );

        let zero = call(
            &runtime,
            "zeros",
            &[
                array(vec![2., 1.], vec![1, 2]),
                Value::from("like"),
                x.value().clone(),
            ],
        )
        .await;
        let no_variables = Value::Struct(runmat_value::StructValue::new());
        let zero_value = call(
            &runtime,
            "evaluate",
            &[zero.value().clone(), no_variables.clone()],
        )
        .await;
        assert_eq!(numbers(&zero_value), vec![0., 0.]);
        let Value::Tensor(zeros) = zero_value.value() else {
            panic!("double zero array")
        };
        assert_eq!(zeros.shape(), &[2, 1]);
        let zero = assign(&runtime, &zero, vec![paren(2.)], sum.value().clone()).await;
        let full_point = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
            "x".into(),
            array(vec![3., 4.], vec![2, 1]),
        )]));
        assert_eq!(
            numbers(
                &call(
                    &runtime,
                    "evaluate",
                    &[zero.value().clone(), full_point.clone()]
                )
                .await
            ),
            vec![0., 7.]
        );

        let initial = call(&runtime, "optimexpr", &[]).await;
        let empty_value = call(
            &runtime,
            "evaluate",
            &[initial.value().clone(), no_variables.clone()],
        )
        .await;
        assert!(numbers(&empty_value).is_empty());
        let accumulated = call(
            &runtime,
            "optim.problemdef.OptimizationExpression.plus",
            &[initial.value().clone(), x.value().clone()],
        )
        .await;
        let accumulated = call(
            &runtime,
            "optim.problemdef.OptimizationExpression.minus",
            &[accumulated.value().clone(), Value::Num(1.)],
        )
        .await;
        assert_eq!(
            numbers(
                &call(
                    &runtime,
                    "evaluate",
                    &[accumulated.value().clone(), full_point]
                )
                .await
            ),
            vec![2., 3.]
        );
        for invalid in [
            Value::Num(f64::NAN),
            Value::Num(f64::INFINITY),
            Value::Num(f64::NEG_INFINITY),
            Value::Complex(1., 1.),
        ] {
            let error = crate::call_builtin_async_in_context(
                &runtime,
                "optim.problemdef.OptimizationExpression.plus",
                &[x.value().clone(), invalid],
            )
            .await
            .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:OptimModel:InvalidInput"));
        }
        let overflow_point = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
            "x".into(),
            array(vec![1000., 0.], vec![2, 1]),
        )]));
        let operation_result =
            call(&runtime, "evaluate", &[exp.value().clone(), overflow_point]).await;
        assert!(numbers(&operation_result)[0].is_infinite());
    });
}

#[test]
fn immutable_graph_roots_survive_collection_and_original_admission_stays_typed() {
    block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let expression = {
            let variable = call(&runtime, "optimvar", &[Value::from("retained")]).await;
            call(
                &runtime,
                "optim.problemdef.OptimizationExpression.plus",
                &[variable.value().clone(), Value::Num(3.)],
            )
            .await
        };
        let handle = owner::reference(
            expression.value(),
            Policy {
                identity: runmat_builtins::OPTIMEXPR_CATALOG_ENTRY.identity,
                invalid: &runmat_builtins::OPTIMMODEL_ERROR_INVALID_INPUT,
            },
        )
        .unwrap();
        runmat_gc::gc_collect_major().unwrap();
        let point = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
            "retained".into(),
            Value::Num(4.),
        )]));
        assert_eq!(
            numbers(&call(&runtime, "evaluate", &[expression.value().clone(), point]).await),
            vec![7.]
        );
        assert!(crate::execution::value_codec::encode_inline_value(expression.value()).is_err());
        drop(expression);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_err());
        use crate::context::{
            host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
            RuntimeParallelService, RuntimeServicePorts,
        };
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
        let finite = RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(std::rc::Rc::new(Finite(ledger.clone()))),
        );
        let policy = Policy {
            identity: runmat_builtins::OPTIMEXPR_CATALOG_ENTRY.identity,
            invalid: &runmat_builtins::OPTIMMODEL_ERROR_INVALID_INPUT,
        };
        let error = policy
            .cooperation(Some(&finite))
            .await
            .err()
            .expect("finite model backing is not certified");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
        assert_eq!(ledger.snapshot().peak_reserved_bytes, 0);
        let source = runmat_execution::cancellation::CancellationSource::new();
        let cancelled = RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_cancellation_token(source.token());
        cancelled.set_error_namespace("OriginalModel");
        source.cancel(runmat_execution::CancellationReason::User);
        let policy = Policy {
            identity: runmat_builtins::OPTIMEXPR_CATALOG_ENTRY.identity,
            invalid: &runmat_builtins::OPTIMMODEL_ERROR_INVALID_INPUT,
        };
        let error = policy
            .cooperation(Some(&cancelled))
            .await
            .err()
            .expect("original cancellation");
        assert_eq!(error.identifier(), Some("OriginalModel:Cancelled"));
        assert!(crate::runtime_error::error_has_control_cancellation_origin(
            &error
        ));
    });
}
