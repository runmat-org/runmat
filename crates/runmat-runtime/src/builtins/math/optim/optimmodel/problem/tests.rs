//! Actual constructor/functional-field/native-GC consumer, not a DTO fixture.
use super::*;
use futures::executor::block_on;
async fn call(runtime: &RuntimeContext, name: &str, args: &[Value]) -> RootedValue {
    crate::call_builtin_async_in_context(runtime, name, args)
        .await
        .unwrap()
}
async fn get(runtime: &RuntimeContext, value: &RootedValue, name: &str) -> RootedValue {
    call(
        runtime,
        "getfield",
        &[value.value().clone(), Value::from(name)],
    )
    .await
}
async fn put(
    runtime: &RuntimeContext,
    value: &RootedValue,
    names: &[&str],
    rhs: Value,
) -> RootedValue {
    let mut args = vec![value.value().clone()];
    args.extend(names.iter().map(|&name| Value::from(name)));
    args.push(rhs);
    call(runtime, "setfield", &args).await
}
fn point(values: Vec<f64>) -> Value {
    Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
        "decision".into(),
        Value::Tensor(runmat_value::Tensor::new(values, vec![2, 1]).unwrap()),
    )]))
}
#[test]
fn original_problem_generations_project_live_variables_and_real_collections() {
    block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        runtime.set_runmat_extensions_enabled(true);
        let foreign = crate::user_functions::test_support::exact_fixture_runtime();
        let _ambient = foreign.enter();
        let mut problem = call(&runtime, "optimproblem", &[]).await;
        assert!(
            matches!(get(&runtime,&problem,"Description").await.value(),Value::CharArray(value) if value.shape().contains(&0))
        );
        assert!(
            matches!(get(&runtime,&problem,"Variables").await.value(),Value::StructArray(value) if value.is_empty())
        );
        let initial = get(&runtime, &problem, "Objective").await;
        let Value::Object(initial) = initial.value() else {
            panic!("empty expression")
        };
        assert_eq!(
            initial.optimization_model_shape().unwrap().as_slice(),
            &[0, 0]
        );
        let x = call(
            &runtime,
            "optimvar",
            &[
                Value::from("decision"),
                Value::Num(2.),
                Value::from("LowerBound"),
                Value::Num(0.),
            ],
        )
        .await;
        let objective = call(
            &runtime,
            "optim.problemdef.OptimizationExpression.sum",
            &[x.value().clone()],
        )
        .await;
        let constraint = call(
            &runtime,
            "optim.problemdef.OptimizationExpression.le",
            &[objective.value().clone(), Value::Num(5.)],
        )
        .await;
        problem = put(
            &runtime,
            &problem,
            &["Objective"],
            objective.value().clone(),
        )
        .await;
        problem = put(
            &runtime,
            &problem,
            &["Constraints", "capacity"],
            constraint.value().clone(),
        )
        .await;
        drop(objective);
        drop(constraint);
        runmat_gc::gc_collect_major().unwrap();
        let objective = get(&runtime, &problem, "Objective").await;
        let result = call(
            &runtime,
            "evaluate",
            &[objective.value().clone(), point(vec![1., 2.])],
        )
        .await;
        let result = match result.value() {
            Value::Num(value) => vec![*value],
            Value::Tensor(value) => value.materialize_f64(),
            _ => panic!("double evaluated objective"),
        };
        assert_eq!(result, vec![3.]);
        let variables = get(&runtime, &problem, "Variables").await;
        let variable = get(&runtime, &variables, "decision").await;
        let _changed = put(&runtime, &variable, &["UpperBound"], Value::Num(4.)).await;
        let upper = get(&runtime, &x, "UpperBound").await;
        let Value::Tensor(upper) = upper.value() else {
            panic!("original bounds")
        };
        assert_eq!(upper.materialize_f64(), vec![4., 4.]);

        let updated = put(&runtime, &problem, &["Description"], Value::from("updated")).await;
        assert!(
            matches!(get(&runtime,&problem,"Description").await.value(),Value::CharArray(value) if value.data().is_empty())
        );
        assert_eq!(
            get(&runtime, &updated, "Description").await.value(),
            &Value::from("updated")
        );
        let mut named = call(&runtime, "optimproblem", &[]).await;
        named = put(
            &runtime,
            &named,
            &["Objective", "first"],
            objective.value().clone(),
        )
        .await;
        named = put(
            &runtime,
            &named,
            &["Objective", "second"],
            objective.value().clone(),
        )
        .await;
        named = put(
            &runtime,
            &named,
            &["ObjectiveSense", "second"],
            Value::from("max"),
        )
        .await;
        let senses = get(&runtime, &named, "ObjectiveSense").await;
        assert_eq!(
            get(&runtime, &senses, "first").await.value(),
            &Value::from("minimize")
        );
        assert_eq!(
            get(&runtime, &senses, "second").await.value(),
            &Value::from("maximize")
        );
        let empty = Value::Tensor(runmat_value::Tensor::new(vec![], vec![0, 0]).unwrap());
        problem = put(&runtime, &problem, &["Constraints", "capacity"], empty).await;
        assert!(
            matches!(get(&runtime,&problem,"Constraints").await.value(),Value::Struct(fields) if fields.fields.is_empty())
        );
        let collision = crate::call_builtin_async_in_context(
            &runtime,
            "setfield",
            &[
                named.value().clone(),
                Value::from("Constraints"),
                Value::from("decision"),
                call(
                    &runtime,
                    "optim.problemdef.OptimizationExpression.le",
                    &[x.value().clone(), Value::Num(4.)],
                )
                .await
                .value()
                .clone(),
            ],
        )
        .await
        .unwrap_err();
        assert_eq!(
            collision.identifier(),
            Some("RunMat:OptimProblem:InvalidInput")
        );
        assert!(crate::call_builtin_async_in_context(
            &runtime,
            "setfield",
            &[
                problem.value().clone(),
                Value::from("Variables"),
                Value::Num(1.)
            ]
        )
        .await
        .is_err());
    });
}
