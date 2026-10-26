//! Public native equation ingress, original residual shapes and real solver work.
use super::*;
pub(super) async fn controls(runtime: &RuntimeContext) {
    let foreign = crate::user_functions::test_support::exact_fixture_runtime();
    let _ambient = foreign.enter();
    let x = call(runtime, "optimvar", &[Value::from("answer")]).await;
    let linear = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &x,
        Value::Tensor(runmat_value::Tensor::new(vec![2., 4.], vec![2, 1]).unwrap()),
    )
    .await;
    let equality = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.eq",
        &linear,
        Value::Tensor(runmat_value::Tensor::new(vec![6., 12.], vec![2, 1]).unwrap()),
    )
    .await;
    let named = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
        "answer".into(),
        equality.value().clone(),
    )]));
    let problem = call(runtime, "eqnproblem", &[Value::from("Equations"), named]).await;
    let result = crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "solve",
        &[problem.value().clone()],
        4,
    )
    .await
    .unwrap();
    let Value::OutputList(output) = result.value() else {
        panic!("equation products");
    };
    assert!((numbers(field(&output[0], "answer"))[0] - 3.).abs() < 1e-6);
    let Value::Tensor(residual) = &output[1] else {
        panic!("one named equation returns an array");
    };
    assert_eq!(residual.shape(), &[2, 1]);
    assert!(residual.materialize_f64().iter().all(|r| r.abs() < 1e-6));
    assert_eq!(field(&output[3], "Solver"), &Value::from("lsqlin"));
    assert_eq!(
        call(runtime, "string", &[output[2].clone()]).await.value(),
        &Value::from("EquationSolved")
    );
    assert!(numbers(call(runtime, "double", &[output[2].clone()]).await.value())[0] > 0.);
    let error = crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "solve",
        &[problem.value().clone()],
        5,
    )
    .await
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);

    let square = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.power",
        &x,
        Value::Num(2.),
    )
    .await;
    let equality = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.eq",
        &square,
        Value::Num(2.),
    )
    .await;
    let problem = call(
        runtime,
        "eqnproblem",
        &[Value::from("Equations"), equality.value().clone()],
    )
    .await;
    let start = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
        "answer".into(),
        Value::Num(1.),
    )]));
    let result = crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "solve",
        &[problem.value().clone(), start.clone()],
        4,
    )
    .await
    .unwrap();
    let Value::OutputList(output) = result.value() else {
        panic!("scalar equation products");
    };
    let selected = numbers(field(&output[0], "answer"))[0];
    assert!((selected - 2.0f64.sqrt()).abs() < 1e-6);
    assert!((numbers(&output[1])[0] - (selected * selected - 2.)).abs() < 1e-12);
    assert_eq!(field(&output[3], "Solver"), &Value::from("fzero"));
    assert_eq!(
        call(runtime, "string", &[output[2].clone()]).await.value(),
        &Value::from("EquationSolved")
    );
    assert_eq!(
        field(&output[3], "rootcriterion"),
        &Value::from("numerical-bracket-or-zero")
    );
    assert_eq!(
        numbers(call(runtime, "double", &[output[2].clone()]).await.value()),
        vec![1.]
    );
    assert!(numbers(field(&output[3], "funcCount"))[0] > 1.);

    let options = call(
        runtime,
        "optimoptions",
        &[
            Value::from("fsolve"),
            Value::from("MaxFunEvals"),
            Value::Num(1.),
            Value::from("Display"),
            Value::from("off"),
        ],
    )
    .await;
    let result = crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "solve",
        &[
            problem.value().clone(),
            start,
            Value::from("Solver"),
            Value::from("fsolve"),
            Value::from("Options"),
            options.value().clone(),
        ],
        4,
    )
    .await
    .unwrap();
    let Value::OutputList(output) = result.value() else {
        panic!("system equation products");
    };
    assert_eq!(numbers(field(&output[0], "answer")), vec![1.]);
    assert_eq!(numbers(&output[1]), vec![-1.]);
    assert_eq!(numbers(field(&output[3], "funcCount")), vec![1.]);
    assert_eq!(
        numbers(call(runtime, "double", &[output[2].clone()]).await.value()),
        vec![0.]
    );
    let positive = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        &square,
        Value::Num(1.),
    )
    .await;
    let equality = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.eq",
        &positive,
        Value::Num(0.),
    )
    .await;
    let problem = call(
        runtime,
        "eqnproblem",
        &[Value::from("Equations"), equality.value().clone()],
    )
    .await;
    let start = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
        "answer".into(),
        Value::Num(0.),
    )]));
    let result = crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "solve",
        &[
            problem.value().clone(),
            start,
            Value::from("Solver"),
            Value::from("fsolve"),
        ],
        3,
    )
    .await
    .unwrap();
    let Value::OutputList(output) = result.value() else {
        panic!("nonroot equation products");
    };
    assert_eq!(numbers(&output[1]), vec![1.]);
    assert_eq!(
        numbers(call(runtime, "double", &[output[2].clone()]).await.value()),
        vec![-2.]
    );
    assert_eq!(
        call(runtime, "string", &[output[2].clone()]).await.value(),
        &Value::from("NoFeasiblePointFound")
    );
}
