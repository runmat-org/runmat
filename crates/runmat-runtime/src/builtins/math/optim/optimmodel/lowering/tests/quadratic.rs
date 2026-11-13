//! Public typed QP consumer, source objective units and exact ray/curvature guards.
use super::operation as invoke;
use super::*;
async fn problem(runtime: &RuntimeContext, objective: &RootedValue) -> RootedValue {
    call(
        runtime,
        "optimproblem",
        &[Value::from("Objective"), objective.value().clone()],
    )
    .await
}
async fn solve(
    runtime: &RuntimeContext,
    problem: &RootedValue,
    extra: &[Value],
    count: usize,
) -> RootedValue {
    let mut args = vec![problem.value().clone()];
    args.extend_from_slice(extra);
    crate::call_builtin_async_with_outputs_in_context(runtime, "solve", &args, count)
        .await
        .unwrap()
}
fn outputs(result: &RootedValue) -> &[Value] {
    let Value::OutputList(values) = result.value() else {
        panic!("actual model outputs")
    };
    values
}
pub(super) async fn controls(runtime: &RuntimeContext) {
    let foreign = crate::user_functions::test_support::exact_fixture_runtime();
    let _ambient = foreign.enter();
    mixed_norm(runtime).await;
    let x = call(
        runtime,
        "optimvar",
        &[
            Value::from("q"),
            Value::from("LowerBound"),
            Value::Num(0.),
            Value::from("UpperBound"),
            Value::Num(1.),
        ],
    )
    .await;
    let square = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &x,
        x.value().clone(),
    )
    .await;
    let three = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &square,
        Value::Num(3.),
    )
    .await;
    let twice = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &x,
        Value::Num(2.),
    )
    .await;
    let objective = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.minus",
        &three,
        twice.value().clone(),
    )
    .await;
    let objective = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        &objective,
        Value::Num(5.),
    )
    .await;
    let model = problem(runtime, &objective).await;
    let result = solve(runtime, &model, &[], 5).await;
    runmat_gc::gc_collect_major().unwrap();
    let values = outputs(&result);
    assert!((numbers(field(&values[0], "q"))[0] - 1. / 3.).abs() < 1e-5);
    assert!((numbers(&values[1])[0] - 14. / 3.).abs() < 1e-6);
    assert_eq!(
        call(runtime, "double", &[values[2].clone()]).await.value(),
        &Value::Num(1.)
    );
    assert_eq!(field(&values[3], "Solver"), &Value::from("quadprog"));
    assert_eq!(
        field(&values[3], "algorithm"),
        &Value::from("interior-point-convex")
    );
    assert!(numbers(field(&values[3], "wallclocktime"))[0] >= 0.);
    let actual = call(
        runtime,
        "evaluate",
        &[objective.value().clone(), values[0].clone()],
    )
    .await;
    assert_eq!(numbers(actual.value()), numbers(&values[1]));

    // Explicit QP is also a documented LP and residual solver selection.
    let linear = problem(runtime, &x).await;
    let result = solve(
        runtime,
        &linear,
        &[Value::from("Solver"), Value::from("quadprog")],
        4,
    )
    .await;
    let values = outputs(&result);
    assert!(numbers(field(&values[0], "q"))[0].abs() < 1e-5);
    assert_eq!(field(&values[3], "Solver"), &Value::from("quadprog"));

    let u = call(
        runtime,
        "optimvar",
        &[Value::from("u"), Value::from("LowerBound"), Value::Num(0.)],
    )
    .await;
    let residual = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.minus",
        &u,
        Value::Num(2.),
    )
    .await;
    let norm = call(
        runtime,
        "optim.problemdef.OptimizationExpression.norm",
        &[residual.value().clone()],
    )
    .await;
    let squared = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.mpower",
        &norm,
        Value::Num(2.),
    )
    .await;
    let objective = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        &squared,
        Value::Num(7.),
    )
    .await;
    let constraint = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.ge",
        &u,
        Value::Num(3.),
    )
    .await;
    let model = super::model(runtime, &objective, &constraint).await;
    let result = solve(
        runtime,
        &model,
        &[Value::from("Solver"), Value::from("quadprog")],
        5,
    )
    .await;
    let values = outputs(&result);
    assert!((numbers(field(&values[0], "u"))[0] - 3.).abs() < 1e-5);
    assert!((numbers(&values[1])[0] - 8.).abs() < 1e-5);
    assert!((numbers(field(&values[4], "Constraints"))[0] - 2.).abs() < 1e-4);

    let negative = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &squared,
        Value::Num(-1.),
    )
    .await;
    let objective = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        &negative,
        Value::Num(7.),
    )
    .await;
    let model = call(
        runtime,
        "optimproblem",
        &[
            Value::from("Objective"),
            objective.value().clone(),
            Value::from("ObjectiveSense"),
            Value::from("max"),
        ],
    )
    .await;
    let result = solve(runtime, &model, &[], 4).await;
    let values = outputs(&result);
    assert!((numbers(field(&values[0], "u"))[0] - 2.).abs() < 1e-5);
    assert!((numbers(&values[1])[0] - 7.).abs() < 1e-6);
    assert_eq!(field(&values[3], "Solver"), &Value::from("quadprog"));

    // Original off-diagonal curvature survives below the smallest double.
    let a = call(runtime, "optimvar", &[Value::from("a")]).await;
    let b = call(runtime, "optimvar", &[Value::from("b")]).await;
    let tiny = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &a,
        Value::Num(f64::from_bits(1)),
    )
    .await;
    let cross = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &tiny,
        b.value().clone(),
    )
    .await;
    let cross = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.rdivide",
        &cross,
        Value::Num(2.),
    )
    .await;
    let model = problem(runtime, &cross).await;
    let result = solve(runtime, &model, &[], 3).await;
    let values = outputs(&result);
    assert_eq!(
        call(runtime, "double", &[values[2].clone()]).await.value(),
        &Value::Num(-6.)
    );
    assert_eq!(
        call(runtime, "string", &[values[2].clone()]).await.value(),
        &Value::from("Nonconvex")
    );
    assert!(matches!(&values[1],Value::Tensor(value)if value.is_empty()));

    // Rounded H=0 cannot certify a ray through a tiny original positive H.
    let curved = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &tiny,
        a.value().clone(),
    )
    .await;
    let curved = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.rdivide",
        &curved,
        Value::Num(2.),
    )
    .await;
    let curved = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.minus",
        &curved,
        a.value().clone(),
    )
    .await;
    let model = problem(runtime, &curved).await;
    let error = crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "solve",
        &[model.value().clone()],
        3,
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::OPTIMPROBLEM_SOLVE_ERROR_UNVERIFIED.identifier
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}

async fn mixed_norm(runtime: &RuntimeContext) {
    let order = Value::Tensor(
        runmat_value::Tensor::from_numeric_storage(
            runmat_value::NumericStorage::U16(vec![2]),
            vec![1, 1],
        )
        .unwrap(),
    );
    let variable = call(
        runtime,
        "optimvar",
        &[Value::from("normrow"), Value::Num(2.), Value::Num(1.)],
    )
    .await;
    let residual = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.minus",
        &variable,
        Value::Tensor(runmat_value::Tensor::new(vec![1., 2.], vec![2, 1]).unwrap()),
    )
    .await;
    let norm = call(
        runtime,
        "optim.problemdef.OptimizationExpression.norm",
        &[residual.value().clone(), order],
    )
    .await;
    let square = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.mpower",
        &norm,
        Value::Num(2.),
    )
    .await;
    let second = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.power",
        &norm,
        Value::Num(2.),
    )
    .await;
    let sum = call(
        runtime,
        "optim.problemdef.OptimizationExpression.sum",
        &[variable.value().clone()],
    )
    .await;
    for repeated in [false, true] {
        let quadratic = if repeated {
            invoke(
                runtime,
                "optim.problemdef.OptimizationExpression.plus",
                &square,
                second.value().clone(),
            )
            .await
        } else {
            square.clone()
        };
        let linear = invoke(
            runtime,
            "optim.problemdef.OptimizationExpression.times",
            &sum,
            Value::Num(if repeated { 1. } else { 0. }),
        )
        .await;
        let objective = invoke(
            runtime,
            "optim.problemdef.OptimizationExpression.plus",
            &quadratic,
            linear.value().clone(),
        )
        .await;
        let model = problem(runtime, &objective).await;
        let result = solve(runtime, &model, &[], 4).await;
        let values = outputs(&result);
        let expected = if repeated { [0.75, 1.75] } else { [1., 2.] };
        for (&actual, expected) in numbers(field(&values[0], "normrow")).iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert!((numbers(&values[1])[0] - if repeated { 2.75 } else { 0. }).abs() < 1e-6);
        assert_eq!(field(&values[3], "Solver"), &Value::from("quadprog"));
        let reevaluated = call(
            runtime,
            "evaluate",
            &[objective.value().clone(), values[0].clone()],
        )
        .await;
        assert!((numbers(reevaluated.value())[0] - numbers(&values[1])[0]).abs() < 1e-12);
    }
    // A matrix default norm is spectral, never an atomic entry-square sum.
    let matrix = call(
        runtime,
        "optimvar",
        &[Value::from("spectral"), Value::Num(2.), Value::Num(2.)],
    )
    .await;
    let norm = call(
        runtime,
        "optim.problemdef.OptimizationExpression.norm",
        &[matrix.value().clone()],
    )
    .await;
    let square = invoke(
        runtime,
        "optim.problemdef.OptimizationExpression.mpower",
        &norm,
        Value::Num(2.),
    )
    .await;
    let model = problem(runtime, &square).await;
    let error = crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "solve",
        &[
            model.value().clone(),
            Value::from("Solver"),
            Value::from("quadprog"),
        ],
        3,
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::OPTIMPROBLEM_ERROR_INVALID_INPUT.identifier
    );
}
