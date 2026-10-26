//! Same weighted original model, now with the selected bound-only algorithm.
use super::*;
pub(super) async fn check(runtime: &RuntimeContext, objective: &RootedValue, starting: &Value) {
    let problem = call(
        runtime,
        "optimproblem",
        &[Value::from("Objective"), objective.value().clone()],
    )
    .await;
    let options = call(
        runtime,
        "optimoptions",
        &[
            Value::from("lsqlin"),
            Value::from("Algorithm"),
            Value::from("trust-region-reflective"),
            Value::from("Display"),
            Value::from("off"),
            Value::from("MaxPCGIter"),
            Value::Num(2.),
            Value::from("FunctionTolerance"),
            Value::Num(0.),
            Value::from("OptimalityTolerance"),
            Value::Num(1e-8),
        ],
    )
    .await;
    let result = crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "solve",
        &[
            problem.value().clone(),
            starting.clone(),
            Value::from("Options"),
            options.value().clone(),
        ],
        5,
    )
    .await
    .unwrap();
    let Value::OutputList(result) = result.value() else {
        panic!("reflective original model outputs")
    };
    let x = numbers(field(&result[0], "allocation"));
    assert!((x[0] - 2.).abs() < 1e-6 && (x[1] - 4.).abs() < 1e-6);
    assert!((numbers(&result[1])[0] - 7.).abs() < 1e-6);
    assert_eq!(
        call(runtime, "double", &[result[2].clone()]).await.value(),
        &Value::Num(1.)
    );
    assert_eq!(
        field(&result[3], "algorithm"),
        &Value::from("trust-region-reflective")
    );
    assert!(numbers(field(&result[3], "cgiterations"))[0] > 0.);
    assert_eq!(numbers(field(starting, "allocation")), vec![100., 200.]);
    let stopped_options = call(
        runtime,
        "optimoptions",
        &[
            options.value().clone(),
            Value::from("FunctionTolerance"),
            Value::Num(2.),
            Value::from("OptimalityTolerance"),
            Value::Num(0.),
        ],
    )
    .await;
    let stopped = crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "solve",
        &[
            problem.value().clone(),
            starting.clone(),
            Value::from("Options"),
            stopped_options.value().clone(),
        ],
        5,
    )
    .await
    .unwrap();
    let Value::OutputList(stopped) = stopped.value() else {
        panic!("function-change model outputs")
    };
    assert_eq!(
        call(runtime, "string", &[stopped[2].clone()]).await.value(),
        &Value::from("FunctionChangeBelowTolerance")
    );
    assert_eq!(
        call(runtime, "double", &[stopped[2].clone()]).await.value(),
        &Value::Num(3.)
    );
    assert!(numbers(field(&stopped[0], "allocation"))[0] > 2.);
    assert!(numbers(&stopped[1])[0].is_finite());
}
