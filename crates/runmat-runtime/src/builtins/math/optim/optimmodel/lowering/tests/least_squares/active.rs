//! Original values/shapes survive stopped active results without feasible proof.
use super::*;
pub(super) async fn check(
    runtime: &RuntimeContext,
    problem: &RootedValue,
    objective: &RootedValue,
    starting: &Value,
) {
    let options = call(
        runtime,
        "optimoptions",
        &[
            Value::from("lsqlin"),
            Value::from("Algorithm"),
            Value::from("active-set"),
            Value::from("Display"),
            Value::from("off"),
            Value::from("ObjectiveLimit"),
            Value::Num(15.),
        ],
    )
    .await;
    for (limit, member, expected) in [
        (2000., "OptimalSolution", vec![0.5, 2.5]),
        (0., "SolverLimitExceeded", vec![100., 200.]),
    ] {
        let options = call(
            runtime,
            "optimoptions",
            &[
                options.value().clone(),
                Value::from("MaxIterations"),
                Value::Num(limit),
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
        runmat_gc::gc_collect_major().unwrap();
        let Value::OutputList(result) = result.value() else {
            panic!("active model outputs")
        };
        let x = numbers(field(&result[0], "allocation"));
        assert_eq!(x.len(), expected.len());
        for (x, expected) in x.into_iter().zip(expected) {
            assert!((x - expected).abs() < 1e-5);
        }
        assert_eq!(
            call(runtime, "string", &[result[2].clone()]).await.value(),
            &Value::from(member)
        );
        let evaluated = call(
            runtime,
            "evaluate",
            &[objective.value().clone(), result[0].clone()],
        )
        .await;
        assert_eq!(numbers(evaluated.value()), numbers(&result[1]));
        if limit == 0. {
            assert!(numbers(field(&result[4], "Constraints")).is_empty());
            assert!(numbers(field(&result[3], "constrviolation"))[0] > 0.);
        }
    }
    let threshold = call(
        runtime,
        "optimoptions",
        &[
            options.value().clone(),
            Value::from("ObjectiveLimit"),
            Value::Num(100.),
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
            threshold.value().clone(),
        ],
        5,
    )
    .await
    .unwrap();
    let Value::OutputList(result) = result.value() else {
        panic!("threshold outputs")
    };
    assert_eq!(
        call(runtime, "string", &[result[2].clone()]).await.value(),
        &Value::from("ObjectiveLimitReached")
    );
    assert_eq!(
        call(runtime, "double", &[result[2].clone()]).await.value(),
        &Value::Num(-3.)
    );
    assert!(numbers(&result[1])[0] < 100.);
    assert!(numbers(field(&result[4], "Constraints")).is_empty());
}
