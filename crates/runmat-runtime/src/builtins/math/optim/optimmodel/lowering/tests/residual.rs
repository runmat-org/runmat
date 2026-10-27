//! Actual problem ingress, source fval, accepted probe and point-budget control.
use super::*;
pub(super) async fn controls(runtime: &RuntimeContext) {
    let foreign = crate::user_functions::test_support::exact_fixture_runtime();
    let _ambient = foreign.enter();
    let x = call(
        runtime,
        "optimvar",
        &[
            Value::from("rate"),
            Value::from("LowerBound"),
            Value::Num(0.),
            Value::from("UpperBound"),
            Value::Num(1.),
        ],
    )
    .await;
    let sine = call(
        runtime,
        "optim.problemdef.OptimizationExpression.sin",
        &[x.value().clone()],
    )
    .await;
    let r = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.minus",
        &sine,
        Value::Num(0.5),
    )
    .await;
    let square = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.power",
        &r,
        Value::Num(2.),
    )
    .await;
    let cost = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        &square,
        Value::Num(7.),
    )
    .await;
    let problem = call(
        runtime,
        "optimproblem",
        &[Value::from("Objective"), cost.value().clone()],
    )
    .await;
    let start = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
        "rate".into(),
        Value::Num(0.3),
    )]));
    for limit in [0., 1., 200.] {
        let options = call(
            runtime,
            "optimoptions",
            &[
                Value::from("lsqnonlin"),
                Value::from("Algorithm"),
                Value::from("levenberg-marquardt"),
                Value::from("Display"),
                Value::from("off"),
                Value::from("TolFun"),
                Value::Num(1e-10),
                Value::from("TolX"),
                Value::Num(1e-10),
                Value::from("MaxFunEvals"),
                Value::Num(limit),
            ],
        )
        .await;
        let result = crate::call_builtin_async_with_outputs_in_context(
            runtime,
            "solve",
            &[
                problem.value().clone(),
                start.clone(),
                Value::from("Options"),
                options.value().clone(),
            ],
            5,
        )
        .await
        .unwrap();
        let Value::OutputList(values) = result.value() else {
            panic!("model residual outputs");
        };
        let rate = numbers(field(&values[0], "rate"))[0];
        let fval = numbers(&values[1])[0];
        assert_eq!(field(&values[3], "Solver"), &Value::from("lsqnonlin"));
        if limit == 0. {
            assert_eq!(rate, 0.3);
            assert!(fval.is_nan());
            assert_eq!(
                field(&values[3], "objectivederivative"),
                &Value::from("not-evaluated")
            );
            assert_eq!(field(&values[3], "algorithm"), &Value::from("not-started"));
            assert_eq!(numbers(field(&values[3], "funcCount")), vec![0.]);
            assert_eq!(
                call(runtime, "double", &[values[2].clone()]).await.value(),
                &Value::Num(0.)
            );
            let variables = field(&values[4], "Variables");
            assert!(numbers(field(field(variables, "rate"), "Lower")).is_empty());
            continue;
        }
        assert_eq!(
            field(&values[3], "objectivederivative"),
            &Value::from("finite-differences")
        );
        assert_eq!(
            field(&values[3], "algorithm"),
            &Value::from("levenberg-marquardt")
        );
        assert!((fval - (7. + (rate.sin() - 0.5).powi(2))).abs() < 1e-12);
        let evaluated = call(
            runtime,
            "evaluate",
            &[cost.value().clone(), values[0].clone()],
        )
        .await;
        assert_eq!(numbers(evaluated.value()), vec![fval]);
        let count = numbers(field(&values[3], "funcCount"))[0];
        if limit == 1. {
            assert_eq!(rate, 0.3);
            assert_eq!(count, 1.);
            assert_eq!(
                call(runtime, "double", &[values[2].clone()]).await.value(),
                &Value::Num(0.)
            );
            let variables = field(&values[4], "Variables");
            assert!(numbers(field(field(variables, "rate"), "Lower")).is_empty());
        } else {
            assert!((rate - 0.5f64.asin()).abs() < 1e-5);
            assert!((fval - 7.).abs() < 1e-9);
            assert!(count > 1. && count <= limit);
            assert!(numbers(call(runtime, "double", &[values[2].clone()]).await.value())[0] > 0.);
        }
        let cap = operation(
            runtime,
            "optim.problemdef.OptimizationExpression.le",
            &x,
            Value::Num(0.8),
        )
        .await;
        let constrained = model(runtime, &cost, &cap).await;
        let error = crate::call_builtin_async_with_outputs_in_context(
            runtime,
            "solve",
            &[
                constrained.value().clone(),
                start.clone(),
                Value::from("Solver"),
                Value::from("lsqnonlin"),
                Value::from("Options"),
                options.value().clone(),
            ],
            1,
        )
        .await
        .unwrap_err();
        assert!(error.message().contains("bounds only"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
}
