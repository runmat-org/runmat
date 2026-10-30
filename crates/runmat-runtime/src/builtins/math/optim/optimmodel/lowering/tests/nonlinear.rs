//! Original nonlinear callbacks are the retained model, not synthetic handles.
use super::*;
#[test]
fn nonlinear_models_use_original_points_algorithms_counts_and_constraint_roles() {
    block_on(async {
        let original = crate::user_functions::test_support::exact_fixture_runtime();
        let foreign = crate::user_functions::test_support::exact_fixture_runtime();
        let _ambient = foreign.enter();
        let x = call(
            &original,
            "optimvar",
            &[
                Value::from("rate"),
                Value::from("LowerBound"),
                Value::Num(-2.),
                Value::from("UpperBound"),
                Value::Num(2.),
            ],
        )
        .await;
        let shifted = operation(
            &original,
            "optim.problemdef.OptimizationExpression.minus",
            &x,
            Value::Num(1.),
        )
        .await;
        let objective = operation(
            &original,
            "optim.problemdef.OptimizationExpression.power",
            &shifted,
            Value::Num(4.),
        )
        .await;
        let square = operation(
            &original,
            "optim.problemdef.OptimizationExpression.power",
            &x,
            Value::Num(2.),
        )
        .await;
        let cap = operation(
            &original,
            "optim.problemdef.OptimizationExpression.le",
            &square,
            Value::Num(0.25),
        )
        .await;
        let problem = model(&original, &objective, &cap).await;
        let starting = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
            "rate".into(),
            Value::Num(0.1),
        )]));
        for algorithm in ["interior-point", "sqp", "active-set"] {
            let options = call(
                &original,
                "optimoptions",
                &[
                    Value::from("fmincon"),
                    Value::from("Algorithm"),
                    Value::from(algorithm),
                    Value::from("Display"),
                    Value::from("off"),
                    Value::from("MaxIterations"),
                    Value::Num(200.),
                    Value::from("MaxFunctionEvaluations"),
                    Value::Num(2000.),
                    Value::from("OptimalityTolerance"),
                    Value::Num(1e-6),
                    Value::from("ConstraintTolerance"),
                    Value::Num(1e-7),
                ],
            )
            .await;
            let result = crate::call_builtin_async_with_outputs_in_context(
                &original,
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
            let Value::OutputList(values) = result.value() else {
                panic!("five nonlinear outputs");
            };
            assert_eq!(values.len(), 5);
            let rate = numbers(field(&values[0], "rate"))[0];
            assert!((rate - 0.5).abs() < 2e-4, "{algorithm}: {rate}");
            assert!((numbers(&values[1])[0] - 0.0625).abs() < 2e-4);
            assert_eq!(field(&values[3], "Solver"), &Value::from("fmincon"));
            assert_eq!(field(&values[3], "algorithm"), &Value::from(algorithm));
            assert!(numbers(field(&values[3], "funcCount"))[0] > 0.);
            assert!(numbers(field(&values[3], "wallclocktime"))[0] >= 0.);
            let evaluated = call(
                &original,
                "evaluate",
                &[objective.value().clone(), values[0].clone()],
            )
            .await;
            assert_eq!(numbers(evaluated.value()), numbers(&values[1]));
            let flag = call(&original, "double", &[values[2].clone()]).await;
            assert!(numbers(flag.value())[0] > 0.);
            let multiplier = field(&values[4], "Constraints");
            assert!((numbers(multiplier)[0] - 0.5).abs() < 2e-3);
            assert_eq!(field(&starting, "rate"), &Value::Num(0.1));
        }
        let options = call(
            &original,
            "optimoptions",
            &[
                Value::from("fmincon"),
                Value::from("Display"),
                Value::from("off"),
                Value::from("MaxFunctionEvaluations"),
                Value::Num(0.),
            ],
        )
        .await;
        let stopped = crate::call_builtin_async_with_outputs_in_context(
            &original,
            "solve",
            &[
                problem.value().clone(),
                starting.clone(),
                Value::from("Options"),
                options.value().clone(),
            ],
            4,
        )
        .await
        .unwrap();
        let Value::OutputList(values) = stopped.value() else {
            panic!("four stopped outputs");
        };
        assert_eq!(numbers(field(&values[0], "rate")), vec![0.1]);
        assert!(matches!(&values[1], Value::Tensor(value) if value.is_empty()));
        assert_eq!(
            call(&original, "double", &[values[2].clone()])
                .await
                .value(),
            &Value::Num(0.)
        );
        assert_eq!(numbers(field(&values[3], "funcCount")), vec![0.]);

        // A constraints-only finite-difference probe must not evaluate log(x).
        let logarithm = call(
            &original,
            "optim.problemdef.OptimizationExpression.log",
            &[x.value().clone()],
        )
        .await;
        let domain_problem = model(&original, &logarithm, &cap).await;
        let mut c = ScalarCooperation::for_builtin(policy().identity, Some(&original));
        let source = super::super::super::problem::data(domain_problem.value(), policy()).unwrap();
        let formulation = formulation::Formulation::prepare(source, policy(), &mut c)
            .await
            .unwrap();
        let source = evaluation::Plan::prepare_roots(&formulation.roots, policy(), &mut c)
            .await
            .unwrap();
        let layout = layout::Layout::prepare(&source, policy(), &mut c)
            .await
            .unwrap();
        let model = super::super::nonlinear::Model::prepare(
            source,
            layout,
            &formulation,
            None,
            policy(),
            &mut c,
        )
        .await
        .unwrap();
        super::super::nonlinear::check_probe_roles(&model, &formulation, policy(), &mut c).await;
        unconstrained(&original).await;
    });
}

async fn unconstrained(original: &RuntimeContext) {
    let x = call(
        original,
        "optimvar",
        &[Value::from("free"), Value::Num(2.), Value::Num(1.)],
    )
    .await;
    let target = Value::Tensor(runmat_value::Tensor::new(vec![1., 2.], vec![2, 1]).unwrap());
    let shifted = operation(
        original,
        "optim.problemdef.OptimizationExpression.minus",
        &x,
        target,
    )
    .await;
    let quartic = operation(
        original,
        "optim.problemdef.OptimizationExpression.power",
        &shifted,
        Value::Num(4.),
    )
    .await;
    let square = operation(
        original,
        "optim.problemdef.OptimizationExpression.power",
        &shifted,
        Value::Num(2.),
    )
    .await;
    let sum = operation(
        original,
        "optim.problemdef.OptimizationExpression.plus",
        &quartic,
        square.value().clone(),
    )
    .await;
    let objective = call(
        original,
        "optim.problemdef.OptimizationExpression.sum",
        &[sum.value().clone()],
    )
    .await;
    let problem = call(
        original,
        "optimproblem",
        &[Value::from("Objective"), objective.value().clone()],
    )
    .await;
    let start = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
        "free".into(),
        Value::Tensor(runmat_value::Tensor::new(vec![0., 0.], vec![2, 1]).unwrap()),
    )]));
    for explicit in [false, true] {
        let options = call(
            original,
            "optimoptions",
            &[
                Value::from("fminunc"),
                Value::from("Display"),
                Value::from("off"),
                Value::from("HessianApproximation"),
                Value::from(if explicit { "lbfgs" } else { "bfgs" }),
            ],
        )
        .await;
        let mut args = vec![
            problem.value().clone(),
            start.clone(),
            Value::from("Options"),
            options.value().clone(),
        ];
        if explicit {
            args.extend([Value::from("Solver"), Value::from("fminunc")]);
        }
        let solved = crate::call_builtin_async_with_outputs_in_context(original, "solve", &args, 5)
            .await
            .unwrap();
        let Value::OutputList(values) = solved.value() else {
            panic!("five original-model outputs");
        };
        let coordinates = numbers(field(&values[0], "free"));
        assert!((coordinates[0] - 1.).abs() < 1e-3 && (coordinates[1] - 2.).abs() < 1e-3);
        assert!(numbers(&values[1])[0] < 1e-6);
        assert_eq!(field(&values[3], "Solver"), &Value::from("fminunc"));
        assert_eq!(
            field(&values[3], "objectivederivative"),
            &Value::from("finite-differences")
        );
        let Value::Struct(info) = &values[3] else {
            panic!("QN information");
        };
        assert!(!info.fields.contains_key("wallclocktime"));
        assert!(matches!(&values[4], Value::Tensor(value) if value.is_empty()));
        let evaluated = call(
            original,
            "evaluate",
            &[objective.value().clone(), values[0].clone()],
        )
        .await;
        assert_eq!(numbers(evaluated.value()), numbers(&values[1]));
        assert!(numbers(call(original, "double", &[values[2].clone()]).await.value())[0] > 0.);
    }
    let options = call(
        original,
        "optimoptions",
        &[
            Value::from("fminunc"),
            Value::from("Display"),
            Value::from("off"),
            Value::from("ObjectiveLimit"),
            Value::Num(100.),
        ],
    )
    .await;
    let stopped = crate::call_builtin_async_with_outputs_in_context(
        original,
        "solve",
        &[
            problem.value().clone(),
            start.clone(),
            Value::from("Options"),
            options.value().clone(),
        ],
        4,
    )
    .await
    .unwrap();
    let Value::OutputList(values) = stopped.value() else {
        panic!("threshold outputs");
    };
    assert_eq!(numbers(field(&values[0], "free")), vec![0., 0.]);
    assert_eq!(numbers(&values[1]), vec![22.]);
    assert_eq!(
        numbers(call(original, "double", &[values[2].clone()]).await.value()),
        vec![-3.]
    );
    assert!(
        matches!(field(&values[3],"message"),Value::String(runmat_value::StringElement::Text(message)) if message.contains("ObjectiveLimit"))
    );
    let bounded = call(
        original,
        "optimvar",
        &[
            Value::from("bound"),
            Value::from("LowerBound"),
            Value::Num(0.),
        ],
    )
    .await;
    let bounded_problem = call(
        original,
        "optimproblem",
        &[Value::from("Objective"), bounded.value().clone()],
    )
    .await;
    assert!(crate::call_builtin_async_in_context(
        original,
        "solve",
        &[
            bounded_problem.value().clone(),
            Value::from("Solver"),
            Value::from("fminunc")
        ]
    )
    .await
    .is_err());
}
