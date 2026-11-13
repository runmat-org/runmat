//! Zero-or-active intervals must not be replaced by ordinary positive bounds.
use super::*;
pub(super) async fn controls(runtime: &RuntimeContext, options: &RootedValue) {
    for (domain, lower, upper, cap, expected) in [
        ("semi-continuous", 2., 4., 1.5, 0.),
        ("semi-integer", 1.2, 4.5, 3.8, 3.),
    ] {
        let x = call(
            runtime,
            "optimvar",
            &[
                Value::from("semi"),
                Value::from("Type"),
                Value::from(domain),
                Value::from("LowerBound"),
                Value::Num(lower),
                Value::from("UpperBound"),
                Value::Num(upper),
            ],
        )
        .await;
        let constraint = operation(
            runtime,
            "optim.problemdef.OptimizationExpression.le",
            &x,
            Value::Num(cap),
        )
        .await;
        let problem = call(
            runtime,
            "optimproblem",
            &[
                Value::from("Objective"),
                x.value().clone(),
                Value::from("ObjectiveSense"),
                Value::from("max"),
                Value::from("Constraints"),
                constraint.value().clone(),
            ],
        )
        .await;
        let start = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
            "semi".into(),
            Value::Num(0.),
        )]));
        let result = solve(runtime, &problem, Some(&start), options)
            .await
            .unwrap();
        let values = outputs(&result);
        assert_eq!(numbers(field(&values[0], "semi")), vec![expected]);
        assert_eq!(numbers(&values[1]), vec![expected]);
        assert_eq!(
            call(runtime, "double", &[values[2].clone()]).await.value(),
            &Value::Num(1.),
            "{domain}: {:?}",
            values[3]
        );
        assert_eq!(field(&values[3], "Solver"), &Value::from("intlinprog"));
        if domain == "semi-continuous" {
            let one = operation(
                runtime,
                "optim.problemdef.OptimizationExpression.power",
                &x,
                Value::Num(0.),
            )
            .await;
            let square = operation(
                runtime,
                "optim.problemdef.OptimizationExpression.power",
                &one,
                Value::Num(2.),
            )
            .await;
            let constant = operation(
                runtime,
                "optim.problemdef.OptimizationExpression.minus",
                &square,
                Value::Num(5.),
            )
            .await;
            let problem = call(
                runtime,
                "optimproblem",
                &[
                    Value::from("Objective"),
                    constant.value().clone(),
                    Value::from("ObjectiveSense"),
                    Value::from("max"),
                    Value::from("Constraints"),
                    constraint.value().clone(),
                ],
            )
            .await;
            let result = solve(runtime, &problem, Some(&start), options)
                .await
                .unwrap();
            let values = outputs(&result);
            assert_eq!(numbers(&values[1]), vec![-4.]);
            assert_eq!(numbers(field(&values[3], "certifiedobjective")), vec![4.]);
            assert_eq!(numbers(field(&values[3], "lowerbound")), vec![4.]);
        }
    }
}
