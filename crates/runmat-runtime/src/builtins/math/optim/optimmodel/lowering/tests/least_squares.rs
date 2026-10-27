//! One actual sparse residual consumer: syntax joins, original output units,
//! stopped points and original-domain terminal rejection share the public path.
use super::*;
mod active;
mod reflective;
mod scales;
#[test]
fn squared_residual_model_preserves_original_dag_shapes_duals_and_terminal_authority() {
    block_on(async {
        let original = crate::user_functions::test_support::exact_fixture_runtime();
        let foreign = crate::user_functions::test_support::exact_fixture_runtime();
        let _ambient = foreign.enter();
        let x = call(
            &original,
            "optimvar",
            &[
                Value::from("allocation"),
                Value::Num(2.),
                Value::Num(1.),
                Value::from("LowerBound"),
                Value::Num(0.),
            ],
        )
        .await;
        let residual = operation(
            &original,
            "optim.problemdef.OptimizationExpression.minus",
            &x,
            Value::Tensor(runmat_value::Tensor::new(vec![2., 4.], vec![2, 1]).unwrap()),
        )
        .await;
        let squared = operation(
            &original,
            "optim.problemdef.OptimizationExpression.power",
            &residual,
            Value::Num(2.),
        )
        .await;
        let sum = call(
            &original,
            "optim.problemdef.OptimizationExpression.sum",
            &[squared.value().clone()],
        )
        .await;
        let twice = operation(
            &original,
            "optim.problemdef.OptimizationExpression.times",
            &sum,
            Value::Num(2.),
        )
        .await;
        let norm = call(
            &original,
            "optim.problemdef.OptimizationExpression.norm",
            &[residual.value().clone()],
        )
        .await;
        let norm = operation(
            &original,
            "optim.problemdef.OptimizationExpression.mpower",
            &norm,
            Value::Num(2.),
        )
        .await;
        let objective = operation(
            &original,
            "optim.problemdef.OptimizationExpression.plus",
            &twice,
            norm.value().clone(),
        )
        .await;
        let objective = operation(
            &original,
            "optim.problemdef.OptimizationExpression.plus",
            &objective,
            Value::Num(7.),
        )
        .await;
        let total = call(
            &original,
            "optim.problemdef.OptimizationExpression.sum",
            &[x.value().clone()],
        )
        .await;
        let cap = operation(
            &original,
            "optim.problemdef.OptimizationExpression.le",
            &total,
            Value::Num(3.),
        )
        .await;
        let problem = model(&original, &objective, &cap).await;
        let starting = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
            "allocation".into(),
            Value::Tensor(runmat_value::Tensor::new(vec![100., 200.], vec![2, 1]).unwrap()),
        )]));
        let output = crate::call_builtin_async_with_outputs_in_context(
            &original,
            "solve",
            &[
                problem.value().clone(),
                starting.clone(),
                Value::from("Solver"),
                Value::from("lsqlin"),
            ],
            5,
        )
        .await
        .unwrap();
        runmat_gc::gc_collect_major().unwrap();
        let Value::OutputList(values) = output.value() else {
            panic!("five original model outputs");
        };
        assert_eq!(values.len(), 5);
        let allocation = field(&values[0], "allocation");
        assert!(matches!(allocation,Value::Tensor(value)if value.shape()==[2,1]));
        let point = numbers(allocation);
        assert!((point[0] - 0.5).abs() < 1e-5 && (point[1] - 2.5).abs() < 1e-5);
        assert!((numbers(&values[1])[0] - 20.5).abs() < 1e-5);
        assert_eq!(
            call(&original, "double", &[values[2].clone()])
                .await
                .value(),
            &Value::Num(1.)
        );
        assert_eq!(field(&values[3], "Solver"), &Value::from("lsqlin"));
        assert_eq!(
            field(&values[3], "algorithm"),
            &Value::from("interior-point")
        );
        assert!(numbers(field(&values[3], "wallclocktime"))[0] >= 0.);
        assert!((numbers(field(&values[4], "Constraints"))[0] - 9.).abs() < 1e-4);
        assert_eq!(numbers(field(&starting, "allocation")), vec![100., 200.]);
        let evaluated = call(
            &original,
            "evaluate",
            &[objective.value().clone(), values[0].clone()],
        )
        .await;
        assert_eq!(numbers(evaluated.value()), numbers(&values[1]));

        reflective::check(&original, &objective, &starting).await;
        active::check(&original, &problem, &objective, &starting).await;

        scales::check(&original).await;

        // Same original residual node reached through norm^2 and sum(.^2):
        // one actual residual lift with exact accumulated weight three.
        let mut cooperation = ScalarCooperation::for_builtin(policy().identity, Some(&original));
        let root = owner::reference(objective.value(), policy()).unwrap();
        let source = evaluation::Plan::prepare(root, policy(), &mut cooperation)
            .await
            .unwrap();
        let objective::Classification::LeastSquares(classified) =
            objective::classify(&source, root, policy(), &mut cooperation)
                .await
                .unwrap()
        else {
            panic!("selected squared residual syntax");
        };
        assert_eq!(classified.terms.len(), 1);
        assert_eq!(classified.terms[0].weight.proposal, 3.);

        // A matrix's default/2 norm is spectral, not an entrywise square sum.
        let matrix = call(
            &original,
            "optimvar",
            &[Value::from("matrix"), Value::Num(2.), Value::Num(2.)],
        )
        .await;
        for arguments in [vec![], vec![Value::Num(2.)]] {
            let mut args = vec![matrix.value().clone()];
            args.extend(arguments);
            let spectral = call(
                &original,
                "optim.problemdef.OptimizationExpression.norm",
                &args,
            )
            .await;
            let point = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
                "matrix".into(),
                Value::Tensor(runmat_value::Tensor::new(vec![3., 0., 0., 4.], vec![2, 2]).unwrap()),
            )]));
            assert_eq!(
                numbers(
                    call(&original, "evaluate", &[spectral.value().clone(), point])
                        .await
                        .value()
                ),
                vec![4.],
                "matrix spectral norm must not become the Frobenius norm of five",
            );
            let spectral = operation(
                &original,
                "optim.problemdef.OptimizationExpression.mpower",
                &spectral,
                Value::Num(2.),
            )
            .await;
            let root = owner::reference(spectral.value(), policy()).unwrap();
            let source = evaluation::Plan::prepare(root, policy(), &mut cooperation)
                .await
                .unwrap();
            assert!(matches!(
                objective::classify(&source, root, policy(), &mut cooperation)
                    .await
                    .unwrap(),
                objective::Classification::Unrecognized
            ));
        }

        let options = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
            "MaxIter".into(),
            Value::Num(0.),
        )]));
        let stopped = crate::call_builtin_async_with_outputs_in_context(
            &original,
            "solve",
            &[problem.value().clone(), Value::from("Options"), options],
            3,
        )
        .await
        .unwrap();
        let Value::OutputList(stopped) = stopped.value() else {
            panic!("stopped model point");
        };
        assert_eq!(numbers(field(&stopped[0], "allocation")), vec![0., 0.]);
        assert!((numbers(&stopped[1])[0] - 67.).abs() < 1e-12);
        assert_eq!(
            call(&original, "double", &[stopped[2].clone()])
                .await
                .value(),
            &Value::Num(0.)
        );

        let impossible = operation(
            &original,
            "optim.problemdef.OptimizationExpression.le",
            &total,
            Value::Num(-1.),
        )
        .await;
        let impossible = model(&original, &objective, &impossible).await;
        let result = crate::call_builtin_async_with_outputs_in_context(
            &original,
            "solve",
            &[impossible.value().clone()],
            3,
        )
        .await
        .unwrap();
        let Value::OutputList(result) = result.value() else {
            panic!("infeasible result");
        };
        assert!(matches!(&result[1],Value::Tensor(value)if value.is_empty()));
        assert_eq!(
            call(&original, "double", &[result[2].clone()])
                .await
                .value(),
            &Value::Num(-2.)
        );

        // Rounded assembled rows say 0>=1; the original tiny affine law is
        // feasible at finite large x, so it must not authorize infeasibility.
        let tiny = tiny_affine(&original, &x).await;
        let tiny = operation(
            &original,
            "optim.problemdef.OptimizationExpression.minus",
            &tiny,
            x.value().clone(),
        )
        .await;
        let constraint = operation(
            &original,
            "optim.problemdef.OptimizationExpression.ge",
            &tiny,
            Value::Num(1.),
        )
        .await;
        let unverified = model(&original, &objective, &constraint).await;
        let error = crate::call_builtin_async_with_outputs_in_context(
            &original,
            "solve",
            &[unverified.value().clone()],
            3,
        )
        .await
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::OPTIMPROBLEM_SOLVE_ERROR_UNVERIFIED.identifier
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        // Original structural constancy, not cancelled working coefficients:
        // preserve the pre-existing affine/maximization feasibility route.
        let ones = operation(
            &original,
            "optim.problemdef.OptimizationExpression.power",
            &x,
            Value::Num(0.),
        )
        .await;
        let constant = call(
            &original,
            "optim.problemdef.OptimizationExpression.sum",
            &[ones.value().clone()],
        )
        .await;
        let constant = operation(
            &original,
            "optim.problemdef.OptimizationExpression.power",
            &constant,
            Value::Num(2.),
        )
        .await;
        let constant = call(
            &original,
            "optimproblem",
            &[
                Value::from("Objective"),
                constant.value().clone(),
                Value::from("ObjectiveSense"),
                Value::from("max"),
                Value::from("Constraints"),
                cap.value().clone(),
            ],
        )
        .await;
        let result = crate::call_builtin_async_with_outputs_in_context(
            &original,
            "solve",
            &[constant.value().clone()],
            4,
        )
        .await
        .unwrap();
        let Value::OutputList(result) = result.value() else {
            panic!("constant objective result");
        };
        assert_eq!(numbers(&result[1]), vec![4.]);
        assert_eq!(field(&result[3], "Solver"), &Value::from("linprog"));
        assert_eq!(
            call(&original, "double", &[result[2].clone()])
                .await
                .value(),
            &Value::Num(1.)
        );
    });
}
