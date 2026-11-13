//! Nonvanishing original-data normalization in the existing public control.
use super::*;
pub(super) async fn check(runtime: &RuntimeContext) {
    let x = call(runtime, "optimvar", &[Value::from("nonbinary")]).await;
    let triple = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &x,
        Value::Num(3.),
    )
    .await;
    let residual = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.minus",
        &triple,
        Value::Num(1.),
    )
    .await;
    let square = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.power",
        &residual,
        Value::Num(2.),
    )
    .await;
    let problem = call(
        runtime,
        "optimproblem",
        &[Value::from("Objective"), square.value().clone()],
    )
    .await;
    let output = crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "solve",
        &[problem.value().clone()],
        3,
    )
    .await
    .unwrap();
    let Value::OutputList(output) = output.value() else {
        panic!("original nonbinary optimum")
    };
    assert!((numbers(field(&output[0], "nonbinary"))[0] - 1. / 3.).abs() < 1e-6);
    assert_eq!(
        call(runtime, "double", &[output[2].clone()]).await.value(),
        &Value::Num(1.)
    );
    sparse_scale_rows(runtime).await;
    for scale in [1., 2f64.powi(-500)] {
        let scaled = operation(
            runtime,
            "optim.problemdef.OptimizationExpression.times",
            &residual,
            Value::Num(scale),
        )
        .await;
        let squared = operation(
            runtime,
            "optim.problemdef.OptimizationExpression.power",
            &scaled,
            Value::Num(2.),
        )
        .await;
        let problem = call(
            runtime,
            "optimproblem",
            &[Value::from("Objective"), squared.value().clone()],
        )
        .await;
        let mut c = ScalarCooperation::for_builtin(policy().identity, Some(runtime));
        let formulation = formulation::Formulation::prepare(
            crate::builtins::math::optim::optimmodel::problem::data(problem.value(), policy())
                .unwrap(),
            policy(),
            &mut c,
        )
        .await
        .unwrap();
        let Prepared::Algebraic(program) = objective::compile(&formulation, None, policy(), &mut c)
            .await
            .unwrap()
            .unwrap()
        else {
            panic!("affine residual program");
        };
        let mut authority = proof::Authority::new(&program, &formulation, &[], policy(), &mut c)
            .await
            .unwrap();
        assert!(authority
            .kkt_at_tolerance(&[1. / 3.], &[], &[], 1e-7, &mut c)
            .await
            .unwrap());
        assert!(!authority
            .kkt_at_tolerance(&[0.], &[], &[], 1e-7, &mut c)
            .await
            .unwrap());
    }
}

async fn sparse_scale_rows(runtime: &RuntimeContext) {
    use crate::builtins::math::optim::common::proof_rational::Rational as Exact;
    let width = 128usize;
    let x = call(
        runtime,
        "optimvar",
        &[
            Value::from("wide"),
            Value::Num(width as f64),
            Value::Num(1.),
        ],
    )
    .await;
    let matrix = runmat_value::SparseTensor::new(
        width,
        width,
        (0..=width).collect::<Vec<_>>(),
        (0..width).collect::<Vec<_>>(),
        vec![3.; width],
    )
    .unwrap();
    let residual = call(
        runtime,
        "optim.problemdef.OptimizationExpression.mtimes",
        &[Value::SparseTensor(matrix), x.value().clone()],
    )
    .await;
    let residual = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.minus",
        &residual,
        Value::Num(1.),
    )
    .await;
    let square = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.power",
        &residual,
        Value::Num(2.),
    )
    .await;
    let objective = call(
        runtime,
        "optim.problemdef.OptimizationExpression.sum",
        &[square.value().clone()],
    )
    .await;
    let problem = call(
        runtime,
        "optimproblem",
        &[Value::from("Objective"), objective.value().clone()],
    )
    .await;
    let mut c = ScalarCooperation::for_builtin(policy().identity, Some(runtime));
    let formulation = formulation::Formulation::prepare(
        crate::builtins::math::optim::optimmodel::problem::data(problem.value(), policy()).unwrap(),
        policy(),
        &mut c,
    )
    .await
    .unwrap();
    let Prepared::Algebraic(program) = objective::compile(&formulation, None, policy(), &mut c)
        .await
        .unwrap()
        .unwrap()
    else {
        panic!("affine residual program");
    };
    let origin = exact::evaluate(
        &program,
        exact::Coordinates::Origin,
        exact::Retain::ConstantOperands,
        policy(),
        &mut c,
    )
    .await
    .unwrap();
    let scale = program
        .objective
        .squares()
        .unwrap()
        .scale(&program, &origin, policy(), &mut c)
        .await
        .unwrap();
    for column in [0, width - 1] {
        let value = scale
            .at(column, &Exact::zero(), policy(), &mut c)
            .await
            .unwrap();
        assert!(value
            .compare(&Exact::from_usize(24), policy(), &mut c)
            .await
            .unwrap()
            .is_eq());
    }
}
