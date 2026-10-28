//! One compact extension of the existing original-terminal mechanism control.
use super::*;
use crate::builtins::math::optim::common::proof_rational::Rational as Exact;
async fn check(
    runtime: &RuntimeContext,
    value: &RootedValue,
    point: &[f64],
    expected: &[f64],
    gradient: &[f64],
) {
    let policy = policy();
    let mut c = policy.cooperation(Some(runtime)).await.unwrap();
    let root = owner::reference(value.value(), policy).unwrap();
    let program = compile(&[root], policy, &mut c).await.unwrap().unwrap();
    assert_eq!(program.layout.columns, point.len());
    let array = program.array(root, policy).unwrap();
    let mut proposal = (0..expected.len())
        .map(|row| array.offset.at(row, policy).unwrap())
        .collect::<Vec<_>>();
    for term in array.coefficients.entries() {
        proposal[term.row] += term.value * point[term.column];
    }
    for (actual, expected) in proposal.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let point = point
        .iter()
        .map(|&v| Exact::from_float(v).unwrap())
        .collect::<Vec<_>>();
    let exact = exact::evaluate(
        &program,
        exact::Coordinates::Point(&point),
        exact::Retain::Roots,
        policy,
        &mut c,
    )
    .await
    .unwrap();
    for (row, &expected) in expected.iter().enumerate() {
        assert!(exact
            .value(root, row, policy)
            .unwrap()
            .unwrap()
            .compare(&Exact::from_float(expected).unwrap(), policy, &mut c)
            .await
            .unwrap()
            .is_eq());
    }
    let constants = exact::evaluate(
        &program,
        exact::Coordinates::Origin,
        exact::Retain::ConstantOperands,
        policy,
        &mut c,
    )
    .await
    .unwrap();
    let seeds = (0..expected.len())
        .map(|row| exact::Seed {
            root,
            row,
            value: Exact::from_usize(1),
        })
        .collect();
    let actual = exact::accumulate(&program, &constants, seeds, policy, &mut c)
        .await
        .unwrap();
    assert_eq!(actual.len(), gradient.len());
    for (actual, &expected) in actual.iter().zip(gradient) {
        assert!(actual
            .compare(&Exact::from_float(expected).unwrap(), policy, &mut c)
            .await
            .unwrap()
            .is_eq());
    }
}
async fn rejected(runtime: &RuntimeContext, value: &RootedValue, expected: Stop) {
    let policy = policy();
    let mut c = policy.cooperation(Some(runtime)).await.unwrap();
    let root = owner::reference(value.value(), policy).unwrap();
    assert!(matches!(compile(&[root],policy,&mut c).await.unwrap(),Err(stop) if stop==expected));
}
pub(super) async fn controls(runtime: &RuntimeContext) {
    let foreign = crate::user_functions::test_support::exact_fixture_runtime();
    let _ambient = foreign.enter();
    let x = call(runtime, "optimvar", &[Value::from("factor")]).await;
    let column = call(
        runtime,
        "optim.problemdef.OptimizationExpression.vertcat",
        &[x.value().clone(), Value::Num(2.)],
    )
    .await;
    let opposite = call(
        runtime,
        "optim.problemdef.OptimizationExpression.vertcat",
        &[Value::Num(3.), x.value().clone()],
    )
    .await;
    let row = call(
        runtime,
        "optim.problemdef.OptimizationExpression.reshape",
        &[opposite.value().clone(), Value::Num(1.), Value::Num(2.)],
    )
    .await;
    let dot = call(
        runtime,
        "optim.problemdef.OptimizationExpression.dot",
        &[column.value().clone(), row.value().clone()],
    )
    .await;
    check(runtime, &dot, &[3.], &[15.], &[5.]).await;
    let row = call(
        runtime,
        "optim.problemdef.OptimizationExpression.reshape",
        &[column.value().clone(), Value::Num(1.), Value::Num(2.)],
    )
    .await;
    let matrix = call(
        runtime,
        "optim.problemdef.OptimizationExpression.mtimes",
        &[row.value().clone(), opposite.value().clone()],
    )
    .await;
    check(runtime, &matrix, &[3.], &[15.], &[5.]).await;
    let product = call(
        runtime,
        "optim.problemdef.OptimizationExpression.prod",
        &[column.value().clone()],
    )
    .await;
    check(runtime, &product, &[3.], &[6.], &[2.]).await;
    let prefix = call(
        runtime,
        "optim.problemdef.OptimizationExpression.cumprod",
        &[column.value().clone()],
    )
    .await;
    check(runtime, &prefix, &[3.], &[3., 6.], &[3.]).await;
    let suffix = call(
        runtime,
        "optim.problemdef.OptimizationExpression.cumprod",
        &[column.value().clone(), Value::from("reverse")],
    )
    .await;
    check(runtime, &suffix, &[3.], &[6., 2.], &[2.]).await;
    let three = call(
        runtime,
        "optim.problemdef.OptimizationExpression.vertcat",
        &[x.value().clone(), Value::Num(2.), x.value().clone()],
    )
    .await;
    let difference = call(
        runtime,
        "optim.problemdef.OptimizationExpression.diff",
        &[three.value().clone()],
    )
    .await;
    check(runtime, &difference, &[3.], &[-1., 1.], &[0.]).await;
    let second = call(
        runtime,
        "optim.problemdef.OptimizationExpression.diff",
        &[three.value().clone(), Value::Num(2.)],
    )
    .await;
    check(runtime, &second, &[3.], &[2.], &[2.]).await;
    for (exponent, expected, gradient) in [(0., 1., 0.), (1., 3., 1.)] {
        let power = call(
            runtime,
            "optim.problemdef.OptimizationExpression.power",
            &[x.value().clone(), Value::Num(exponent)],
        )
        .await;
        check(runtime, &power, &[3.], &[expected], &[gradient]).await;
    }
    let square = call(
        runtime,
        "optim.problemdef.OptimizationExpression.power",
        &[x.value().clone(), Value::Num(2.)],
    )
    .await;
    rejected(runtime, &square, Stop::NonAffine).await;
    let nonlinear_product = call(
        runtime,
        "optim.problemdef.OptimizationExpression.prod",
        &[opposite.value().clone(), Value::Num(2.)],
    )
    .await;
    // Dim 2 is an identity reduction; the default reduction below is not.
    check(runtime, &nonlinear_product, &[3.], &[3., 3.], &[1.]).await;
    let repeated = call(
        runtime,
        "optim.problemdef.OptimizationExpression.vertcat",
        &[x.value().clone(), x.value().clone()],
    )
    .await;
    let nonlinear_product = call(
        runtime,
        "optim.problemdef.OptimizationExpression.prod",
        &[repeated.value().clone()],
    )
    .await;
    rejected(runtime, &nonlinear_product, Stop::NonAffine).await;
    let cancelled = call(
        runtime,
        "optim.problemdef.OptimizationExpression.minus",
        &[x.value().clone(), x.value().clone()],
    )
    .await;
    let false_product = call(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &[x.value().clone(), cancelled.value().clone()],
    )
    .await;
    rejected(runtime, &false_product, Stop::NonAffine).await;
    let integer = call(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        &[Value::Num(1.), Value::Num(1.)],
    )
    .await;
    let power = call(
        runtime,
        "optim.problemdef.OptimizationExpression.power",
        &[Value::Num(2.), integer.value().clone()],
    )
    .await;
    check(runtime, &power, &[], &[4.], &[]).await;
    let rounded = call(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        &[Value::Num(1.), Value::Num(2f64.powi(-54))],
    )
    .await;
    let false_identity = call(
        runtime,
        "optim.problemdef.OptimizationExpression.power",
        &[x.value().clone(), rounded.value().clone()],
    )
    .await;
    rejected(runtime, &false_identity, Stop::UnsupportedFunction).await;
    let error = crate::call_builtin_async_in_context(
        runtime,
        "optim.problemdef.OptimizationExpression.power",
        &[Value::Num(2.), cancelled.value().clone()],
    )
    .await
    .expect_err("numerical cancellation does not make a variable-backed exponent constant");
    assert_eq!(error.identifier(), Some("RunMat:OptimModel:InvalidInput"));
    // Identity powers keep the original immutable numeric owner, not a
    // new count-sized destination that happens to have the same values.
    let backing = runmat_value::Tensor::new(vec![2., 4.], vec![2, 1]).unwrap();
    let retained = call(
        runtime,
        "optim.problemdef.OptimizationExpression.power",
        &[Value::Tensor(backing.clone()), Value::Num(1.)],
    )
    .await;
    let policy = policy();
    let mut c = policy.cooperation(Some(runtime)).await.unwrap();
    let root = owner::reference(retained.value(), policy).unwrap();
    let program = compile(&[root], policy, &mut c).await.unwrap().unwrap();
    let array::Offset::Dense(actual) = &program.array(root, policy).unwrap().offset else {
        panic!("retained identity backing");
    };
    assert!(actual
        .host_buffer()
        .shares_allocation_with(backing.host_buffer()));
    let overflow = call(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &[Value::Num(f64::MAX), Value::Num(2.)],
    )
    .await;
    let masked = call(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &[Value::Num(0.), overflow.value().clone()],
    )
    .await;
    rejected(runtime, &masked, Stop::NonFinite).await;
    let matrix_variable = call(
        runtime,
        "optimvar",
        &[Value::from("matrixFactor"), Value::Num(2.), Value::Num(2.)],
    )
    .await;
    let identity = call(
        runtime,
        "optim.problemdef.OptimizationExpression.mpower",
        &[matrix_variable.value().clone(), Value::Num(0.)],
    )
    .await;
    check(
        runtime,
        &identity,
        &[1., 2., 3., 4.],
        &[1., 0., 0., 1.],
        &[0.; 4],
    )
    .await;
    let empty = Value::Tensor(runmat_value::Tensor::new(Vec::new(), vec![0, 0]).unwrap());
    let empty_inverse = call(
        runtime,
        "optim.problemdef.OptimizationExpression.mpower",
        &[empty, Value::Num(-1.)],
    )
    .await;
    check(runtime, &empty_inverse, &[], &[], &[]).await;
    let exhausted = call(
        runtime,
        "optim.problemdef.OptimizationExpression.diff",
        &[column.value().clone(), Value::Num(4.)],
    )
    .await;
    check(runtime, &exhausted, &[3.], &[], &[0.]).await;
    let inverse = call(
        runtime,
        "optim.problemdef.OptimizationExpression.mpower",
        &[
            Value::Tensor(runmat_value::Tensor::new(vec![2., 0., 1., 4.], vec![2, 2]).unwrap()),
            Value::Num(-1.),
        ],
    )
    .await;
    check(runtime, &inverse, &[], &[0.5, 0., -0.125, 0.25], &[]).await;
}
