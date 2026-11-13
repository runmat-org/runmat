//! Real model construction, sparse solve/publication and original certificate
//! boundaries. No numerical callback or matrix is substituted for the DAG.
use super::super::super::linprog::compiled::{CertificateCandidate, RowRole};
use super::*;
mod equation;
mod integer;
mod least_squares;
mod nonlinear;
mod products;
mod quadratic;
mod residual;
use futures::executor::block_on;
use num_rational::BigRational as Rational;
async fn call(runtime: &RuntimeContext, name: &str, args: &[Value]) -> RootedValue {
    crate::call_builtin_async_in_context(runtime, name, args)
        .await
        .unwrap()
}
async fn operation(
    runtime: &RuntimeContext,
    name: &str,
    left: &RootedValue,
    right: Value,
) -> RootedValue {
    call(runtime, name, &[left.value().clone(), right]).await
}
async fn model(
    runtime: &RuntimeContext,
    objective: &RootedValue,
    constraint: &RootedValue,
) -> RootedValue {
    call(
        runtime,
        "optimproblem",
        &[
            Value::from("Objective"),
            objective.value().clone(),
            Value::from("Constraints"),
            constraint.value().clone(),
        ],
    )
    .await
}
fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Struct(fields) = value else {
        panic!("original output structure");
    };
    &fields.fields[name]
}
fn numbers(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(value) => vec![*value],
        Value::Tensor(value) => value.materialize_f64(),
        _ => panic!("double output"),
    }
}
async fn compile(
    roots: &[runmat_gc::GcHandle],
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Program> {
    let source = evaluation::Plan::prepare_roots(roots, policy, c).await?;
    let layout = layout::Layout::prepare(&source, policy, c).await?;
    engine::compile(source, layout, None, None, roots, policy, c).await
}
fn policy() -> Policy {
    Policy {
        identity: runmat_builtins::OPTIMPROBLEM_SOLVE_CATALOG_ENTRY.identity,
        invalid: &runmat_builtins::OPTIMPROBLEM_ERROR_INVALID_INPUT,
    }
}
#[test]
fn sparse_public_solve_keeps_original_layout_objective_duals_and_context() {
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
                Value::Num(2.),
                Value::from("LowerBound"),
                Value::Num(0.),
            ],
        )
        .await;
        let flat = call(
            &original,
            "optim.problemdef.OptimizationExpression.reshape",
            &[x.value().clone(), Value::Num(4.), Value::Num(1.)],
        )
        .await;
        let cost = call(
            &original,
            "optim.problemdef.OptimizationExpression.mtimes",
            &[
                Value::Tensor(runmat_value::Tensor::new(vec![3., 1., 2., 1.], vec![1, 4]).unwrap()),
                flat.value().clone(),
            ],
        )
        .await;
        let cost = operation(
            &original,
            "optim.problemdef.OptimizationExpression.plus",
            &cost,
            Value::Num(7.),
        )
        .await;
        let total = call(
            &original,
            "optim.problemdef.OptimizationExpression.mtimes",
            &[
                Value::SparseTensor(
                    runmat_value::SparseTensor::new(
                        1,
                        4,
                        vec![0, 1, 2, 3, 4],
                        vec![0, 0, 0, 0],
                        vec![1.; 4],
                    )
                    .unwrap(),
                ),
                flat.value().clone(),
            ],
        )
        .await;
        let capacity = operation(
            &original,
            "optim.problemdef.OptimizationExpression.le",
            &total,
            Value::Num(4.),
        )
        .await;
        let problem = call(
            &original,
            "optimproblem",
            &[
                Value::from("Objective"),
                cost.value().clone(),
                Value::from("ObjectiveSense"),
                Value::from("max"),
                Value::from("Constraints"),
                Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
                    "capacity".into(),
                    capacity.value().clone(),
                )])),
            ],
        )
        .await;
        let result = crate::call_builtin_async_with_outputs_in_context(
            &original,
            "solve",
            &[problem.value().clone()],
            5,
        )
        .await
        .unwrap();
        runmat_gc::gc_collect_major().unwrap();
        let Value::OutputList(outputs) = result.value() else {
            panic!("five original outputs");
        };
        assert_eq!(outputs.len(), 5);
        let allocation = field(&outputs[0], "allocation");
        let Value::Tensor(array) = allocation else {
            panic!("original array solution");
        };
        assert_eq!(array.shape(), &[2, 2]);
        let values = numbers(allocation);
        assert!((values[0] - 4.).abs() < 1e-6);
        assert!((numbers(&outputs[1])[0] - 19.).abs() < 1e-6);
        let numeric = call(&original, "double", &[outputs[2].clone()]).await;
        assert_eq!(numeric.value(), &Value::Num(1.));
        let text = call(&original, "string", &[outputs[2].clone()]).await;
        assert_eq!(text.value(), &Value::from("OptimalSolution"));
        assert_eq!(
            call(&original, "eq", &[outputs[2].clone(), outputs[2].clone()])
                .await
                .value(),
            &Value::Bool(true)
        );
        assert_eq!(field(&outputs[3], "Solver"), &Value::from("linprog"));
        let bounds = field(field(&outputs[4], "Variables"), "allocation");
        let Value::Tensor(lower) = field(bounds, "Lower") else {
            panic!("original lower dual shape");
        };
        assert_eq!(lower.shape(), &[2, 2]);
        let constraint = field(field(&outputs[4], "Constraints"), "capacity");
        assert!((numbers(constraint)[0] - 3.).abs() < 1e-5);

        let limits = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
            "MaxIter".into(),
            Value::Num(0.),
        )]));
        let limited = crate::call_builtin_async_with_outputs_in_context(
            &original,
            "solve",
            &[problem.value().clone(), Value::from("Options"), limits],
            4,
        )
        .await
        .unwrap();
        let Value::OutputList(stopped) = limited.value() else {
            panic!("limited original outputs");
        };
        assert_eq!(
            call(&original, "double", &[stopped[2].clone()])
                .await
                .value(),
            &Value::Num(0.)
        );
        assert_eq!(field(&stopped[3], "iterations"), &Value::Num(0.));
        // A zero-iteration limit does not prescribe the backend's candidate.
        // Verify the original objective at the actual returned feasible point.
        let allocation = field(&stopped[0], "allocation");
        let Value::Tensor(array) = allocation else {
            panic!("stopped original array solution");
        };
        assert_eq!(array.shape(), &[2, 2]);
        let values = numbers(allocation);
        assert!(values
            .iter()
            .all(|value| value.is_finite() && *value >= -1e-7));
        assert!(values.iter().sum::<f64>() <= 4. + 1e-7);
        let expected = 7.
            + values
                .iter()
                .zip([3., 1., 2., 1.])
                .map(|(x, w)| x * w)
                .sum::<f64>();
        assert!((numbers(&stopped[1])[0] - expected).abs() < 1e-7);
        let bad = operation(
            &original,
            "optim.problemdef.OptimizationExpression.le",
            &total,
            Value::Num(-1.),
        )
        .await;
        let infeasible = model(&original, &cost, &bad).await;
        let result = crate::call_builtin_async_with_outputs_in_context(
            &original,
            "solve",
            &[infeasible.value().clone()],
            3,
        )
        .await
        .unwrap();
        let Value::OutputList(outputs) = result.value() else {
            panic!("infeasible original outputs");
        };
        assert!(matches!(field(&outputs[0],"allocation"),Value::Tensor(array) if array.is_empty()));
        assert!(matches!(&outputs[1],Value::Tensor(array) if array.is_empty()));
        assert_eq!(
            call(&original, "double", &[outputs[2].clone()])
                .await
                .value(),
            &Value::Num(-2.)
        );
    });
}
async fn tiny_affine(runtime: &RuntimeContext, x: &RootedValue) -> RootedValue {
    let tiny = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        x,
        Value::Num(2f64.powi(-53)),
    )
    .await;
    let first = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        x,
        tiny.value().clone(),
    )
    .await;
    operation(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        &first,
        tiny.value().clone(),
    )
    .await
}
#[test]
fn terminal_candidates_use_original_affine_law_not_rounded_coefficients() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    // Independent controls share the original runtime, not a single coroutine
    // containing every control's cold construction frame. Construct each
    // pinned fixture before entering its executor; retain every assertion.
    #[inline(never)]
    fn control<'a, F>(
        make: impl FnOnce() -> F,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>
    where
        F: std::future::Future<Output = ()> + 'a,
    {
        Box::pin(make())
    }
    block_on(control(|| linear_function_controls(&runtime)));
    block_on(control(|| products::controls(&runtime)));
    block_on(control(|| quadratic::controls(&runtime)));
    block_on(control(|| integer::controls(&runtime)));
    block_on(control(|| residual::controls(&runtime)));
    block_on(control(|| equation::controls(&runtime)));
    block_on(async {
        let policy = policy();
        crate::builtins::math::optim::common::proof_rational::check_arithmetic(policy).await;
        let mut cooperation = policy.cooperation(Some(&runtime)).await.unwrap();
        let x = call(
            &runtime,
            "optimvar",
            &[
                Value::from("x"),
                Value::from("LowerBound"),
                Value::Num(3.),
                Value::from("UpperBound"),
                Value::Num(3.),
            ],
        )
        .await;
        let law = tiny_affine(&runtime, &x).await;
        let constraint = operation(
            &runtime,
            "optim.problemdef.OptimizationExpression.ge",
            &law,
            Value::Num(3. + 2f64.powi(-51)),
        )
        .await;
        let problem = model(&runtime, &x, &constraint).await;
        let source = super::super::problem::data(problem.value(), policy).unwrap();
        let formulation = formulation::Formulation::prepare(source, policy, &mut cooperation)
            .await
            .unwrap();
        let program = compile(&formulation.roots, policy, &mut cooperation)
            .await
            .unwrap()
            .unwrap();
        let assembled = assembly::assemble(
            &program,
            &formulation,
            assembly::ObjectiveKind::Affine,
            policy,
            &mut cooperation,
        )
        .await
        .unwrap()
        .unwrap();
        let mut authority = proof::Authority::new(
            &program,
            &formulation,
            &assembled.rows,
            policy,
            &mut cooperation,
        )
        .await
        .unwrap();
        let roles = [RowRole::Inequality(0), RowRole::Upper(0)];
        let weights = [
            Rational::from_integer(1.into()),
            Rational::from_integer(1.into()),
        ];
        assert!(!authority
            .certificate(
                CertificateCandidate::Farkas {
                    rows: &roles,
                    multipliers: &weights
                },
                &mut cooperation
            )
            .await
            .unwrap());
        assert!(authority
            .certificate(
                CertificateCandidate::Feasible(&[Rational::from_integer(3.into())]),
                &mut cooperation
            )
            .await
            .unwrap());
        drop(authority);
        drop(assembled);
        drop(program);

        let free = call(&runtime, "optimvar", &[Value::from("free")]).await;
        let law = tiny_affine(&runtime, &free).await;
        let slope = operation(
            &runtime,
            "optim.problemdef.OptimizationExpression.minus",
            &law,
            free.value().clone(),
        )
        .await;
        let constraint = operation(
            &runtime,
            "optim.problemdef.OptimizationExpression.le",
            &slope,
            Value::Num(0.),
        )
        .await;
        let cost = operation(
            &runtime,
            "optim.problemdef.OptimizationExpression.times",
            &free,
            Value::Num(-1.),
        )
        .await;
        let problem = model(&runtime, &cost, &constraint).await;
        let source = super::super::problem::data(problem.value(), policy).unwrap();
        let formulation = formulation::Formulation::prepare(source, policy, &mut cooperation)
            .await
            .unwrap();
        let program = compile(&formulation.roots, policy, &mut cooperation)
            .await
            .unwrap()
            .unwrap();
        let assembled = assembly::assemble(
            &program,
            &formulation,
            assembly::ObjectiveKind::Affine,
            policy,
            &mut cooperation,
        )
        .await
        .unwrap()
        .unwrap();
        let mut authority = proof::Authority::new(
            &program,
            &formulation,
            &assembled.rows,
            policy,
            &mut cooperation,
        )
        .await
        .unwrap();
        let direction = [Rational::from_integer(1.into())];
        assert!(!authority
            .certificate(CertificateCandidate::Descent(&direction), &mut cooperation)
            .await
            .unwrap());
        assert!(authority
            .certificate(
                CertificateCandidate::Feasible(&[Rational::from_integer(0.into())]),
                &mut cooperation
            )
            .await
            .unwrap());
    });
}

async fn linear_function_controls(runtime: &RuntimeContext) {
    let foreign = crate::user_functions::test_support::exact_fixture_runtime();
    let _ambient = foreign.enter();
    let x = call(
        runtime,
        "optimvar",
        &[
            Value::from("rate"),
            Value::Num(2.),
            Value::Num(1.),
            Value::from("LowerBound"),
            Value::Num(0.),
            Value::from("UpperBound"),
            Value::Num(2.),
        ],
    )
    .await;
    let weights = runmat_value::Tensor::new(vec![1., 3.], vec![2, 1]).unwrap();
    let mean = call(
        runtime,
        "optim.problemdef.OptimizationExpression.mean",
        &[
            x.value().clone(),
            Value::Num(1.),
            Value::from("Weights"),
            Value::Tensor(weights),
        ],
    )
    .await;
    let diagonal = call(
        runtime,
        "optim.problemdef.OptimizationExpression.diag",
        &[x.value().clone()],
    )
    .await;
    let trace = call(
        runtime,
        "optim.problemdef.OptimizationExpression.trace",
        &[diagonal.value().clone()],
    )
    .await;
    let scan = call(
        runtime,
        "optim.problemdef.OptimizationExpression.cumsum",
        &[x.value().clone(), Value::from("reverse")],
    )
    .await;
    let total = call(
        runtime,
        "optim.problemdef.OptimizationExpression.sum",
        &[scan.value().clone()],
    )
    .await;
    let objective = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        &mean,
        trace.value().clone(),
    )
    .await;
    let objective = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        &objective,
        total.value().clone(),
    )
    .await;
    let amount = call(
        runtime,
        "optim.problemdef.OptimizationExpression.sum",
        &[x.value().clone()],
    )
    .await;
    let constraint = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.ge",
        &amount,
        Value::Num(2.),
    )
    .await;
    let problem = model(runtime, &objective, &constraint).await;
    let result = crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "solve",
        &[problem.value().clone()],
        5,
    )
    .await
    .unwrap();
    let Value::OutputList(values) = result.value() else {
        panic!("five original solve outputs")
    };
    assert_eq!(values.len(), 5);
    let point = numbers(field(&values[0], "rate"));
    assert!((point[0] - 2.).abs() < 1e-5 && point[1].abs() < 1e-5);
    assert!((numbers(&values[1])[0] - 4.5).abs() < 1e-5);
    assert_eq!(
        call(runtime, "double", &[values[2].clone()]).await.value(),
        &Value::Num(1.)
    );
    let evaluated = call(
        runtime,
        "evaluate",
        &[objective.value().clone(), values[0].clone()],
    )
    .await;
    assert!((numbers(evaluated.value())[0] - numbers(&values[1])[0]).abs() < 1e-12);
    let extracted = call(
        runtime,
        "optim.problemdef.OptimizationExpression.diag",
        &[diagonal.value().clone()],
    )
    .await;
    let evaluated = call(
        runtime,
        "evaluate",
        &[extracted.value().clone(), values[0].clone()],
    )
    .await;
    assert_eq!(numbers(evaluated.value()), point);
}
