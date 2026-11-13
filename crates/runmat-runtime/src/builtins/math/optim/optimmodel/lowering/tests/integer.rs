//! Integer model consumer and original bound/certificate checks share the
//! existing public original-terminal control. No alternate MILP is installed.
use super::*;
use crate::builtins::math::optim::common::proof_rational::Rational as Exact;
use std::sync::{Arc, Mutex};
mod semi;
fn outputs(value: &RootedValue) -> &[Value] {
    let Value::OutputList(values) = value.value() else {
        panic!("actual model outputs");
    };
    values
}
async fn solve(
    runtime: &RuntimeContext,
    model: &RootedValue,
    start: Option<&Value>,
    options: &RootedValue,
) -> BuiltinResult<RootedValue> {
    let mut values = vec![model.value().clone()];
    if let Some(start) = start {
        values.push(start.clone());
    }
    values.extend([Value::from("Options"), options.value().clone()]);
    crate::call_builtin_async_with_outputs_in_context(runtime, "solve", &values, 5).await
}
pub(super) async fn controls(runtime: &RuntimeContext) {
    let foreign = crate::user_functions::test_support::exact_fixture_runtime();
    let _ambient = foreign.enter();
    let x = call(
        runtime,
        "optimvar",
        &[
            Value::from("count"),
            Value::Num(2.),
            Value::Num(1.),
            Value::from("Type"),
            Value::from("integer"),
            Value::from("LowerBound"),
            Value::Num(0.),
            Value::from("UpperBound"),
            Value::Num(1.),
        ],
    )
    .await;
    let sum = call(
        runtime,
        "optim.problemdef.OptimizationExpression.sum",
        &[x.value().clone()],
    )
    .await;
    let cost = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &sum,
        Value::Num(3.),
    )
    .await;
    let cost = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        &cost,
        Value::Num(5.),
    )
    .await;
    let twice = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &sum,
        Value::Num(2.),
    )
    .await;
    let constraint = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.le",
        &twice,
        Value::Num(3.),
    )
    .await;
    let problem = call(
        runtime,
        "optimproblem",
        &[
            Value::from("Objective"),
            cost.value().clone(),
            Value::from("ObjectiveSense"),
            Value::from("max"),
            Value::from("Constraints"),
            constraint.value().clone(),
        ],
    )
    .await;
    let start = Value::Struct(runmat_value::StructValue::from_fields_unbounded([(
        "count".into(),
        Value::Tensor(runmat_value::Tensor::new(vec![1., 0.], vec![2, 1]).unwrap()),
    )]));
    let options = call(
        runtime,
        "optimoptions",
        &[
            Value::from("intlinprog"),
            Value::from("Display"),
            Value::from("off"),
            Value::from("MaxNodes"),
            Value::Num(100.),
        ],
    )
    .await;
    let result = solve(runtime, &problem, Some(&start), &options)
        .await
        .unwrap();
    let values = outputs(&result);
    let point = numbers(field(&values[0], "count"));
    assert_eq!(point.len(), 2);
    assert!(point.iter().all(|&v| v == 0. || v == 1.));
    assert_eq!(point.iter().sum::<f64>(), 1.);
    assert_eq!(numbers(&values[1]), vec![8.]);
    assert_eq!(
        call(runtime, "double", &[values[2].clone()]).await.value(),
        &Value::Num(1.)
    );
    assert_eq!(field(&values[3], "Solver"), &Value::from("intlinprog"));
    assert!(numbers(field(&values[3], "numfeaspoints"))[0] >= 1.);
    assert!(matches!(&values[4],Value::Tensor(value)if value.is_empty()));
    assert_eq!(
        numbers(
            call(
                runtime,
                "evaluate",
                &[cost.value().clone(), values[0].clone()]
            )
            .await
            .value()
        ),
        vec![8.]
    );
    observers(runtime, &problem, &start).await;
    lower_bounds(runtime).await;
    fractional_unbounded_relaxation(runtime, &options).await;
    semi::controls(runtime, &options).await;
}
async fn observers(runtime: &RuntimeContext, problem: &RootedValue, start: &Value) {
    for fail in [false, true] {
        let _binding = crate::user_functions::test_support::install_exact_named_forwarders(
            runtime,
            &[("integer_model_observer", 2160)],
        )
        .unwrap();
        let events = Arc::new(Mutex::new(Vec::<(String, Vec<f64>)>::new()));
        let observed = Arc::clone(&events);
        let _invoker = {
            // Install in the selected runtime, then restore the foreign ambient
            // runtime before invoking the model and its observer.
            let _entry = runtime.enter();
            crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
                move |id, args, requested| {
                    assert_eq!(id, 2160);
                    assert_eq!(requested, 1);
                    let state =
                        crate::builtins::common::validation::value_text_scalar_string(&args[2])
                            .unwrap();
                    let point = numbers(&args[0]);
                    let Value::Struct(info) = &args[1] else {
                        panic!("MILP metadata");
                    };
                    if !point.is_empty() {
                        assert_eq!(point.iter().sum::<f64>(), 1.);
                        assert_eq!(numbers(&info.fields["fval"]), vec![-8.]);
                    }
                    if state != "iter" {
                        assert!(point.is_empty());
                    }
                    observed
                        .lock()
                        .unwrap()
                        .push((state.clone(), point.clone()));
                    let response = if fail && !point.is_empty() {
                        Err(
                            crate::build_runtime_error("originating model observer error")
                                .with_identifier("Original:IntegerModelObserver")
                                .with_source(std::io::Error::other("original model observer cause"))
                                .build(),
                        )
                    } else {
                        Ok(Value::Bool(!point.is_empty()))
                    };
                    Box::pin(async move {
                        crate::user_functions::test_support::rooted_single_value_sequence(response?)
                    })
                },
            )))
            .unwrap()
        };
        let callbacks = Value::Cell(
            runmat_value::CellArray::new(
                vec![Value::FunctionHandle("integer_model_observer".into())],
                1,
                1,
            )
            .unwrap(),
        );
        let options = call(
            runtime,
            "optimoptions",
            &[
                Value::from("intlinprog"),
                Value::from("Display"),
                Value::from("off"),
                Value::from("OutputFcn"),
                callbacks,
            ],
        )
        .await;
        let result = solve(runtime, problem, Some(start), &options).await;
        if fail {
            let error = result.unwrap_err();
            assert_eq!(error.identifier(), Some("Original:IntegerModelObserver"));
            assert!(std::error::Error::source(&error)
                .unwrap()
                .is::<std::io::Error>());
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert_eq!(events.lock().unwrap().len(), 2);
        } else {
            let result = result.unwrap();
            let values = outputs(&result);
            assert_eq!(
                call(runtime, "string", &[values[2].clone()]).await.value(),
                &Value::from("OutputFcnStop")
            );
            assert_eq!(numbers(&values[1]), vec![8.]);
            assert_eq!(numbers(field(&values[3], "numnodes")), vec![0.]);
            assert_eq!(numbers(field(&values[3], "numfeaspoints")), vec![1.]);
            let events = events.lock().unwrap();
            assert_eq!(
                events.iter().map(|e| e.0.as_str()).collect::<Vec<_>>(),
                vec!["init", "iter", "done"]
            );
            assert_eq!(events.iter().filter(|e| !e.1.is_empty()).count(), 1);
        }
    }
}
async fn lower_bounds(runtime: &RuntimeContext) {
    let policy = policy();
    let mut c = policy.cooperation(Some(runtime)).await.unwrap();
    let x = call(
        runtime,
        "optimvar",
        &[
            Value::from("proofint"),
            Value::from("Type"),
            Value::from("integer"),
            Value::from("LowerBound"),
            Value::Num(0.),
            Value::from("UpperBound"),
            Value::Num(1.),
        ],
    )
    .await;
    let law = tiny_affine(runtime, &x).await;
    let negative = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &law,
        Value::Num(-1.),
    )
    .await;
    let cost = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.plus",
        &negative,
        Value::Num(5.),
    )
    .await;
    let problem = call(
        runtime,
        "optimproblem",
        &[Value::from("Objective"), cost.value().clone()],
    )
    .await;
    let formulation = formulation::Formulation::prepare(
        super::super::super::problem::data(problem.value(), policy).unwrap(),
        policy,
        &mut c,
    )
    .await
    .unwrap();
    let super::super::Prepared::Algebraic(program) =
        objective::compile(&formulation, None, policy, &mut c)
            .await
            .unwrap()
            .unwrap()
    else {
        panic!("affine integer source");
    };
    let assembled = assembly::assemble_domains(
        &program,
        &formulation,
        assembly::ObjectiveKind::Affine,
        assembly::Domains::Integer,
        policy,
        &mut c,
    )
    .await
    .unwrap()
    .unwrap();
    let assembly::Cost::Affine(f) = &assembled.cost else {
        panic!("affine cost");
    };
    assert_eq!(f, &[-1.]);
    assert_eq!(assembled.integers, vec![0]);
    let mut authority =
        proof::Authority::new(&program, &formulation, &assembled.rows, policy, &mut c)
            .await
            .unwrap();
    let weights = [Exact::zero(), Exact::from_usize(1)];
    let bound = authority
        .integer_lower_bound(
            &[RowRole::Lower(0), RowRole::Upper(0)],
            &weights,
            proof::BoundValues {
                lower: &[0.],
                upper: &[1.],
            },
            &mut c,
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        bound.export(policy, &mut c).await.unwrap(),
        Rational::from_integer(4.into()) - Rational::from_float(2f64.powi(-52)).unwrap()
    );
    assert!(authority
        .integer_lower_bound(
            &[],
            &[],
            proof::BoundValues {
                lower: &[0.],
                upper: &[f64::INFINITY]
            },
            &mut c
        )
        .await
        .unwrap()
        .is_none());
    let constraint = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.ge",
        &x,
        Value::Num(1.),
    )
    .await;
    let problem = model(runtime, &cost, &constraint).await;
    let formulation = formulation::Formulation::prepare(
        super::super::super::problem::data(problem.value(), policy).unwrap(),
        policy,
        &mut c,
    )
    .await
    .unwrap();
    let super::super::Prepared::Algebraic(program) =
        objective::compile(&formulation, None, policy, &mut c)
            .await
            .unwrap()
            .unwrap()
    else {
        panic!("affine integer source");
    };
    let assembled = assembly::assemble_domains(
        &program,
        &formulation,
        assembly::ObjectiveKind::Affine,
        assembly::Domains::Integer,
        policy,
        &mut c,
    )
    .await
    .unwrap()
    .unwrap();
    let mut authority =
        proof::Authority::new(&program, &formulation, &assembled.rows, policy, &mut c)
            .await
            .unwrap();
    let roles = [RowRole::Inequality(0), RowRole::Upper(0)];
    let weights = [Exact::from_usize(1), Exact::from_usize(1)];
    assert!(authority
        .integer_farkas(
            &roles,
            &weights,
            proof::BoundValues {
                lower: &[0.],
                upper: &[0.]
            },
            &mut c
        )
        .await
        .unwrap());
    assert!(!authority
        .integer_farkas(
            &roles,
            &weights,
            proof::BoundValues {
                lower: &[0.],
                upper: &[1.]
            },
            &mut c
        )
        .await
        .unwrap());
    assert!(!authority
        .integer_farkas(
            &roles,
            &[-Exact::from_usize(1), Exact::from_usize(1)],
            proof::BoundValues {
                lower: &[0.],
                upper: &[0.]
            },
            &mut c
        )
        .await
        .unwrap());
}
async fn fractional_unbounded_relaxation(runtime: &RuntimeContext, options: &RootedValue) {
    let x = call(
        runtime,
        "optimvar",
        &[
            Value::from("fractional"),
            Value::from("Type"),
            Value::from("integer"),
            Value::from("LowerBound"),
            Value::Num(0.),
            Value::from("UpperBound"),
            Value::Num(1.),
        ],
    )
    .await;
    let y = call(runtime, "optimvar", &[Value::from("ray")]).await;
    let cost = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.times",
        &y,
        Value::Num(-1.),
    )
    .await;
    let constraint = operation(
        runtime,
        "optim.problemdef.OptimizationExpression.eq",
        &x,
        Value::Num(0.5),
    )
    .await;
    let problem = model(runtime, &cost, &constraint).await;
    let result = solve(runtime, &problem, None, options).await.unwrap();
    let values = outputs(&result);
    assert_eq!(
        call(runtime, "double", &[values[2].clone()]).await.value(),
        &Value::Num(-2.)
    );
    assert!(matches!(&values[1],Value::Tensor(value)if value.is_empty()));
}
