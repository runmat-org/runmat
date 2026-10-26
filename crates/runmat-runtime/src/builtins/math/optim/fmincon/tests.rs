use super::*;
#[path = "reflective_tests.rs"]
mod reflective_tests;
use futures::executor::block_on;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
fn array(values: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(values, shape).unwrap())
}
fn numbers(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(value) => vec![*value],
        Value::Tensor(value) => value.materialize_f64(),
        _ => panic!("numeric"),
    }
}
fn args(values: Vec<Value>) -> crate::call::BuiltinArguments {
    crate::builtins::common::test_support::builtin_arguments(values)
}
fn structure(values: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in values {
        fields.insert(name, value);
    }
    Value::Struct(fields.finish())
}
fn result(values: Vec<Value>) -> BuiltinResult<crate::execution::RootedValueSequence> {
    crate::execution::RootedValueSequence::new(
        runmat_value::ValueSequence::comma_separated(values)
            .map_err(crate::sequence::sequence_error_to_runtime)?,
    )
    .map_err(|source| source.into_runtime_error())
}
fn runtime() -> crate::context::RuntimeContext {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let clock = std::rc::Rc::new(
        crate::context::DeterministicTimingService::new(
            runmat_execution::host::NativeClockInputs {
                schema_version: runmat_execution::host::NATIVE_CLOCK_INPUTS_SCHEMA_VERSION,
                initial_monotonic_nanos: 0,
                initial_wall_unix_seconds: 0,
                initial_wall_nanosecond: 0,
                read_step_nanos: 1_000_000,
                local_offset_seconds: 0,
                process_cpu: None,
            },
            runtime.cancellation(),
        )
        .unwrap(),
    );
    let ports = runtime.service_ports().clone().with_timing(clock);
    runtime.with_service_ports(ports)
}
fn options(analytic: bool, extra: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut fields = vec![
        ("Algorithm", Value::from("sqp")),
        ("Display", Value::from("off")),
        ("SpecifyObjectiveGradient", Value::Bool(analytic)),
        ("SpecifyConstraintGradient", Value::Bool(analytic)),
        ("OptimalityTolerance", Value::Num(1e-7)),
        ("ConstraintTolerance", Value::Num(1e-7)),
        ("StepTolerance", Value::Num(1e-9)),
    ];
    fields.extend(extra);
    structure(fields)
}

#[test]
fn sparse_linear_and_nonlinear_sqp_preserve_real_kkt_roles_and_callback_accounting() {
    for analytic in [false, true] {
        let runtime = runtime();
        runtime.set_runmat_extensions_enabled(true);
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("sqp_objective", 2101), ("sqp_constraints", 2102)],
        )
        .unwrap();
        let objective = Arc::new(AtomicUsize::new(0));
        let constraints = Arc::new(AtomicUsize::new(0));
        let oc = objective.clone();
        let cc = constraints.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, requested| {
                let x = numbers(&args[0]);
                let outputs = if id == 2101 {
                    oc.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(requested, if analytic { 2 } else { 1 });
                    let mut out = vec![Value::Num((x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2))];
                    if analytic {
                        out.push(array(
                            vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)],
                            vec![2, 1],
                        ));
                    }
                    out
                } else {
                    assert_eq!(id, 2102);
                    cc.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(requested, if analytic { 4 } else { 2 });
                    let mut out = vec![Value::Num(x[0] * x[0] + x[1] * x[1] - 4.0), empty()];
                    if analytic {
                        out.push(Value::SparseTensor(
                            runmat_value::SparseTensor::new(
                                2,
                                1,
                                vec![0, 2],
                                vec![0, 1],
                                vec![2.0 * x[0], 2.0 * x[1]],
                            )
                            .unwrap(),
                        ));
                        out.push(array(Vec::new(), vec![2, 0]));
                    }
                    out
                };
                Box::pin(async move { result(outputs) })
            },
        )))
        .unwrap();
        let _count = crate::output_count::push_output_count(Some(7)).unwrap();
        let out = block_on(execute(
            Some(&runtime),
            args(vec![
                Value::FunctionHandle("sqp_objective".into()),
                array(vec![0.0, 0.0], vec![1, 2]),
                Value::SparseTensor(
                    runmat_value::SparseTensor::new(
                        1,
                        2,
                        vec![0, 1, 2],
                        vec![0, 0],
                        vec![1.0, 1.0],
                    )
                    .unwrap(),
                ),
                Value::Num(1.0),
                empty(),
                empty(),
                empty(),
                empty(),
                Value::FunctionHandle("sqp_constraints".into()),
                options(
                    analytic,
                    if analytic {
                        Vec::new()
                    } else {
                        vec![
                            (
                                "SpecifyConstraintGradient",
                                Value::from("dormant invalid value"),
                            ),
                            ("FiniteDifferenceType", Value::from("central")),
                        ]
                    },
                ),
            ]),
        ))
        .unwrap();
        let output = out.as_slice();
        let x = numbers(&output[0]);
        assert!((x[0] - 1.0).abs() < 1e-4 && x[1].abs() < 1e-4, "{x:?}");
        assert!(matches!(&output[0],Value::Tensor(value)if value.shape()==[1,2]));
        assert!((numbers(&output[1])[0] - 2.0).abs() < 1e-5);
        assert!(numbers(&output[2])[0] > 0.0);
        let Value::Struct(info) = &output[3] else {
            panic!("information")
        };
        assert_eq!(
            numbers(&info.fields["funcCount"])[0],
            objective.load(Ordering::SeqCst) as f64
        );
        assert_eq!(
            objective.load(Ordering::SeqCst),
            constraints.load(Ordering::SeqCst)
        );
        assert_eq!(
            numbers(&info.fields["constraintCallCount"])[0],
            constraints.load(Ordering::SeqCst) as f64
        );
        let Value::Struct(lambda) = &output[4] else {
            panic!("actual multipliers")
        };
        assert!((numbers(&lambda.fields["ineqlin"])[0] - 2.0).abs() < 1e-4);
        assert!(numbers(&lambda.fields["ineqnonlin"])[0].abs() < 1e-5);
        assert!((numbers(&output[5])[0] + 2.0).abs() < 1e-4);
        assert!(matches!(&output[6],Value::Tensor(value)if value.shape()==[2,2]));
        if !analytic {
            objective.store(0, Ordering::SeqCst);
            constraints.store(0, Ordering::SeqCst);
            runtime.set_runmat_extensions_enabled(false);
            let default_inputs = |settings| {
                args(vec![
                    Value::FunctionHandle("sqp_objective".into()),
                    array(vec![0.2, 0.5], vec![1, 2]),
                    empty(),
                    empty(),
                    Value::SparseTensor(
                        runmat_value::SparseTensor::new(
                            1,
                            2,
                            vec![0, 1, 2],
                            vec![0, 0],
                            vec![1.0, 1.0],
                        )
                        .unwrap(),
                    ),
                    Value::Num(1.0),
                    array(vec![0.0, 0.0], vec![2, 1]),
                    array(vec![0.75, 4.0], vec![2, 1]),
                    empty(),
                    settings,
                ])
            };
            let foreign = crate::user_functions::test_support::exact_fixture_runtime();
            foreign.set_runmat_extensions_enabled(true);
            let _foreign = foreign.enter();
            let interior = block_on(execute(
                Some(&runtime),
                default_inputs(structure([
                    ("Display", Value::from("off")),
                    ("FiniteDifferenceType", Value::from("central")),
                    ("SpecifyObjectiveGradient", Value::Bool(false)),
                    (
                        "SpecifyConstraintGradient",
                        Value::from("unused without nonlcon"),
                    ),
                ])),
            ))
            .unwrap();
            let actual = numbers(&interior.as_slice()[0]);
            assert!(
                (actual[0] - 0.75).abs() < 1e-4 && (actual[1] - 0.25).abs() < 1e-4,
                "{actual:?}"
            );
            assert!(numbers(&interior.as_slice()[2])[0] > 0.0);
            let Value::Struct(information) = &interior.as_slice()[3] else {
                panic!("information")
            };
            assert_eq!(
                information.fields["algorithm"],
                Value::from("interior-point")
            );
            assert_eq!(
                numbers(&information.fields["funcCount"])[0],
                objective.load(Ordering::SeqCst) as f64
            );
            assert_eq!(constraints.load(Ordering::SeqCst), 0);
            let Value::Struct(lambda) = &interior.as_slice()[4] else {
                panic!("lambda")
            };
            assert!((numbers(&lambda.fields["eqlin"])[0] - 1.5).abs() < 1e-4);
            assert!((numbers(&lambda.fields["upper"])[0] - 1.0).abs() < 1e-4);
            let calls = objective.load(Ordering::SeqCst);
            let refused =
                block_on(execute(Some(&runtime), default_inputs(options(false, [])))).unwrap_err();
            assert_eq!(
                refused.identifier(),
                runmat_builtins::FMINCON_SPARSE_EXTENSION.error_identifier
            );
            assert_eq!(
                objective.load(Ordering::SeqCst),
                calls,
                "strict SQP sparse refusal precedes callbacks despite foreign enabled policy"
            );
            drop(_foreign);
            runtime.set_runmat_extensions_enabled(true);
            let fixed = block_on(execute(
                Some(&runtime),
                args(vec![
                    Value::FunctionHandle("sqp_objective".into()),
                    array(vec![1.0, 0.0], vec![2, 1]),
                    empty(),
                    empty(),
                    empty(),
                    empty(),
                    array(vec![1.0, f64::NEG_INFINITY], vec![2, 1]),
                    array(vec![1.0, f64::INFINITY], vec![2, 1]),
                    empty(),
                    options(false, []),
                ]),
            ))
            .unwrap();
            assert_eq!(
                numbers(&fixed.as_slice()[2]),
                vec![2.0],
                "feasible projected stop is not full original-gradient KKT success"
            );
            assert!(
                matches!(&fixed.as_slice()[4],Value::Tensor(value)if value.is_empty()),
                "do not invent full-gradient bound multipliers"
            );
            let gradient = numbers(&fixed.as_slice()[5]);
            assert!(gradient[0].is_nan());
            assert!(gradient[1].abs() < 1e-5);
            let Value::Struct(info) = &fixed.as_slice()[3] else {
                panic!("information")
            };
            assert!(
                matches!(&info.fields["firstorderopt"],Value::Tensor(value)if value.is_empty())
            );
        }
    }
}

#[test]
fn nonlinear_disk_problem_form_and_default_barrier_have_original_duals_and_distinct_hessians() {
    let runtime = runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("sqp_disk_objective", 2111), ("sqp_disk_constraint", 2112)],
    )
    .unwrap();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, requested| {
            let x = numbers(&args[0]);
            let out = if id == 2111 {
                assert_eq!(requested, 2);
                vec![
                    Value::Num((x[0] - 2.0).powi(2) + x[1] * x[1]),
                    array(vec![2.0 * (x[0] - 2.0), 2.0 * x[1]], vec![2, 1]),
                ]
            } else {
                assert_eq!(id, 2112);
                assert_eq!(requested, 4);
                vec![
                    Value::Num(x[0] * x[0] + x[1] * x[1] - 1.0),
                    empty(),
                    array(vec![2.0 * x[0], 2.0 * x[1]], vec![2, 1]),
                    array(Vec::new(), vec![2, 0]),
                ]
            };
            Box::pin(async move { result(out) })
        },
    )))
    .unwrap();
    let _count = crate::output_count::push_output_count(Some(7)).unwrap();
    let problem = structure([
        (
            "objective",
            Value::FunctionHandle("sqp_disk_objective".into()),
        ),
        ("x0", array(vec![0.5, 0.0], vec![2, 1])),
        ("solver", Value::from("fmincon")),
        ("options", options(true, [])),
        (
            "nonlcon",
            Value::FunctionHandle("sqp_disk_constraint".into()),
        ),
    ]);
    let out = block_on(execute(Some(&runtime), args(vec![problem]))).unwrap();
    let x = numbers(&out.as_slice()[0]);
    assert!((x[0] - 1.0).abs() < 1e-4 && x[1].abs() < 1e-4);
    let Value::Struct(lambda) = &out.as_slice()[4] else {
        panic!("lambda")
    };
    assert!((numbers(&lambda.fields["ineqnonlin"])[0] - 1.0).abs() < 1e-4);
    let active = block_on(execute(
        Some(&runtime),
        args(vec![
            Value::FunctionHandle("sqp_disk_objective".into()),
            array(vec![0.5, 0.0], vec![2, 1]),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::FunctionHandle("sqp_disk_constraint".into()),
            options(true, [("Algorithm", Value::from("active-set"))]),
        ]),
    ))
    .unwrap();
    let x = numbers(&active.as_slice()[0]);
    assert!((x[0] - 1.0).abs() < 1e-4 && x[1].abs() < 1e-4);
    assert!(numbers(&active.as_slice()[2])[0] > 0.0);
    let Value::Struct(info) = &active.as_slice()[3] else {
        panic!("active information")
    };
    assert_eq!(info.fields["algorithm"], Value::from("active-set"));
    let Value::Struct(lambda) = &active.as_slice()[4] else {
        panic!("active lambda")
    };
    assert!((numbers(&lambda.fields["ineqnonlin"])[0] - 1.0).abs() < 1e-4);
    let interior = block_on(execute(
        Some(&runtime),
        args(vec![
            Value::FunctionHandle("sqp_disk_objective".into()),
            array(vec![0.5, 0.0], vec![2, 1]),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::FunctionHandle("sqp_disk_constraint".into()),
            structure([
                ("Display", Value::from("off")),
                ("SpecifyObjectiveGradient", Value::Bool(true)),
                ("SpecifyConstraintGradient", Value::Bool(true)),
            ]),
        ]),
    ))
    .unwrap();
    let x = numbers(&interior.as_slice()[0]);
    assert!((x[0] - 1.0).abs() < 1e-4 && x[1].abs() < 1e-4);
    assert!(numbers(&interior.as_slice()[2])[0] > 0.0);
    let Value::Struct(metadata) = &interior.as_slice()[3] else {
        panic!("metadata")
    };
    assert_eq!(metadata.fields["algorithm"], Value::from("interior-point"));
    let Value::Struct(lambda) = &interior.as_slice()[4] else {
        panic!("lambda")
    };
    assert!((numbers(&lambda.fields["ineqnonlin"])[0] - 1.0).abs() < 1e-4);
    for algorithm in ["sqp", "active-set", "interior-point"] {
        let exact = block_on(execute(
            Some(&runtime),
            args(vec![
                Value::FunctionHandle("sqp_disk_objective".into()),
                array(vec![0.0, 0.0], vec![2, 1]),
                empty(),
                empty(),
                empty(),
                empty(),
                empty(),
                empty(),
                empty(),
                structure([
                    ("Algorithm", Value::from(algorithm)),
                    ("Display", Value::from("off")),
                    ("SpecifyObjectiveGradient", Value::Bool(true)),
                    (
                        "SpecifyConstraintGradient",
                        Value::from("inactive without nonlinear input"),
                    ),
                    ("MaxIter", Value::Num(1.0)),
                ]),
            ]),
        ))
        .unwrap();
        assert!((numbers(&exact.as_slice()[0])[0] - 2.0).abs() < 1e-8);
        let expected = if algorithm == "interior-point" {
            2.0
        } else {
            1.0
        };
        assert!(
            (numbers(&exact.as_slice()[6])[0] - expected).abs() < 1e-10,
            "SQP last-used model and IP final-point model must not share publication timing"
        );
    }
    let error = block_on(execute(
        Some(&runtime),
        args(vec![
            Value::FunctionHandle("sqp_disk_objective".into()),
            Value::Num(0.0),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            structure([("Algorithm", Value::from("sqp-legacy"))]),
        ]),
    ))
    .unwrap_err();
    assert!(error.message().contains("unsupported fmincon algorithm"));
}

#[test]
fn zero_point_limit_has_no_callbacks_no_fabricated_derivatives_and_preserves_original_failure() {
    let runtime = runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("sqp_limit", 2121)],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(move |_, _, _| {
            counted.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Err(crate::build_runtime_error("original objective failure")
                    .with_identifier("SqpFixture:OriginalFailure")
                    .build())
            })
        })))
        .unwrap();
    let _count = crate::output_count::push_output_count(Some(7)).unwrap();
    let input = |options| {
        args(vec![
            Value::FunctionHandle("sqp_limit".into()),
            Value::Num(0.0),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            options,
        ])
    };
    let out = block_on(execute(
        Some(&runtime),
        input(options(false, [("MaxFunEvals", Value::Num(0.0))])),
    ))
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(numbers(&out.as_slice()[2]), vec![0.0]);
    for index in [1, 4, 5, 6] {
        assert!(matches!(&out.as_slice()[index],Value::Tensor(value)if value.is_empty()));
    }
    let error = block_on(execute(Some(&runtime), input(options(false, [])))).unwrap_err();
    assert_eq!(error.identifier(), Some("SqpFixture:OriginalFailure"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let active_zero = block_on(execute(
        Some(&runtime),
        input(options(
            false,
            [
                ("Algorithm", Value::from("active-set")),
                ("MaxFunEvals", Value::Num(0.0)),
            ],
        )),
    ))
    .unwrap();
    assert_eq!(numbers(&active_zero.as_slice()[2]), vec![0.0]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let error = block_on(execute(
        Some(&runtime),
        input(options(false, [("Algorithm", Value::from("active-set"))])),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("SqpFixture:OriginalFailure"));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "active-set must not retry callback failures"
    );
    let error = block_on(execute(
        Some(&runtime),
        input(structure([("Display", Value::from("off"))])),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("SqpFixture:OriginalFailure"));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "default barrier must propagate the originating callback failure, not shrink/replay it"
    );
    // Pure original-model arithmetic overflow is numerical undefinedness,
    // unlike the originating callback failures and identifiers above.
    let mut cooperation =
        ScalarCooperation::new(runmat_builtins::OptimSolver::Fmincon, Some(&runtime));
    let point = evaluation::Point {
        semi_infinite: None,
        objectives: None,
        f: 0.0,
        inequality: Vec::new(),
        equality: Vec::new(),
        gradient: None,
        inequality_jacobian: None,
        equality_jacobian: None,
        projected_objective: false,
        projected_constraints: false,
    };
    let mut geometry = model::Geometry {
        variables: 1,
        inequality: model::LinearConstraints {
            matrix: clarabel::algebra::CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![f64::MAX]),
            rhs: vec![0.0],
        },
        equality: model::LinearConstraints {
            matrix: clarabel::algebra::CscMatrix::zeros((0, 1)),
            rhs: Vec::new(),
        },
        lower: vec![f64::NEG_INFINITY],
        upper: vec![f64::INFINITY],
    };
    let owner = runmat_builtins::BuiltinCatalogIdentity { name: "fmincon" };
    assert!(
        block_on(geometry.constraint_values(&[2.0], &point, owner, &mut cooperation))
            .unwrap()
            .is_none(),
        "finite factors can overflow a trial activity"
    );
    assert!(
        block_on(geometry.constraint_values(&[0.5], &point, owner, &mut cooperation))
            .unwrap()
            .is_some(),
        "a smaller finite trial remains admissible"
    );
    geometry.inequality.rhs[0] = -f64::MAX;
    assert!(
        block_on(geometry.constraint_values(&[1.0], &point, owner, &mut cooperation))
            .unwrap()
            .is_none(),
        "finite activity can overflow the residual subtraction"
    );
    geometry.inequality.matrix.nzval[0] = 0.0;
    geometry.inequality.rhs[0] = 0.0;
    geometry.lower[0] = -f64::MAX;
    assert!(
        block_on(geometry.constraint_values(&[f64::MAX], &point, owner, &mut cooperation))
            .unwrap()
            .is_none(),
        "finite bounds can overflow a trial displacement"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "value-stage checks invoke no callbacks"
    );
    runtime.set_error_namespace("SqpOriginal");
    let foreign = crate::user_functions::test_support::exact_fixture_runtime();
    foreign.set_error_namespace("SqpForeign");
    let _foreign = foreign.enter();
    let error =
        evaluation::select_sequence(result(Vec::new()).unwrap(), 1, Some(&runtime)).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("SqpOriginal:CommaSeparatedListRequiresSingleValue")
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(
        crate::context::legacy::active().unwrap().error_namespace(),
        "SqpForeign"
    );
}
