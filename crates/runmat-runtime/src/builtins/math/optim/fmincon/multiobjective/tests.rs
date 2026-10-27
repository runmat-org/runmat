use super::*;
#[path = "tests/negative_goals.rs"]
mod negative_goals;
use futures::executor::block_on;
use runmat_value::{Tensor, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
fn array(values: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(values, shape).unwrap())
}
fn numbers(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(v) => vec![*v],
        Value::Tensor(v) => v.materialize_f64(),
        _ => panic!("numeric"),
    }
}
fn structure(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut value = runmat_value::UnboundedStructValueBuilder::new();
    for (k, v) in fields {
        value.insert(k, v);
    }
    Value::Struct(value.finish())
}
fn tuple(values: Vec<Value>) -> BuiltinResult<crate::execution::RootedValueSequence> {
    crate::execution::RootedValueSequence::new(
        runmat_value::ValueSequence::comma_separated(values).unwrap(),
    )
    .map_err(|e| e.into_runtime_error())
}
fn definition(kind: Kind) -> Definition {
    match kind {
        Kind::Minimax => Definition {
            kind,
            descriptor: &runmat_builtins::FMINIMAX_DESCRIPTOR,
            error: &runmat_builtins::FMINIMAX_ERROR_INVALID_ARGUMENT,
            numeric: &runmat_builtins::FMINIMAX_NUMERIC_EXTENSION,
            single: &runmat_builtins::FMINIMAX_SINGLE_EXTENSION,
            resident: &runmat_builtins::FMINIMAX_RESIDENT_EXTENSION,
            integer_option: &runmat_builtins::FMINIMAX_INTEGER_OPTION_EXTENSION,
            sparse: &runmat_builtins::FMINIMAX_SPARSE_EXTENSION,
        },
        Kind::GoalAttain => Definition {
            kind,
            descriptor: &runmat_builtins::FGOALATTAIN_DESCRIPTOR,
            error: &runmat_builtins::FGOALATTAIN_ERROR_INVALID_ARGUMENT,
            numeric: &runmat_builtins::FGOALATTAIN_NUMERIC_EXTENSION,
            single: &runmat_builtins::FGOALATTAIN_SINGLE_EXTENSION,
            resident: &runmat_builtins::FGOALATTAIN_RESIDENT_EXTENSION,
            integer_option: &runmat_builtins::FGOALATTAIN_INTEGER_OPTION_EXTENSION,
            sparse: &runmat_builtins::FGOALATTAIN_SPARSE_EXTENSION,
        },
    }
}
fn options(analytic: bool, extra: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut fields = vec![
        ("Display", Value::from("off")),
        ("SpecifyObjectiveGradient", Value::Bool(analytic)),
        ("SpecifyConstraintGradient", Value::Bool(analytic)),
        ("OptimalityTolerance", Value::Num(1e-7)),
        ("ConstraintTolerance", Value::Num(1e-7)),
        ("StepTolerance", Value::Num(1e-10)),
        ("FunctionTolerance", Value::Num(1e-10)),
    ];
    fields.extend(extra);
    structure(fields)
}

#[test]
fn vector_shapes_analytic_and_difference_rows_keep_original_counts_and_lambda() {
    for kind in [Kind::Minimax, Kind::GoalAttain] {
        for (merit, analytic) in [("multiobj", false), ("multiobj", true), ("singleobj", true)] {
            let runtime = crate::user_functions::test_support::exact_fixture_runtime();
            runtime.set_runmat_extensions_enabled(analytic);
            let _entry = runtime.enter();
            let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
                &runtime,
                &[("multi_objective", 2241), ("multi_constraint", 2242)],
            )
            .unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let constraints = Arc::new(AtomicUsize::new(0));
            let oc = calls.clone();
            let cc = constraints.clone();
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                Arc::new(move |id, args, requested| {
                    assert_eq!(
                        numbers(&args[0]).len(),
                        1,
                        "original callback must never receive gamma"
                    );
                    let x = numbers(&args[0])[0];
                    let out = if id == 2241 {
                        oc.fetch_add(1, Ordering::SeqCst);
                        assert_eq!(requested, if analytic { 2 } else { 1 });
                        let mut out = vec![array(
                            vec![(x - 1.0).powi(2), (x + 1.0).powi(2)],
                            vec![1, 2],
                        )];
                        if analytic {
                            out.push(Value::SparseTensor(
                                runmat_value::SparseTensor::new(
                                    1,
                                    2,
                                    vec![0, 1, 2],
                                    vec![0, 0],
                                    vec![2.0 * (x - 1.0), 2.0 * (x + 1.0)],
                                )
                                .unwrap(),
                            ));
                        }
                        out
                    } else {
                        assert_eq!(id, 2242);
                        cc.fetch_add(1, Ordering::SeqCst);
                        assert_eq!(requested, if analytic { 4 } else { 2 });
                        let mut out = vec![Value::Num(x - 0.25), inputs::empty()];
                        if analytic {
                            out.push(Value::Num(1.0));
                            out.push(array(Vec::new(), vec![1, 0]));
                        }
                        out
                    };
                    Box::pin(async move { tuple(out) })
                }),
            ))
            .unwrap();
            let _count = crate::output_count::push_output_count(Some(6)).unwrap();
            let mut args = vec![
                Value::FunctionHandle("multi_objective".into()),
                Value::Num(2.0),
            ];
            if kind == Kind::GoalAttain {
                args.extend([
                    array(vec![0.0, 0.0], vec![2, 1]),
                    array(vec![1.0, 1.0], vec![1, 2]),
                ]);
            }
            args.extend([
                inputs::empty(),
                inputs::empty(),
                inputs::empty(),
                inputs::empty(),
                inputs::empty(),
                inputs::empty(),
                Value::FunctionHandle("multi_constraint".into()),
                options(analytic, [("MeritFunction", Value::from(merit))]),
            ]);
            let out = block_on(execute(
                Some(&runtime),
                crate::builtins::common::test_support::builtin_arguments(args),
                definition(kind),
            ))
            .unwrap();
            let out = out.as_slice();
            assert!(numbers(&out[3])[0] > 0.0, "{:?}", out[4]);
            assert!(numbers(&out[0])[0].abs() < 1e-4);
            assert_eq!(
                crate::builtins::math::optim::fmincon::numeric::shape(&out[1]),
                Some(vec![1, 2])
            );
            for value in numbers(&out[1]) {
                assert!((value - 1.0).abs() < 1e-4);
            }
            assert!((numbers(&out[2])[0] - 1.0).abs() < 1e-4);
            let Value::Struct(info) = &out[4] else {
                panic!("information")
            };
            assert_eq!(
                numbers(&info.fields["funcCount"])[0] as usize,
                calls.load(Ordering::SeqCst)
            );
            assert_eq!(
                calls.load(Ordering::SeqCst),
                constraints.load(Ordering::SeqCst)
            );
            assert_eq!(
                numbers(&info.fields["objectiveCallCount"])[0] as usize,
                calls.load(Ordering::SeqCst)
            );
            let Value::Struct(lambda) = &out[5] else {
                panic!("original multipliers")
            };
            assert_eq!(
                numbers(&lambda.fields["ineqnonlin"]).len(),
                1,
                "generated objective rows must not leak into user lambda"
            );
            assert_eq!(numbers(&lambda.fields["lower"]).len(), 1);
        }
    }
}

#[test]
fn absolute_equality_and_zero_weight_goals_have_genuine_distinct_models() {
    for case in 0..3 {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("multi_roles", 2243)],
        )
        .unwrap();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_, args, requested| {
                assert_eq!(requested, 1);
                let x = numbers(&args[0])[0];
                let values = match case {
                    0 => vec![x, -5.0],
                    1 => vec![x, x + 2.0],
                    _ => vec![(x - 2.0).powi(2), x],
                };
                Box::pin(async move { tuple(vec![array(values, vec![2, 1])]) })
            },
        )))
        .unwrap();
        let _count = crate::output_count::push_output_count(Some(6)).unwrap();
        let kind = if case == 0 {
            Kind::Minimax
        } else {
            Kind::GoalAttain
        };
        let args = if case == 1 {
            vec![structure([
                ("objective", Value::FunctionHandle("multi_roles".into())),
                ("x0", Value::Num(0.0)),
                ("goal", array(vec![1.0, 0.0], vec![2, 1])),
                ("weight", array(vec![1.0, 1.0], vec![2, 1])),
                ("solver", Value::from("fgoalattain")),
                (
                    "options",
                    options(false, [("EqualityGoalCount", Value::Num(1.0))]),
                ),
            ])]
        } else {
            let mut args = vec![
                Value::FunctionHandle("multi_roles".into()),
                Value::Num(if case == 0 { -1.5 } else { 0.0 }),
            ];
            if case == 2 {
                args.extend([
                    array(vec![0.0, 1.0], vec![2, 1]),
                    array(vec![1.0, 0.0], vec![2, 1]),
                ]);
            }
            args.extend([
                inputs::empty(),
                inputs::empty(),
                inputs::empty(),
                inputs::empty(),
                Value::Num(if case == 0 { -2.0 } else { 0.0 }),
                Value::Num(if case == 0 { -1.0 } else { 3.0 }),
                inputs::empty(),
                options(
                    false,
                    if case == 0 {
                        vec![("AbsoluteMaxObjectiveCount", Value::Num(1.0))]
                    } else {
                        Vec::new()
                    },
                ),
            ]);
            args
        };
        let out = block_on(execute(
            Some(&runtime),
            crate::builtins::common::test_support::builtin_arguments(args),
            definition(kind),
        ))
        .unwrap();
        let out = out.as_slice();
        assert!(numbers(&out[3])[0] > 0.0, "{:?}", out[4]);
        let (x, attainment) = match case {
            0 => (-1.0, -1.0),
            1 => (-0.5, 1.5),
            _ => (1.0, 1.0),
        };
        assert!((numbers(&out[0])[0] - x).abs() < 1e-4);
        assert!((numbers(&out[2])[0] - attainment).abs() < 1e-4);
    }
}

#[test]
fn hard_limits_original_errors_and_gamma_only_cache_do_not_replay_callbacks() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
        RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Finite(ManagedHostEffectLedger);
    impl RuntimeParallelService for Finite {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let state = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(state.free_bytes()),
                epoch: state.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(1024 * 1024);
    let finite = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger.clone()))),
        );
    for kind in [Kind::Minimax, Kind::GoalAttain] {
        let mut args = vec![
            Value::FunctionHandle("never_invoked_under_finite_policy".into()),
            Value::Num(0.0),
        ];
        if kind == Kind::GoalAttain {
            args.extend([Value::Num(0.0), Value::Num(1.0)]);
        }
        let error = block_on(execute(
            Some(&finite),
            crate::builtins::common::test_support::builtin_arguments(args),
            definition(kind),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
    assert_eq!(ledger.snapshot().peak_reserved_bytes, 0);
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    runtime.set_error_namespace("MultiOriginal");
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("multi_failure", 2244)],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(move |_, _, _| {
            assert_eq!(
                crate::context::legacy::active().unwrap().error_namespace(),
                "MultiOriginal"
            );
            counted.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Err(
                    crate::build_runtime_error("original multiobjective failure")
                        .with_identifier("MultiFixture:OriginalFailure")
                        .build(),
                )
            })
        })))
        .unwrap();
    let _count = crate::output_count::push_output_count(Some(6)).unwrap();
    let arguments = |opts| {
        vec![
            Value::FunctionHandle("multi_failure".into()),
            Value::Num(0.0),
            inputs::empty(),
            inputs::empty(),
            inputs::empty(),
            inputs::empty(),
            inputs::empty(),
            inputs::empty(),
            inputs::empty(),
            opts,
        ]
    };
    let out = block_on(execute(
        Some(&runtime),
        crate::builtins::common::test_support::builtin_arguments(arguments(options(
            false,
            [("MaxFunEvals", Value::Num(0.0))],
        ))),
        definition(Kind::Minimax),
    ))
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(numbers(&out.as_slice()[1]).is_empty());
    assert_eq!(numbers(&out.as_slice()[3]), vec![0.0]);
    let foreign = crate::user_functions::test_support::exact_fixture_runtime();
    foreign.set_error_namespace("MultiForeign");
    let _foreign = foreign.enter();
    let error = block_on(execute(
        Some(&runtime),
        crate::builtins::common::test_support::builtin_arguments(arguments(options(false, []))),
        definition(Kind::Minimax),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("MultiFixture:OriginalFailure"));
    assert_eq!(
        crate::context::legacy::active().unwrap().error_namespace(),
        "MultiForeign"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let mut inconsistent = arguments(options(false, []));
    inconsistent[6] = Value::Num(1.0);
    inconsistent[7] = Value::Num(0.0);
    let stopped = block_on(execute(
        Some(&runtime),
        crate::builtins::common::test_support::builtin_arguments(inconsistent),
        definition(Kind::Minimax),
    ))
    .unwrap();
    assert_eq!(numbers(&stopped.as_slice()[0]), vec![0.0]);
    assert!(numbers(&stopped.as_slice()[1]).is_empty());
    assert_eq!(numbers(&stopped.as_slice()[3]), vec![-2.0]);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "inconsistent bounds must not evaluate the objective"
    );
    let invalid_weight = block_on(execute(
        Some(&runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::FunctionHandle("multi_failure".into()),
            Value::Num(0.0),
            Value::Num(0.0),
            Value::Num(f64::NAN),
        ]),
        definition(Kind::GoalAttain),
    ))
    .unwrap_err();
    assert!(invalid_weight.message().contains("weight"));
    assert_eq!(
        invalid_weight.identifier(),
        Some("RunMat:fgoalattain:InvalidArgument")
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "nonfinite weights must reject before callbacks"
    );
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "fminimax" },
        Some(&runtime),
    );
    let formulation = block_on(
        crate::builtins::math::optim::common::multiobjective::formulation::Specification::Minimax {
            absolute: 0,
        }
        .bind(Kind::Minimax, 2, &mut cooperation),
    )
    .unwrap();
    let original_objective = Value::FunctionHandle("multi_failure".into());
    let mut evaluator = evaluation::Evaluator {
        internal: None,
        semi_infinite: None,
        scaling: Default::default(),
        attainment: Some(evaluation::Attainment {
            binding: evaluation::Binding::Bound(formulation),
            merit: runmat_builtins::OptimMeritFunction::Multiobjective,
        }),
        cooperation: &mut cooperation,
        objective: evaluation::Objective::User(&original_objective),
        feasibility_limit: None,
        selected_constraints: None,
        constraints: None,
        policy: super::super::numeric::Policy {
            owner: runmat_builtins::BuiltinCatalogIdentity { name: "fminimax" },
            invalid: None,
            numeric: Some(&runmat_builtins::FMINIMAX_NUMERIC_EXTENSION),
            single: Some(&runmat_builtins::FMINIMAX_SINGLE_EXTENSION),
            resident: Some(&runmat_builtins::FMINIMAX_RESIDENT_EXTENSION),
            sparse: Some(&runmat_builtins::FMINIMAX_SPARSE_EXTENSION),
        },
        shape: vec![1, 1],
        scalar: true,
        objective_gradient: false,
        constraint_gradient: false,
        budget: evaluation::Budget::new(1),
        constraint_extents: None,
    };
    let mut point = evaluation::Point {
        semi_infinite: None,
        objectives: Some(evaluation::ObjectiveValues {
            values: vec![1.0, 1.0],
            shape: vec![1, 2],
        }),
        f: 0.0,
        inequality: vec![1.0, 1.0],
        equality: Vec::new(),
        gradient: Some(vec![0.0, 1.0]),
        inequality_jacobian: Some(clarabel::algebra::CscMatrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![-2.0, 2.0, -1.0, -1.0],
        )),
        equality_jacobian: Some(clarabel::algebra::CscMatrix::zeros((0, 2))),
        projected_objective: false,
        projected_constraints: false,
    };
    let geometry = super::super::model::Geometry {
        variables: 2,
        inequality: super::super::model::LinearConstraints {
            matrix: clarabel::algebra::CscMatrix::zeros((0, 2)),
            rhs: Vec::new(),
        },
        equality: super::super::model::LinearConstraints {
            matrix: clarabel::algebra::CscMatrix::zeros((0, 2)),
            rhs: Vec::new(),
        },
        lower: vec![f64::NEG_INFINITY; 2],
        upper: vec![f64::INFINITY; 2],
    };
    assert!(block_on(evaluator.reuses_generation(&[0.0, 0.0], &[0.0, 1.0])).unwrap());
    assert!(!block_on(evaluator.reuses_generation(&[0.0, 0.0], &[0.1, 1.0])).unwrap());
    let owner = evaluator.policy.owner;
    let mut values =
        block_on(geometry.constraint_values(&[0.0, 1.0], &point, owner, evaluator.cooperation))
            .unwrap()
            .unwrap();
    assert!(block_on(evaluator.stage_gamma(&point, 1.0, &mut values)).unwrap());
    assert_eq!(values.rhs, vec![0.0, 0.0]);
    let model = block_on(geometry.linearize_with_values(
        &[0.0, 1.0],
        &point,
        values,
        owner,
        evaluator.cooperation,
    ))
    .unwrap();
    block_on(evaluator.accept_retained(&[0.0, 0.0], &[0.0, 1.0], &mut point, &model)).unwrap();
    assert_eq!(point.f, 1.0);
    assert_eq!(point.inequality, vec![0.0, 0.0]);
    assert_eq!(point.objectives.unwrap().values, vec![1.0, 1.0]);
    assert_eq!(evaluator.budget.counts().points, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
