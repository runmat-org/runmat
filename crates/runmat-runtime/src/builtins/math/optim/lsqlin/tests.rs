use super::*;
use futures::executor::block_on;
mod active_cases;
mod reflective_cases;
mod warm_cases;
#[test]
fn original_cancellation_is_not_replaced_by_foreign_ambient_context() {
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::rc::Rc;
    let source = CancellationSource::new();
    let foreign_source = CancellationSource::new();
    let original = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    original.set_error_namespace("OriginalLSQ");
    let foreign = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        foreign_source.token(),
    );
    foreign_source.cancel(CancellationReason::User);
    {
        let _scope = foreign.enter();
        let mut c = ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqlin, Some(&original));
        assert_eq!(
            block_on(numeric::vector(Value::Num(3.0), false, false, &mut c)).unwrap(),
            vec![3.0]
        );
    }
    source.cancel(CancellationReason::User);
    let c = ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqlin, Some(&original));
    let error = block_on(numeric::gather(Value::Num(3.0), &c)).unwrap_err();
    assert_eq!(error.identifier(), Some("OriginalLSQ:Cancelled"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
}
fn matrix(data: Vec<f64>, m: usize, n: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![m, n]).unwrap())
}
fn values(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(v) => vec![*v],
        Value::Tensor(t) => t.materialize_f64(),
        _ => panic!("numeric output"),
    }
}
fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Struct(value) = value else {
        panic!("struct")
    };
    &value.fields[name]
}
fn options(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut out = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in fields {
        out.insert(name, value);
    }
    Value::Struct(out.finish())
}
fn run(args: Vec<Value>) -> Vec<Value> {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    run_in(&runtime, args)
}
fn run_in(runtime: &crate::context::RuntimeContext, args: Vec<Value>) -> Vec<Value> {
    let _runtime = runtime.enter();
    let _scope = crate::output_count::push_output_count(Some(6)).unwrap();
    let result = block_on(execute(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .unwrap();
    let Value::OutputList(values) = result else {
        panic!("six outputs")
    };
    values
        .try_into_owned()
        .unwrap()
        .try_into_unbounded_vec()
        .unwrap()
}

#[test]
fn residual_bounds_and_original_multiplier_factor() {
    reflective_cases::check();
    active_cases::check();
    warm_cases::check();
    let result = run(vec![
        matrix(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], 3, 2),
        matrix(vec![2.0, 3.0, 5.0], 3, 1),
        empty(),
        empty(),
        empty(),
        empty(),
        matrix(vec![0.0, 0.0], 2, 1),
        matrix(vec![1.0, 10.0], 2, 1),
        empty(),
        options([("Display", Value::from("off"))]),
    ]);
    assert_eq!(values(&result[3]), vec![1.0]);
    let x = values(&result[0]);
    assert!((x[0] - 1.0).abs() < 1e-6 && (x[1] - 3.5).abs() < 1e-6);
    assert!((values(&result[1])[0] - 1.5).abs() < 1e-6);
    let r = values(&result[2]);
    assert!((r[0] + 1.0).abs() < 1e-6 && (r[1] - 0.5).abs() < 1e-6 && (r[2] + 0.5).abs() < 1e-6);
    let upper = values(field(&result[5], "upper"));
    let lower = values(field(&result[5], "lower"));
    assert!((r[0] + r[2] + upper[0] - lower[0]).abs() < 1e-6);
    assert!((upper[0] - 1.5).abs() < 1e-6); // not twice the half-squared objective gradient
    assert!(values(field(&result[5], "eqlin")).is_empty());
}
#[test]
fn equality_problem_sparse_and_rank_deficient_geometries() {
    let result = run(vec![options([
        ("C", matrix(vec![1.0, 0.0, 0.0, 1.0], 2, 2)),
        ("d", matrix(vec![2.0, 3.0], 2, 1)),
        ("Aeq", matrix(vec![1.0, 1.0], 1, 2)),
        ("beq", Value::Num(1.0)),
        ("solver", Value::from("lsqlin")),
        ("options", options([("Display", Value::from("off"))])),
    ])]);
    assert_eq!(values(&result[3]), vec![1.0]);
    let x = values(&result[0]);
    assert!(x[0].abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6);
    assert!((values(&result[1])[0] - 8.0).abs() < 1e-6);
    let eq = values(field(&result[5], "eqlin"));
    assert_eq!(eq.len(), 1);
    assert!((eq[0] - 2.0).abs() < 1e-6);
    for c in [
        matrix(vec![1.0, 2.0, 1.0, 2.0], 2, 2),
        Value::SparseTensor(
            runmat_value::SparseTensor::new(
                2,
                2,
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![1.0, 2.0, 1.0, 2.0],
            )
            .unwrap(),
        ),
    ] {
        let result = run(vec![c, matrix(vec![2.0, 4.0], 2, 1), empty(), empty()]);
        assert_eq!(values(&result[3]), vec![1.0]);
        let x = values(&result[0]);
        assert!((x.iter().sum::<f64>() - 2.0).abs() < 1e-6);
        assert!(values(&result[1])[0] < 1e-12);
    }
    let wide = run(vec![
        matrix(vec![1.0, 1.0, 1.0], 1, 3),
        Value::Num(3.0),
        empty(),
        empty(),
    ]);
    assert_eq!(values(&wide[3]), vec![1.0]);
    assert!((values(&wide[0]).iter().sum::<f64>() - 3.0).abs() < 1e-6);
}
#[test]
fn lifted_model_is_sparse_and_original_kkt_rejects_forged_candidate() {
    let mut coop = ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqlin, None);
    let problem = block_on(parse::problem(
        crate::builtins::common::test_support::builtin_arguments(vec![
            matrix(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
            matrix(vec![2.0, 3.0], 2, 1),
            empty(),
            empty(),
        ]),
        &mut coop,
    ))
    .unwrap();
    let (model, controls) = block_on(model::Model::new(problem, &mut coop)).unwrap();
    assert_eq!(model.a.nzval.len(), 4);
    assert_eq!(model.p.nzval.len(), 2);
    assert_eq!(model.a.n, 4);
    assert_eq!(model.a.m, 2);
    // Kernel success alone cannot certify a point with the wrong original residual.
    assert!(
        !block_on(validation::optimum(
            &model,
            &[0.0, 0.0],
            &[0.0, 0.0],
            &controls,
            &mut coop
        ))
        .unwrap()
        .unwrap()
        .valid
    );
}
#[test]
fn infeasible_stopped_and_options_are_not_silent_aliases() {
    let result = run(vec![
        Value::Num(1.0),
        Value::Num(2.0),
        matrix(vec![1.0, -1.0], 2, 1),
        matrix(vec![0.0, -1.0], 2, 1),
    ]);
    assert_eq!(values(&result[3]), vec![-2.0]);
    assert!(values(&result[0]).is_empty());
    let result = run(vec![
        Value::Num(1.0),
        Value::Num(2.0),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        options([
            ("MaxIter", Value::Num(0.0)),
            ("Display", Value::from("off")),
        ]),
    ]);
    assert_eq!(values(&result[3]), vec![0.0]);
    assert_eq!(values(&result[0]), vec![0.0]);
    assert_eq!(values(&result[2]), vec![-2.0]);
    assert_eq!(values(&result[1]), vec![4.0]);
    assert!(values(&result[5]).is_empty());
    let mut coop = ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqlin, None);
    for (name, value) in [
        ("Algorithm", Value::from("interior-point-convex")),
        ("StepTolerance", Value::Num(1e-6)),
        ("LinearSolver", Value::from("dense")),
        ("ConstraintTolerance", Value::Num(0.0)),
    ] {
        assert!(block_on(options::Controls::parse(
            options([(name, value)]),
            false,
            &mut coop
        ))
        .is_err());
    }
}
#[test]
fn ordinary_double_and_gated_single_integer_data_keep_distinct_policy() {
    let single = Value::Tensor(
        Tensor::new_with_dtype(vec![1.0], vec![1, 1], runmat_value::NumericDType::F32).unwrap(),
    );
    let int = Value::Int(runmat_value::IntValue::I32(1));
    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        assert!(numeric::compatibility(
            &single,
            &runmat_builtins::LSQLIN_INTEGER_C_EXTENSION,
            None,
            numeric::CompatibilityRole::Data
        )
        .is_err());
        assert!(numeric::compatibility(
            &int,
            &runmat_builtins::LSQLIN_INTEGER_C_EXTENSION,
            None,
            numeric::CompatibilityRole::Data
        )
        .is_err());
        assert!(numeric::compatibility(
            &Value::Num(1.0),
            &runmat_builtins::LSQLIN_INTEGER_C_EXTENSION,
            None,
            numeric::CompatibilityRole::Data
        )
        .is_ok());
    }
    let make_context = |enabled| {
        let context = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        context.set_runmat_extensions_enabled(enabled);
        context
    };
    let strict = make_context(false);
    let enabled = make_context(true);
    {
        let _foreign = enabled.enter();
        let mut single_controls =
            ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqlin, Some(&strict));
        let tol = Value::Tensor(
            Tensor::new_with_dtype(vec![1e-8], vec![1, 1], runmat_value::NumericDType::F32)
                .unwrap(),
        );
        assert!(block_on(options::Controls::parse(
            options([("TolFun", tol)]),
            false,
            &mut single_controls
        ))
        .is_ok());
        assert!(numeric::compatibility(
            &single,
            &runmat_builtins::LSQLIN_INTEGER_C_EXTENSION,
            Some(&strict),
            numeric::CompatibilityRole::Data
        )
        .is_err());
        let mut coop = ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqlin, Some(&strict));
        assert!(block_on(options::Controls::parse(
            options([("MaxIter", int.clone())]),
            false,
            &mut coop
        ))
        .is_err());
    }
    {
        let _foreign = strict.enter();
        assert!(numeric::compatibility(
            &single,
            &runmat_builtins::LSQLIN_INTEGER_C_EXTENSION,
            Some(&enabled),
            numeric::CompatibilityRole::Data
        )
        .is_ok());
        let mut coop = ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqlin, Some(&enabled));
        assert!(block_on(options::Controls::parse(
            options([("MaxIter", int.clone())]),
            false,
            &mut coop
        ))
        .is_ok());
    }
    let _foreign = strict.enter();
    for value in [single, int] {
        let result = run_in(&enabled, vec![value, Value::Num(2.0), empty(), empty()]);
        assert_eq!(values(&result[3]), vec![1.0]);
        assert!((values(&result[0])[0] - 2.0).abs() < 1e-6);
    }
}
#[test]
fn finite_workspace_policy_rejects_before_argument_work_or_capacity() {
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Managed(ManagedHostEffectLedger);
    impl RuntimeParallelService for Managed {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
    );
    let before = ledger.snapshot();
    let error = block_on(lsqlin_builtin_registered(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let after = ledger.snapshot();
    assert_eq!(before.live_claimed_bytes, after.live_claimed_bytes);
    assert_eq!(before.live_reserved_bytes, after.live_reserved_bytes);
}
