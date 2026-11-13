use super::*;
use futures::executor::block_on;
mod admission;
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
    let _scope = crate::output_count::push_output_count(Some(5)).unwrap();
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let result = block_on(execute(
        Some(&runtime),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .unwrap();
    let Value::OutputList(values) = result else {
        panic!("five outputs")
    };
    values
        .try_into_owned()
        .unwrap()
        .try_into_unbounded_vec()
        .unwrap()
}

#[test]
fn default_convex_matrix_problem_forms_and_original_duals() {
    let h = matrix(vec![1.0, -1.0, -1.0, 2.0], 2, 2);
    let f = matrix(vec![-2.0, -6.0], 2, 1);
    let result = run(vec![
        h.clone(),
        f.clone(),
        matrix(vec![1.0, -1.0, 2.0, 1.0, 2.0, 1.0], 3, 2),
        matrix(vec![2.0, 2.0, 3.0], 3, 1),
    ]);
    assert_eq!(values(&result[2]), vec![1.0]);
    let x = values(&result[0]);
    assert!((x[0] - 2.0 / 3.0).abs() < 1e-6 && (x[1] - 4.0 / 3.0).abs() < 1e-6);
    assert!((values(&result[1])[0] + 74.0 / 9.0).abs() < 1e-6);
    let z = values(field(&result[4], "ineqlin"));
    assert_eq!(z.len(), 3);
    assert!((x[0] - x[1] - 2.0 + z[0] - z[1] + 2.0 * z[2]).abs() < 1e-6);
    assert!((-x[0] + 2.0 * x[1] - 6.0 + z[0] + 2.0 * z[1] + z[2]).abs() < 1e-6);
    let problem = options([
        ("H", h),
        ("f", f),
        ("Aeq", matrix(vec![1.0, 1.0], 1, 2)),
        ("beq", Value::Num(0.0)),
        ("solver", Value::from("quadprog")),
        ("options", options([("Display", Value::from("off"))])),
    ]);
    let result = run(vec![problem]);
    assert_eq!(values(&result[2]), vec![1.0]);
    let x = values(&result[0]);
    assert!((x[0] + 0.8).abs() < 1e-6 && (x[1] - 0.8).abs() < 1e-6);
    assert!((values(&result[1])[0] + 1.6).abs() < 1e-6);
    let sparse = Value::SparseTensor(
        runmat_value::SparseTensor::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, -1.0, -1.0, 2.0],
        )
        .unwrap(),
    );
    let sparse_result = run(vec![
        sparse,
        matrix(vec![-2.0, -6.0], 1, 2),
        empty(),
        empty(),
        matrix(vec![1.0, 1.0], 1, 2),
        Value::Num(0.0),
    ]);
    assert_eq!(values(&sparse_result[2]), vec![1.0]);
    for (a, b) in values(&result[0])
        .into_iter()
        .zip(values(&sparse_result[0]))
    {
        assert!((a - b).abs() < 1e-8);
    }
}
#[test]
fn sparse_semidefinite_psd_and_near_indefinite_admission() {
    let mut cooperation = ScalarCooperation::new(runmat_builtins::OptimSolver::Quadprog, None);
    for (data, expected) in [
        (vec![1.0, 0.2, 0.2, 0.1], true), // SPD but not diagonally dominant
        (vec![1.0, 2.0, 2.0, 4.0], true), // singular and not diagonally dominant
        (vec![1.0, 1.0, 1.0, 1.0_f64.next_down()], false),
        (vec![0.0, 1.0, 1.0, 0.0], false),
    ] {
        let h = block_on(numeric::matrix(matrix(data, 2, 2), &mut cooperation))
            .unwrap()
            .unwrap();
        let p = block_on(psd::symmetric(h, &mut cooperation)).unwrap();
        assert_eq!(
            block_on(psd::certify(&p, &mut cooperation)).unwrap(),
            expected
        );
    }
    // Original exact curvature can be indefinite even when every rounded
    // numerical entry vanishes. Do not promote the rounded PSD decision.
    block_on(async {
        use crate::builtins::math::optim::common::{
            conic::convexity::{certify_original, OriginalHessian},
            model_policy::Policy,
            proof_rational::Rational,
        };
        let policy = Policy {
            identity: runmat_builtins::QUADPROG_CATALOG_ENTRY.identity,
            invalid: &runmat_builtins::QUADPROG_ERROR_INVALID_ARGUMENT,
        };
        let mut original = OriginalHessian::new(2, policy, &mut cooperation)
            .await
            .unwrap();
        let tiny = Rational::from_float(f64::from_bits(1))
            .unwrap()
            .div(&Rational::from_usize(2), policy, &mut cooperation)
            .await
            .unwrap();
        assert!(!tiny.is_zero());
        original
            .add(0, 1, tiny, policy, &mut cooperation)
            .await
            .unwrap();
        assert!(!certify_original(original, policy, &mut cooperation)
            .await
            .unwrap());
    });
    let n = 2048;
    let ptr = (0..=n).collect();
    let rows = (0..n).collect();
    let p = Matrix::new(n, n, ptr, rows, vec![1.0; n]);
    assert!(block_on(psd::certify(&p, &mut cooperation)).unwrap());
    assert_eq!(p.nzval.len(), n);
}
#[test]
fn boundary_bounds_empty_linear_term_and_truthful_terminal_statuses() {
    let zero = empty();
    let result = run(vec![
        Value::Num(2.0),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        Value::Num(1.0),
        Value::Num(1.0),
    ]);
    assert_eq!(values(&result[2]), vec![1.0]);
    assert!((values(&result[0])[0] - 1.0).abs() < 1e-7);
    assert!(
        (values(field(&result[4], "lower"))[0] - values(field(&result[4], "upper"))[0] - 2.0).abs()
            < 1e-6
    );
    let result = run(vec![
        Value::Num(1.0),
        Value::Num(0.0),
        matrix(vec![1.0, -1.0], 2, 1),
        matrix(vec![0.0, -1.0], 2, 1),
    ]);
    assert_eq!(values(&result[2]), vec![-2.0]);
    assert!(values(&result[0]).is_empty());
    let result = run(vec![
        matrix(vec![1.0, 0.0, 0.0, 0.0], 2, 2),
        matrix(vec![0.0, -1.0], 2, 1),
        zero.clone(),
        zero.clone(),
        matrix(vec![3.0, 0.0], 1, 2),
        Value::Num(1.0),
    ]);
    assert_eq!(values(&result[2]), vec![-3.0]);
    assert!(values(&result[4]).is_empty());
    let result = run(vec![Value::Num(-1.0), Value::Num(0.0)]);
    assert_eq!(values(&result[2]), vec![-6.0]);
    let result = run(vec![
        Value::Num(1.0),
        Value::Num(0.0),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero,
        options([
            ("MaxIterations", Value::Num(0.0)),
            ("Display", Value::from("off")),
        ]),
    ]);
    assert_eq!(values(&result[2]), vec![0.0]);
    assert_eq!(values(field(&result[3], "iterations")), vec![0.0]);
    assert!(values(&result[4]).is_empty());
}
#[test]
fn quadratic_recession_requires_exact_original_nullspace() {
    let mut cooperation = ScalarCooperation::new(runmat_builtins::OptimSolver::Quadprog, None);
    let p = Matrix::new(1, 1, vec![0, 1], vec![0], vec![f64::from_bits(1)]);
    let a = Matrix::zeros((0, 1));
    let f = [-1.0];
    let model = conic::Model {
        p: Some(&p),
        a: &a,
        b: &[],
        f: &f,
        cones: &[],
        identity: runmat_builtins::QUADPROG_CATALOG_ENTRY.identity,
        objective_scale: 1.0,
    };
    assert!(!block_on(conic::terminal::descent_ray(
        &model,
        &[1.0],
        &mut cooperation
    ))
    .unwrap());
    let model = conic::Model { p: None, ..model };
    assert!(block_on(conic::terminal::descent_ray(
        &model,
        &[1.0],
        &mut cooperation
    ))
    .unwrap());
}
#[test]
fn controls_keep_distinct_precision_and_reject_fake_algorithm_aliases() {
    let mut cooperation = ScalarCooperation::new(runmat_builtins::OptimSolver::Quadprog, None);
    let controls = block_on(options::Controls::parse(
        options([
            ("OptimalityTolerance", Value::Num(1e-14)),
            ("ConstraintTolerance", Value::Num(1e-10)),
        ]),
        &mut cooperation,
    ))
    .unwrap();
    assert_eq!(controls.optimality, 1e-14);
    assert_eq!(controls.feasibility, 1e-10);
    for name in ["trust-region-reflective", "interior-point"] {
        assert!(block_on(options::Controls::parse(
            options([("Algorithm", Value::from(name))]),
            &mut cooperation
        ))
        .is_err());
    }
    for name in [
        "FunctionTolerance",
        "StepTolerance",
        "OutputFcn",
        "ScaleProblem",
        "LinearSolver",
    ] {
        assert!(block_on(options::Controls::parse(
            options([(name, Value::Num(1.0))]),
            &mut cooperation
        ))
        .is_err());
    }
    active_cases::check();
}
mod active_cases;
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
    original.set_error_namespace("OriginalQP");
    let foreign = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        foreign_source.token(),
    );
    foreign_source.cancel(CancellationReason::User);
    {
        let _scope = foreign.enter();
        let mut c = ScalarCooperation::new(runmat_builtins::OptimSolver::Quadprog, Some(&original));
        assert_eq!(
            block_on(numeric::vector(Value::Num(3.0), false, false, &mut c)).unwrap(),
            vec![3.0]
        );
    }
    source.cancel(CancellationReason::User);
    let c = ScalarCooperation::new(runmat_builtins::OptimSolver::Quadprog, Some(&original));
    let error = block_on(numeric::gather(Value::Num(3.0), &c)).unwrap_err();
    assert_eq!(error.identifier(), Some("OriginalQP:Cancelled"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
    for active in [false, true] {
        let mut args = vec![Value::Num(1.), Value::Num(0.)];
        if active {
            args.resize_with(10, empty);
            args[9] = options([("Algorithm", Value::from("active-set"))]);
        }
        let error = block_on(execute(
            Some(&original),
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("OriginalQP:Cancelled"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
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
    let error = block_on(quadprog_builtin_registered(
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
