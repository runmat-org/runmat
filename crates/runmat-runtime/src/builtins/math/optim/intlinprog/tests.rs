use super::*;
use futures::executor::block_on;
mod residency;
mod observers;
mod domains;
fn context() -> crate::context::RuntimeContext {
    crate::user_functions::test_support::exact_fixture_runtime()
}
fn matrix(values: Vec<f64>, m: usize, n: usize) -> Value {
    Value::Tensor(Tensor::new(values, vec![m, n]).unwrap())
}
fn values(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(v) => vec![*v],
        Value::Tensor(v) => v.materialize_f64(),
        _ => panic!("numeric result"),
    }
}
fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Struct(s) = value else {
        panic!("information struct")
    };
    &s.fields[name]
}
fn options(extra: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("Display", Value::from("off"));
    for (name, value) in extra {
        fields.insert(name, value);
    }
    Value::Struct(fields.finish())
}
fn run(runtime: &crate::context::RuntimeContext, args: Vec<Value>) -> Vec<Value> {
    let _count = crate::output_count::push_output_count(Some(4)).unwrap();
    let result = block_on(execute(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .unwrap();
    let Value::OutputList(result) = result else {
        panic!("four results")
    };
    result
        .try_into_owned()
        .unwrap()
        .try_into_unbounded_vec()
        .unwrap()
}
fn binary_problem(options: Value) -> Vec<Value> {
    vec![
        matrix(vec![-1.0, -1.0], 2, 1),
        matrix(vec![1.0, 2.0], 1, 2),
        matrix(vec![2.0, 2.0], 1, 2),
        Value::Num(3.0),
        empty(),
        empty(),
        matrix(vec![0.0, 0.0], 2, 1),
        matrix(vec![1.0, 1.0], 2, 1),
        empty(),
        options,
    ]
}
#[test]
fn actual_binary_branching_rejects_rounding_and_preserves_original_objective() {
    let result = run(&context(), binary_problem(options([])));
    assert_eq!(values(&result[2]), vec![1.0]);
    let x = values(&result[0]);
    assert_eq!(x.len(), 2);
    assert!(x.iter().all(|v| *v == 0.0 || *v == 1.0));
    assert_eq!(x.iter().sum::<f64>(), 1.0);
    assert_eq!(values(&result[1]), vec![-1.0]);
    assert!(
        values(field(&result[3], "numnodes"))[0] >= 2.0,
        "fractional LP rounding is not an integer solver"
    );
    assert!(values(field(&result[3], "absolutegap"))[0] <= 1e-6);
    assert_eq!(values(field(&result[3], "constrviolation")), vec![0.0]);
}
#[test]
fn range_feasibility_problem_form_uses_initial_fixed_integer_repair() {
    let mut problem = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in [
        ("f", empty()),
        ("intcon", Value::Num(1.0)),
        ("Aineq", Value::Num(1.0)),
        (
            "bineq",
            Value::Cell(
                runmat_value::CellArray::new(vec![Value::Num(1.2), Value::Num(2.2)], 1, 2).unwrap(),
            ),
        ),
        ("lb", Value::Num(0.0)),
        ("ub", Value::Num(3.0)),
        ("x0", Value::Num(1.5)),
        ("solver", Value::from("intlinprog")),
        ("options", options([])),
    ] {
        problem.insert(name, value);
    }
    let result = run(&context(), vec![Value::Struct(problem.finish())]);
    assert_eq!(values(&result[0]), vec![2.0]);
    assert_eq!(values(&result[1]), vec![0.0]);
    assert_eq!(values(&result[2]), vec![1.0]);
    assert!(values(field(&result[3], "numfeaspoints"))[0] >= 1.0);
    assert_eq!(values(field(&result[3], "absolutegap")), vec![0.0]);
}
#[test]
fn original_equality_integer_conflict_and_root_lp_ray_have_distinct_proofs() {
    let runtime = context();
    let conflict = run(
        &runtime,
        vec![
            Value::Num(1.0),
            Value::Num(1.0),
            empty(),
            empty(),
            Value::Num(2.0),
            Value::Num(1.0),
            Value::Num(0.0),
            Value::Num(1.0),
            empty(),
            options([]),
        ],
    );
    assert_eq!(values(&conflict[2]), vec![-2.0]);
    assert!(values(&conflict[0]).is_empty());
    let ray = run(
        &runtime,
        vec![
            Value::Num(-1.0),
            Value::Num(1.0),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(0.0),
            empty(),
            empty(),
            options([]),
        ],
    );
    assert_eq!(values(&ray[2]), vec![-3.0]);
    let lp = run(
        &runtime,
        vec![
            Value::Num(-1.0),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(0.0),
            Value::Num(2.0),
            empty(),
            options([("RelativeGapTolerance", Value::Num(0.0))]),
        ],
    );
    assert_eq!(values(&lp[2]), vec![1.0]);
    let x = values(&lp[0])[0];
    assert!((0.0..=2.0).contains(&x));
    assert!(2.0 - x <= 1e-6, "the requested absolute gap is satisfied");
    assert_eq!(values(&lp[1]), vec![-x]);
    for key in ["relativegap", "absolutegap", "numfeaspoints", "numnodes"] {
        assert!(values(field(&lp[3], key)).is_empty());
    }
}

#[test]
fn unbounded_relaxation_with_no_integer_feasible_point_branches_instead_of_claiming_milp_unbounded()
{
    let result = run(
        &context(),
        vec![
            matrix(vec![0.0, -1.0], 2, 1),
            Value::Num(1.0),
            empty(),
            empty(),
            matrix(vec![1.0, 0.0], 1, 2),
            Value::Num(0.5),
            empty(),
            empty(),
            empty(),
            options([]),
        ],
    );
    assert_eq!(values(&result[2]), vec![-2.0]);
    assert!(values(&result[0]).is_empty());
}

#[test]
fn actual_owned_rational_ray_and_exact_fractional_branch_keep_the_lattice_proof() {
    use common::conic::{self, terminal};
    use num_rational::BigRational;
    let runtime = context();
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::INTLINPROG_CATALOG_ENTRY.identity,
        Some(&runtime),
    );
    let a = clarabel::algebra::CscMatrix::zeros((0, 2));
    let f = [-1.0, 0.0];
    let model = conic::Model {
        p: None,
        a: &a,
        b: &[],
        f: &f,
        cones: &[],
        identity: runmat_builtins::INTLINPROG_CATALOG_ENTRY.identity,
        objective_scale: 1.0,
    };
    let point = block_on(terminal::verified_point(
        &model,
        &[0.0, 0.0],
        &mut cooperation,
    ))
    .unwrap()
    .unwrap();
    let ray = block_on(terminal::verified_descent_ray(
        &model,
        &[0.5, 1.0 / 3.0],
        &mut cooperation,
    ))
    .unwrap()
    .unwrap();
    assert!(block_on(proofs::integral_unbounded(
        &point,
        &ray,
        &[0, 1],
        &[],
        &mut cooperation
    ))
    .unwrap());
    let semi = super::domains::Semi {
        column: 1,
        lower: 2.0,
        upper: 5.0,
    };
    assert!(
        !block_on(proofs::integral_unbounded(
            &point,
            &ray,
            &[0],
            &[semi],
            &mut cooperation,
        ))
        .unwrap(),
        "a nonzero semi recession direction is not a union-domain proof"
    );
    let stationary_semi_ray = block_on(terminal::verified_descent_ray(
        &model,
        &[0.5, 0.0],
        &mut cooperation,
    ))
    .unwrap()
    .unwrap();
    assert!(block_on(proofs::integral_unbounded(
        &point,
        &stationary_semi_ray,
        &[0],
        &[semi],
        &mut cooperation,
    ))
    .unwrap());
    let fractional = block_on(terminal::verified_point(
        &model,
        &[0.5, 0.0],
        &mut cooperation,
    ))
    .unwrap()
    .unwrap();
    assert!(!block_on(proofs::integral_unbounded(
        &fractional,
        &ray,
        &[0],
        &[],
        &mut cooperation
    ))
    .unwrap());
    let large = num_bigint::BigInt::from(1u64 << 53);
    let half = BigRational::new(1.into(), 2.into());
    let near = BigRational::from_integer(large.clone()) + &half;
    assert!(block_on(engine::exact_branch(
        std::slice::from_ref(&near),
        &[0],
        &mut cooperation
    ))
    .unwrap()
    .is_none());
    assert_eq!(
        block_on(engine::exact_branch(
            &[near, half.clone()],
            &[0, 1],
            &mut cooperation
        ))
        .unwrap(),
        Some((1, 0.0, 1.0))
    );
    let negative = -BigRational::from_integer(large) + half;
    assert_eq!(
        block_on(engine::exact_branch(&[negative], &[0], &mut cooperation)).unwrap(),
        Some((0, -9_007_199_254_740_992.0, -9_007_199_254_740_991.0))
    );
}

#[test]
fn binary64_publication_recomputes_original_violation_and_never_labels_it_feasible() {
    use num_rational::BigRational;
    let runtime = context();
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::INTLINPROG_CATALOG_ENTRY.identity,
        Some(&runtime),
    );
    let mut controls = controls::Controls::defaults(&runtime).unwrap();
    controls.display = runmat_builtins::OptimDisplay::Off;
    for (coefficient, rhs) in [(3.0, 2f64.powi(330)), (0.25, f64::MAX)] {
        let exact =
            BigRational::from_float(rhs).unwrap() / BigRational::from_float(coefficient).unwrap();
        let problem = model::Problem {
            a: clarabel::algebra::CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![coefficient]),
            b: vec![rhs],
            f: vec![1.0],
            equalities: 1,
            lower: vec![f64::NEG_INFINITY],
            upper: vec![f64::INFINITY],
            integers: vec![],
            semis: vec![],
            initial: None,
        };
        let workspace = block_on(model::Workspace::new(problem, &mut cooperation)).unwrap();
        let outcome = engine::Outcome {
            incumbent: Some(proofs::Incumbent {
                point: vec![exact.clone()],
                objective: exact.clone(),
            }),
            lower: Some(exact),
            terminal: engine::Terminal::Optimal,
        };
        let _count = crate::output_count::push_output_count(Some(4)).unwrap();
        let result = block_on(output::finalize(
            outcome,
            &workspace,
            &controls,
            &mut cooperation,
        ))
        .unwrap();
        let Value::OutputList(result) = result else {
            panic!("four outputs")
        };
        let result = result
            .try_into_owned()
            .unwrap()
            .try_into_unbounded_vec()
            .unwrap();
        assert_eq!(values(&result[2]), vec![-9.0]);
        assert!(values(&result[0]).is_empty());
        assert!(values(&result[1]).is_empty());
        let violation = values(field(&result[3], "constrviolation"))[0];
        if coefficient == 3.0 {
            assert!(violation > controls.constraint_tolerance);
        } else {
            assert!(
                violation.is_nan(),
                "no finite point was available to inspect"
            );
        }
    }
}
#[test]
fn node_and_cutoff_stops_retain_the_real_lower_certificate_not_fake_optimality() {
    let runtime = context();
    let stopped = run(
        &runtime,
        binary_problem(options([("MaxNodes", Value::Num(1.0))])),
    );
    assert_eq!(values(&stopped[2]), vec![0.0]);
    assert!(values(&stopped[0]).is_empty());
    assert_eq!(values(field(&stopped[3], "numnodes")), vec![1.0]);
    assert!(values(field(&stopped[3], "lowerbound"))[0] <= -1.0);
    assert_eq!(
        values(field(&stopped[3], "absolutegap")),
        vec![f64::INFINITY]
    );
    let cutoff = run(
        &runtime,
        binary_problem(options([("ObjectiveCutOff", Value::Num(-2.0))])),
    );
    assert_eq!(values(&cutoff[2]), vec![-2.0]);
    let lower = values(field(&cutoff[3], "lowerbound"))[0];
    assert!(lower.is_finite() && lower > -2.0 && lower <= -1.0);
}
#[test]
fn exact_gap_comparisons_do_not_round_away_tiny_or_zero_objective_gaps() {
    use num_rational::BigRational;
    let zero = BigRational::from_integer(0.into());
    let one = BigRational::from_integer(1.into());
    assert!(!gap::satisfies(&zero, &(-&one), 0.0, 1.0));
    assert!(gap::satisfies(&zero, &zero, 0.0, 0.0));
    let tiny = BigRational::new(1.into(), num_bigint::BigInt::from(1u8) << 1100);
    assert!(!gap::satisfies(&one, &(&one - &tiny), 0.0, 0.0));
}
#[test]
fn selected_logical_deadline_is_typed_and_survives_numeric_finalization() {
    use crate::context::DeterministicTimingService;
    use common::conic::solve::IterationControl;
    use std::rc::Rc;
    let runtime = context();
    let clock = Rc::new(
        DeterministicTimingService::new(
            runmat_execution::host::NativeClockInputs {
                schema_version: runmat_execution::host::NATIVE_CLOCK_INPUTS_SCHEMA_VERSION,
                initial_monotonic_nanos: 0,
                initial_wall_unix_seconds: 0,
                initial_wall_nanosecond: 0,
                read_step_nanos: 1_000_000_000,
                local_offset_seconds: 0,
                process_cpu: None,
            },
            runtime.cancellation(),
        )
        .unwrap(),
    );
    let ports = runtime.service_ports().clone().with_timing(clock);
    let runtime = runtime.with_service_ports(ports);
    let mut controls = controls::Controls::defaults(&runtime).unwrap();
    controls.max_time = 1.0;
    assert_eq!(
        controls.progress_control(),
        clarabel::solver::ProgressControl::Stop(clarabel::solver::ProgressStop::TimeLimit)
    );
    controls.max_time = f64::INFINITY;
    assert!(
        matches!(controls.stopped(), Some(engine::Stop::Time)),
        "postprocessing cannot erase an observed semantic deadline"
    );
    let denied = context();
    let ports = denied.service_ports().clone().without_timing();
    let denied = denied.with_service_ports(ports);
    let error = controls::Controls::defaults(&denied).err().unwrap();
    assert_eq!(error.identifier(), Some("RunMat:Clock:ServiceUnavailable"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}
#[test]
fn original_cancellation_and_missing_context_refuse_before_solver_work() {
    let error = block_on(execute(
        None,
        crate::builtins::common::test_support::builtin_arguments(binary_problem(options([]))),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    let cancellation = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    );
    runtime.set_error_namespace("OriginalMILP");
    cancellation.cancel(CancellationReason::User);
    let error = block_on(execute(
        Some(&runtime),
        crate::builtins::common::test_support::builtin_arguments(binary_problem(options([]))),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("OriginalMILP:Cancelled"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
}
