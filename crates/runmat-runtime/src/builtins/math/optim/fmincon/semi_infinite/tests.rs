use super::*;
use crate::BuiltinResult;
use futures::executor::block_on;
use runmat_value::{Tensor, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
fn empty() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap())
}
fn structure(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut out = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in fields {
        out.insert(name, value);
    }
    Value::Struct(out.finish())
}
fn args(values: Vec<Value>) -> crate::call::BuiltinArguments {
    crate::builtins::common::test_support::builtin_arguments(values)
}
fn numbers(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(v) => vec![*v],
        Value::Tensor(v) => v.materialize_f64(),
        _ => panic!("numeric"),
    }
}
fn result(values: Vec<Value>) -> BuiltinResult<crate::execution::RootedValueSequence> {
    crate::execution::RootedValueSequence::new(
        runmat_value::ValueSequence::comma_separated(values)
            .map_err(crate::sequence::sequence_error_to_runtime)?,
    )
    .map_err(|e| e.into_runtime_error())
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
#[test]
fn public_sqp_resolves_off_grid_peak_and_publishes_only_original_rows() {
    for central in [false, true] {
        let runtime = runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[
                ("semi_public_objective", 2611),
                ("semi_public_constraints", 2612),
            ],
        )
        .unwrap();
        let objective = Arc::new(AtomicUsize::new(0));
        let constraints = Arc::new(AtomicUsize::new(0));
        let oc = objective.clone();
        let cc = constraints.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, values, requested| {
                let x = numbers(&values[0])[0];
                let outputs = if id == 2611 {
                    oc.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(requested, 2);
                    vec![Value::Num((x - 1.0).powi(2)), Value::Num(2.0 * (x - 1.0))]
                } else {
                    assert_eq!(id, 2612);
                    assert_eq!(requested, 4);
                    cc.fetch_add(1, Ordering::SeqCst);
                    let spacing = numbers(&values[1]);
                    let h = if spacing[0].is_nan() { 0.5 } else { spacing[0] };
                    let count = (1.0 / h).round() as usize + 1;
                    let grid = (0..count)
                        .map(|i| x - 0.5 - (i as f64 * h - 0.37).powi(2))
                        .collect();
                    vec![
                        empty(),
                        empty(),
                        Value::Tensor(Tensor::new(grid, vec![1, count]).unwrap()),
                        Value::Tensor(Tensor::new(vec![h, 0.0], vec![1, 2]).unwrap()),
                    ]
                };
                Box::pin(async move { result(outputs) })
            },
        )))
        .unwrap();
        let options = structure([
            ("Display", Value::from("off")),
            ("SpecifyObjectiveGradient", Value::Bool(true)),
            (
                "FiniteDifferenceType",
                Value::from(if central { "central" } else { "forward" }),
            ),
            ("FiniteDifferenceStepSize", Value::Num(1e-5)),
            ("TypicalX", Value::Num(2.0)),
            ("DiffMinChange", Value::Num(1e-7)),
            ("DiffMaxChange", Value::Num(1e-3)),
            ("MaxSQPIter", Value::Num(100.0)),
            ("TolConSQP", Value::Num(1e-8)),
            ("StepTolerance", Value::Num(1e-9)),
            ("FunctionTolerance", Value::Num(1e-9)),
        ]);
        let _count = crate::output_count::push_output_count(Some(5)).unwrap();
        let returned = block_on(execute(
            Some(&runtime),
            args(vec![
                Value::FunctionHandle("semi_public_objective".into()),
                Value::Num(0.2),
                Value::Num(1.0),
                Value::FunctionHandle("semi_public_constraints".into()),
                empty(),
                empty(),
                empty(),
                empty(),
                Value::Num(0.0),
                Value::Num(2.0),
                options,
            ]),
        ))
        .unwrap();
        let values = returned.as_slice();
        assert!((numbers(&values[0])[0] - 0.5).abs() < 1e-5);
        assert!((numbers(&values[1])[0] - 0.25).abs() < 1e-5);
        assert!(numbers(&values[2])[0] > 0.0);
        let Value::Struct(output) = &values[3] else {
            panic!("output")
        };
        assert_eq!(
            numbers(&output.fields["objectiveCallCount"])[0],
            objective.load(Ordering::SeqCst) as f64
        );
        assert_eq!(
            numbers(&output.fields["constraintCallCount"])[0],
            constraints.load(Ordering::SeqCst) as f64
        );
        assert_eq!(
            numbers(&output.fields["funcCount"])[0],
            constraints.load(Ordering::SeqCst) as f64
        );
        let Value::Struct(lambda) = &values[4] else {
            panic!("lambda")
        };
        assert!(
            numbers(&lambda.fields["ineqnonlin"]).is_empty(),
            "interpolated rows are internal"
        );
        assert!(
            constraints.load(Ordering::SeqCst) > objective.load(Ordering::SeqCst),
            "Jacobian probes never replay fun"
        );
    }
}
#[test]
fn full_problem_form_inconsistent_bounds_never_calls_callbacks() {
    let runtime = runtime();
    let _entry = runtime.enter();
    let _count = crate::output_count::push_output_count(Some(5)).unwrap();
    let error = block_on(execute(
        Some(&runtime),
        args(vec![
            Value::FunctionHandle("must_not_run".into()),
            Value::Num(3.0),
            Value::Num(0.0),
            Value::FunctionHandle("must_not_run_either".into()),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::FSEMINF_ERROR_INVALID_ARGUMENT.identifier
    );
    let problem = structure([
        ("objective", Value::FunctionHandle("must_not_run".into())),
        ("x0", Value::Num(3.0)),
        ("ntheta", Value::Num(1.0)),
        (
            "seminfcon",
            Value::FunctionHandle("must_not_run_either".into()),
        ),
        ("solver", Value::from("fseminf")),
        ("options", empty()),
        ("lb", Value::Num(4.0)),
        ("ub", Value::Num(2.0)),
    ]);
    let returned = block_on(execute(Some(&runtime), args(vec![problem]))).unwrap();
    let values = returned.as_slice();
    assert_eq!(numbers(&values[0]), vec![3.0]);
    assert!(numbers(&values[1]).is_empty());
    assert_eq!(numbers(&values[2]), vec![-2.0]);
    let Value::Struct(output) = &values[3] else {
        panic!("output")
    };
    assert_eq!(numbers(&output.fields["funcCount"]), vec![0.0]);
    assert!(numbers(&values[4]).is_empty());
}
