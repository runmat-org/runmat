//! Compact controls using the existing exact semantic callback fixture.
use super::*;
use futures::executor::block_on;
use runmat_value::Value;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
fn structure(values: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in values {
        fields.insert(name, value);
    }
    Value::Struct(fields.finish())
}
fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Struct(value) = value else {
        panic!("structure")
    };
    &value.fields[name]
}
fn number(value: &Value) -> f64 {
    inputs::scalar(value, "test scalar").unwrap()
}
fn args(values: Vec<Value>) -> crate::call::BuiltinArguments {
    crate::builtins::common::test_support::builtin_arguments(values)
}
fn returned(values: Vec<Value>) -> BuiltinResult<crate::execution::RootedValueSequence> {
    crate::execution::RootedValueSequence::new(
        runmat_value::ValueSequence::comma_separated(values)
            .map_err(crate::sequence::sequence_error_to_runtime)?,
    )
    .map_err(|source| source.into_runtime_error())
}
fn make_runtime() -> crate::context::RuntimeContext {
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
fn distinct_acceptance_cooling_and_random_bound_repair_are_not_local_solver_aliases() {
    assert_eq!(operators::probability(2.0, 1.0, 0.0), 1.0);
    assert_eq!(operators::probability(2.0, 2.0, 0.0), 0.5);
    assert_eq!(operators::probability(1.0, 2.0, 0.0), 0.0);
    let expected = 1.0 / (1.0 + 2.0_f64.exp());
    assert!((operators::probability(-f64::MAX, f64::MAX, f64::MAX) - expected).abs() < 1e-15);
    assert_eq!(
        operators::temperature(runmat_builtins::OptimTemperature::Fast, 100.0, 4.0).unwrap(),
        25.0
    );
    assert!(
        (operators::temperature(runmat_builtins::OptimTemperature::Exponential, 100.0, 2.0)
            .unwrap()
            - 90.25)
            .abs()
            < 1e-12
    );
    let runtime = make_runtime();
    let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, Some(&runtime));
    let repaired = block_on(operators::honor_bounds(
        vec![-100.0, 100.0],
        &[2.0, 3.0],
        &[0.0, 0.0],
        &[5.0, 5.0],
        &mut cooperation,
    ))
    .unwrap();
    assert!((0.0..=2.0).contains(&repaired[0]));
    assert!((3.0..=5.0).contains(&repaired[1]));
    assert_ne!(
        repaired,
        vec![0.0, 5.0],
        "a bound violation is not just clipped"
    );
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("sa_default", 29200)],
    )
    .unwrap();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |id, args, requested| {
            assert_eq!((id, requested), (29200, 1));
            let value = number(&args[0]).powi(2);
            Box::pin(async move { returned(vec![Value::Num(value)]) })
        },
    )))
    .unwrap();
    let generated = block_on(crate::dispatcher::call_builtin_async_in_context(
        &runtime,
        "optimoptions",
        &[
            Value::FunctionHandle("simulannealbnd".into()),
            Value::from("MaxIterations"),
            Value::Num(1.0),
            Value::from("InitialTemperature"),
            Value::Num(100.0),
            Value::from("Display"),
            Value::from("off"),
        ],
    ))
    .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(4)).unwrap();
    let result = block_on(execute(
        Some(&runtime),
        args(vec![
            Value::FunctionHandle("sa_default".into()),
            Value::Num(2.0),
            Value::Num(2.0),
            Value::Num(2.0),
            generated.value().clone(),
        ]),
    ))
    .unwrap();
    assert_eq!(number(&result.as_slice()[0]), 2.0);
    assert_eq!(number(&result.as_slice()[1]), 4.0);
    assert_eq!(number(field(&result.as_slice()[3], "funccount")), 2.0);
    assert!(
        (number(field(&result.as_slice()[3], "temperature")) - 95.0).abs() < 1e-12,
        "the first completed iteration uses k=1, not k=2"
    );
    runtime.set_runmat_extensions_enabled(true);
    let maximum = Value::Int(runmat_value::IntValue::U64(usize::MAX as u64));
    let mut maximum_options = block_on(options::Options::parse(
        structure([
            ("MaxFunctionEvaluations", maximum.clone()),
            ("MaxStallIterations", maximum),
            ("Display", Value::from("off")),
        ]),
        &mut cooperation,
    ))
    .unwrap();
    block_on(maximum_options.resolve(1, &mut cooperation)).unwrap();
    assert_eq!(maximum_options.evaluations, usize::MAX);
    assert_eq!(maximum_options.stall, usize::MAX);
}
#[test]
fn accepted_generation_reanneals_with_real_budgeted_probes_and_original_four_output_authority() {
    let runtime = make_runtime();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[
            ("sa_objective", 29201),
            ("sa_proposal", 29202),
            ("sa_accept", 29203),
        ],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let measured = calls.clone();
    let _entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, requested| {
            assert_eq!(requested, 1);
            let value = match id {
                29201 => {
                    measured.fetch_add(1, Ordering::SeqCst);
                    Value::Num(number(&args[0]).powi(2))
                }
                29202 => {
                    assert_eq!(args.len(), 2);
                    Value::Num(number(field(&args[0], "x")) - 1.0)
                }
                29203 => {
                    assert_eq!(args.len(), 3);
                    Value::Bool(true)
                }
                _ => panic!("selected target"),
            };
            Box::pin(async move { returned(vec![value]) })
        },
    )))
    .unwrap();
    let options = structure([
        ("Display", Value::from("off")),
        ("MaxIterations", Value::Num(1.0)),
        ("MaxFunctionEvaluations", Value::Num(3.0)),
        ("ReannealInterval", Value::Num(1.0)),
        ("AnnealingFcn", Value::FunctionHandle("sa_proposal".into())),
        ("AcceptanceFcn", Value::FunctionHandle("sa_accept".into())),
    ]);
    let request = args(vec![
        Value::FunctionHandle("sa_objective".into()),
        Value::Num(3.0),
        Value::Num(-10.0),
        Value::Num(10.0),
        options,
    ]);
    let _outputs = crate::output_count::push_output_count(Some(4)).unwrap();
    let foreign = make_runtime();
    let _foreign = foreign.enter();
    let _foreign_outputs = crate::output_count::push_output_count(Some(1)).unwrap();
    let result = block_on(execute(Some(&runtime), request)).unwrap();
    assert_eq!(result.len(), 4);
    let result = result.as_slice();
    assert_eq!(number(&result[0]), 2.0);
    assert_eq!(number(&result[1]), 4.0);
    assert_eq!(number(&result[2]), 0.0);
    assert_eq!(number(field(&result[3], "funccount")), 3.0);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "initial, proposal, and one actual sensitivity probe"
    );
}
#[test]
fn custom_payload_and_options_first_three_output_observer_change_the_actual_next_generation() {
    let runtime = make_runtime();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[
            ("sa_custom_objective", 29211),
            ("sa_custom_proposal", 29212),
            ("sa_output", 29213),
        ],
    )
    .unwrap();
    let phases = Arc::new(Mutex::new(Vec::new()));
    let seen = phases.clone();
    let _entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, requested| {
            let values = match id {
                29211 => {
                    assert_eq!(requested, 1);
                    vec![Value::Num(number(field(&args[0], "score")).powi(2))]
                }
                29212 => {
                    assert_eq!(requested, 1);
                    assert_eq!(args.len(), 2);
                    vec![structure([(
                        "score",
                        Value::Num(number(field(field(&args[0], "x"), "score")) - 1.0),
                    )])]
                }
                29213 => {
                    assert_eq!(requested, 3);
                    assert_eq!(args.len(), 3);
                    assert!(
                        matches!(field(&args[0], "MaxIterations"), Value::Num(_)),
                        "options is the FIRST argument"
                    );
                    let Value::String(runmat_value::StringElement::Text(phase)) = &args[2] else {
                        panic!("phase")
                    };
                    seen.lock().unwrap().push(phase.to_string());
                    let replacement = match &args[0] {
                        Value::Struct(v) => v.clone(),
                        _ => panic!("options"),
                    };
                    // Reconstruct through the ordinary builder, not an alternate
                    // callback result carrier or a hidden loop-control switch.
                    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
                    for (name, value) in &replacement.fields {
                        if name != "MaxIterations" {
                            fields.insert(name, value.clone());
                        }
                    }
                    fields.insert("MaxIterations", Value::Num(1.0));
                    vec![
                        Value::Bool(false),
                        Value::Struct(fields.finish()),
                        Value::Bool(phase.as_str() == "init"),
                    ]
                }
                _ => panic!("target"),
            };
            Box::pin(async move { returned(values) })
        },
    )))
    .unwrap();
    let options = structure([
        ("DataType", Value::from("custom")),
        ("Display", Value::from("off")),
        ("MaxIterations", Value::Num(10.0)),
        ("ReannealInterval", Value::Num(0.0)),
        (
            "AnnealingFcn",
            Value::FunctionHandle("sa_custom_proposal".into()),
        ),
        ("OutputFcn", Value::FunctionHandle("sa_output".into())),
    ]);
    let _outputs = crate::output_count::push_output_count(Some(4)).unwrap();
    let result = block_on(execute(
        Some(&runtime),
        args(vec![
            Value::FunctionHandle("sa_custom_objective".into()),
            structure([("score", Value::Num(2.0))]),
            inputs::empty(),
            inputs::empty(),
            options,
        ]),
    ))
    .unwrap();
    assert_eq!(number(field(&result.as_slice()[0], "score")), 1.0);
    assert_eq!(number(&result.as_slice()[1]), 1.0);
    assert_eq!(number(field(&result.as_slice()[3], "funccount")), 2.0);
    assert_eq!(*phases.lock().unwrap(), vec!["init", "iter", "done"]);
}
