//! Compact controls in the existing exact semantic-invoker fixture.
use super::*;
use futures::executor::block_on;
use runmat_value::{Tensor, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
fn args(values: Vec<Value>) -> crate::call::BuiltinArguments {
    crate::builtins::common::test_support::builtin_arguments(values)
}
fn object(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut out = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in fields {
        out.insert(name, value);
    }
    Value::Struct(out.finish())
}
fn matrix(values: Vec<f64>, rows: usize, columns: usize) -> Value {
    Value::Tensor(Tensor::new(values, vec![rows, columns]).unwrap())
}
fn numeric(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(v) => vec![*v],
        Value::Tensor(v) => v.materialize_f64(),
        _ => panic!("numeric"),
    }
}
fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Struct(value) = value else {
        panic!("structure")
    };
    &value.fields[name]
}
#[test]
fn supplied_fitness_only_reuses_exact_coordinates_and_public_outputs_use_original_request() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("ps_quadratic", 19101), ("ps_creation", 19105)],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let _entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, requested| {
            assert_eq!(requested, 1);
            let value = if id == 19105 {
                assert_eq!(args.len(), 1);
                assert_eq!(
                    numeric(field(field(&args[0], "options"), "SwarmSize")),
                    vec![3.0]
                );
                assert_eq!(
                    numeric(field(field(&args[0], "options"), "MaxIterations")),
                    vec![0.0]
                );
                matrix(vec![-2.0, 0.0, 1.0], 3, 1)
            } else {
                assert_eq!(id, 19101);
                counted.fetch_add(1, Ordering::SeqCst);
                let x = numeric(&args[0]);
                assert_eq!(x.len(), 1);
                Value::Num(x[0] * x[0])
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(value)
            })
        },
    )))
    .unwrap();
    let initial = object([
        ("X", matrix(vec![-3.0, -0.0, 1.0], 3, 1)),
        ("Fval", matrix(vec![9.0, 0.0], 2, 1)),
    ]);
    let options = object([
        ("Display", Value::from("off")),
        ("SwarmSize", Value::Num(3.0)),
        ("MaxIterations", Value::Num(0.0)),
        ("InitialPoints", initial),
        ("CreationFcn", Value::FunctionHandle("ps_creation".into())),
    ]);
    let request = args(vec![
        Value::FunctionHandle("ps_quadratic".into()),
        Value::Num(1.0),
        Value::Num(-2.0),
        Value::Num(2.0),
        options,
    ]);
    let _outputs = crate::output_count::push_output_count(Some(5)).unwrap();
    let foreign = crate::user_functions::test_support::exact_fixture_runtime();
    let _foreign_entry = foreign.enter();
    let _foreign_outputs = crate::output_count::push_output_count(Some(1)).unwrap();
    let result = block_on(execute(Some(&runtime), request)).unwrap();
    assert_eq!(
        result.len(),
        5,
        "foreign ambient one-output request must not replace original five"
    );
    let values = result.as_slice();
    assert_eq!(numeric(&values[0]), vec![0.0]);
    assert_eq!(numeric(&values[1]), vec![0.0]);
    assert_eq!(numeric(&values[2]), vec![0.0]);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "custom +0 must not reuse a supplied -0 measurement"
    );
    assert_eq!(numeric(field(&values[3], "funccount")), vec![3.0]);
    assert_eq!(numeric(field(&values[4], "X")), vec![-2.0, 0.0, 1.0]);
    assert_eq!(numeric(field(&values[4], "Fval")), vec![4.0, 0.0, 1.0]);
}
#[test]
fn vectorized_active_rows_and_two_argument_observers_leave_parallel_carrier_dormant() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("ps_vectorized", 19102), ("ps_observer", 19103)],
    )
    .unwrap();
    let _entry = runtime.enter();
    let batches = Arc::new(Mutex::new(Vec::new()));
    let measured = batches.clone();
    let phases = Arc::new(Mutex::new(Vec::new()));
    let seen = phases.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, requested| {
            assert_eq!(requested, 1);
            let value = if id == 19102 {
                assert_eq!(args.len(), 1);
                let Value::Tensor(x) = &args[0] else {
                    panic!("matrix")
                };
                measured.lock().unwrap().push(x.shape().to_vec());
                let rows = x.shape()[0];
                let cols = x.shape()[1];
                let x = x.as_f64_slice().unwrap();
                let f = (0..rows)
                    .map(|row| (0..cols).map(|col| x[row + rows * col].powi(2)).sum())
                    .collect();
                matrix(f, rows, 1)
            } else {
                assert_eq!(id, 19103);
                assert_eq!(args.len(), 2);
                let Value::String(runmat_value::StringElement::Text(phase)) = &args[1] else {
                    panic!("phase")
                };
                seen.lock().unwrap().push(phase.to_string());
                assert_eq!(numeric(field(&args[0], "funccount")), vec![2.0]);
                assert_eq!(numeric(field(&args[0], "swarmfvals")), vec![2.0, 0.0, 2.0]);
                Value::Bool(phase.as_str() == "init")
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(value)
            })
        },
    )))
    .unwrap();
    let initial = object([
        ("X", matrix(vec![-1.0, 0.0, 1.0, 1.0, 0.0, -1.0], 3, 2)),
        ("Fval", matrix(vec![2.0], 1, 1)),
    ]);
    let options = object([
        ("Display", Value::from("off")),
        ("SwarmSize", Value::Num(3.0)),
        ("InitialPoints", initial),
        ("UseVectorized", Value::Bool(true)),
        ("UseParallel", object([("bad", Value::from("dormant"))])),
        ("OutputFcn", Value::FunctionHandle("ps_observer".into())),
    ]);
    let _outputs = crate::output_count::push_output_count(Some(5)).unwrap();
    let result = block_on(execute(
        Some(&runtime),
        args(vec![
            Value::FunctionHandle("ps_vectorized".into()),
            Value::Num(2.0),
            matrix(vec![-2.0, -2.0], 1, 2),
            matrix(vec![2.0, 2.0], 1, 2),
            options,
        ]),
    ))
    .unwrap();
    assert_eq!(*batches.lock().unwrap(), vec![vec![2, 2]]);
    assert_eq!(*phases.lock().unwrap(), vec!["init", "done"]);
    assert_eq!(numeric(&result.as_slice()[2]), vec![-1.0]);
    assert_eq!(
        numeric(field(&result.as_slice()[3], "swarmObjectiveCallCount")),
        vec![1.0]
    );
}
#[test]
fn a_real_generation_updates_velocities_bests_and_keeps_actual_final_measurements() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("ps_generation", 19104)],
    )
    .unwrap();
    let _entry = runtime.enter();
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, _| {
            assert_eq!(id, 19104);
            counted.fetch_add(1, Ordering::SeqCst);
            let x = numeric(&args[0]);
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                    (x[0] - 0.25).powi(2),
                ))
            })
        },
    )))
    .unwrap();
    let options = object([
        ("Display", Value::from("off")),
        ("SwarmSize", Value::Num(4.0)),
        ("MaxIterations", Value::Num(3.0)),
        ("FunctionTolerance", Value::Num(0.0)),
        ("InitialPoints", matrix(vec![-1.0, -0.5, 0.5, 1.0], 4, 1)),
        ("InitialSwarmSpan", Value::Num(0.5)),
    ]);
    let _outputs = crate::output_count::push_output_count(Some(5)).unwrap();
    let result = block_on(execute(
        Some(&runtime),
        args(vec![
            Value::FunctionHandle("ps_generation".into()),
            Value::Num(1.0),
            Value::Num(-1.0),
            Value::Num(1.0),
            options,
        ]),
    ))
    .unwrap();
    let values = result.as_slice();
    let x = numeric(&values[0])[0];
    let f = numeric(&values[1])[0];
    assert_eq!(numeric(&values[2]), vec![0.0]);
    assert_eq!(calls.load(Ordering::SeqCst), 16);
    assert_eq!(numeric(field(&values[3], "iterations")), vec![3.0]);
    assert!((f - (x - 0.25).powi(2)).abs() < 1e-14);
    assert!(f <= 0.25f64.powi(2));
    let positions = numeric(field(&values[4], "X"));
    let fitness = numeric(field(&values[4], "Fval"));
    for (&x, &f) in positions.iter().zip(&fitness) {
        assert!((-1.0..=1.0).contains(&x));
        assert!((f - (x - 0.25).powi(2)).abs() < 1e-14);
    }
}
#[test]
fn dynamics_adapts_neighborhood_inertia_and_zero_tolerance_does_not_fake_stall_success() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, Some(&runtime));
    block_on(async {
        let population = population::Population::new(
            super::POLICY,
            Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap(),
            4,
            1,
        )
        .unwrap();
        let parameters = dynamics::Parameters {
            inertia: [0.1, 1.1],
            neighbors_fraction: 0.25,
            personal_weight: 1.49,
            social_weight: 1.49,
        };
        let mut dynamics = dynamics::Dynamics::new(
            vec![0.0; 4],
            &population,
            vec![Some(0.0), Some(1.0), Some(4.0), Some(9.0)],
            &parameters,
        )
        .unwrap();
        assert_eq!(dynamics.neighborhood, 2);
        assert_eq!(dynamics.inertia, 1.1);
        dynamics
            .accept(
                &population,
                &[Some(0.0), Some(1.0), Some(4.0), Some(9.0)],
                false,
                &parameters,
                &mut cooperation,
            )
            .await
            .unwrap();
        assert_eq!(dynamics.neighborhood, 4);
        dynamics
            .accept(
                &population,
                &[Some(-1.0), Some(1.0), Some(4.0), Some(9.0)],
                true,
                &parameters,
                &mut cooperation,
            )
            .await
            .unwrap();
        assert_eq!(dynamics.neighborhood, 2);
        let fitness = fitness::Fitness::from_measurements(
            POLICY,
            vec![fitness::Measurement::Real(1.0); 4],
            &mut cooperation,
        )
        .await
        .unwrap();
        let dynamics =
            dynamics::Dynamics::new(vec![0.0; 4], &population, vec![Some(1.0); 4], &parameters)
                .unwrap();
        let mut state = state::State::new(population, fitness, dynamics, &mut cooperation)
            .await
            .unwrap();
        let fitness = fitness::Fitness::from_measurements(
            POLICY,
            vec![fitness::Measurement::Real(1.0); 4],
            &mut cooperation,
        )
        .await
        .unwrap();
        state
            .accept(fitness, &parameters, 1, &mut cooperation)
            .await
            .unwrap();
        assert!(!state.function_small(1, 0.0));
        assert!(state.function_small(1, 1e-6));
        let mut population = population::Population::new(
            POLICY,
            Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap(),
            2,
            1,
        )
        .unwrap();
        let fixed = dynamics::Parameters {
            inertia: [1.0, 1.0],
            neighbors_fraction: 1.0,
            personal_weight: 0.0,
            social_weight: 0.0,
        };
        let mut dynamics =
            dynamics::Dynamics::new(vec![-1.0, -1.0], &population, vec![Some(1.0); 2], &fixed)
                .unwrap();
        dynamics
            .propose(&mut population, &fixed, &[0.0], &[2.0], &mut cooperation)
            .await
            .unwrap();
        assert_eq!(population.get(0, 0), 0.0);
        assert_eq!(
            dynamics.velocity,
            vec![-1.0, -1.0],
            "landing exactly on a bound does not clip/reset"
        );
        dynamics
            .propose(&mut population, &fixed, &[0.0], &[2.0], &mut cooperation)
            .await
            .unwrap();
        assert_eq!(population.get(0, 0), 0.0);
        assert_eq!(
            dynamics.velocity,
            vec![0.0, 0.0],
            "an actual crossing clips and resets outward velocity"
        );
    });
}
