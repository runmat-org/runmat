use super::*;
use crate::builtins::deep_learning::{
    layers::{lstm_layer_builtin, sequence_input_layer_builtin},
    model::parameters::{self, ParameterRole},
    supervised::native::factors::FactorPlan,
    training::training_options_builtin,
};
use futures::executor::block_on;
use runmat_value::{Tensor, Value};

fn config_with_decay(override_decay: Option<f64>) -> TrainingConfig {
    let rest = override_decay.map_or_else(Vec::new, |value| {
        vec![
            Value::String("SquaredGradientDecayFactor".into()),
            Value::Num(value),
        ]
    });
    let options = block_on(training_options_builtin(
        Value::String("rmsprop".into()),
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
    .unwrap();
    TrainingConfig::from_value(&options, "trainnet").unwrap()
}

fn two_steps<T: Real>(config: &TrainingConfig) -> (T, T, T) {
    let factors = native_factors::<T>(config, 1.0, 0.0);
    let (first, slot) = run(&[T::one()], &[T::from_f64(2.0)], None, config, &factors, 1);
    let (second, slot) = run(
        &first,
        &[T::from_f64(-1.0)],
        Some(&slot),
        config,
        &factors,
        2,
    );
    let SolverSlot::Rmsprop { squared } = slot else {
        panic!("RMSProp slot")
    };
    (first[0], second[0], squared[0])
}

fn assert_default_oracle<T: Real>() {
    let default = config_with_decay(None);
    assert_eq!(default.solver, Solver::Rmsprop);
    assert_eq!(default.squared_gradient_decay_factor, 0.9);
    assert_eq!(default.learn_rate, 0.001);
    let (first, second, square) = two_steps::<T>(&default);
    // Independent recurrence: s1=.9*0+.1*2^2=.4;
    // s2=.9*.4+.1*(-1)^2=.46. RMSProp has no bias correction.
    let expected_first = 1.0 - 0.002 / (0.4_f64.sqrt() + 1.0e-8);
    let expected_second = expected_first + 0.001 / (0.46_f64.sqrt() + 1.0e-8);
    let tolerance = T::from_f64(2.0e-6);
    assert!((first - T::from_f64(expected_first)).abs() < tolerance);
    assert!((second - T::from_f64(expected_second)).abs() < tolerance);
    assert!((square - T::from_f64(0.46)).abs() < tolerance);

    let explicit = config_with_decay(Some(0.999));
    let (explicit_first, explicit_second, explicit_square) = two_steps::<T>(&explicit);
    assert!((explicit_square - T::from_f64(0.004996)).abs() < tolerance);
    assert!(explicit_first < first);
    assert!(explicit_second < second);
}

#[test]
fn rmsprop_default_and_explicit_decay_have_distinct_native_single_two_step_oracles() {
    assert_default_oracle::<f32>();
}

#[test]
fn rmsprop_default_and_explicit_decay_have_distinct_native_double_two_step_oracles() {
    assert_default_oracle::<f64>();
}

fn recurrent_step_oracle<T: Real>() {
    let input = block_on(sequence_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let layer = block_on(lstm_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("InputWeightsLearnRateFactor".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).unwrap()),
            Value::String("InputWeightsL2Factor".into()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0, 2.0, 0.0], vec![1, 4]).unwrap()),
        ]),
    ))
    .unwrap();
    let layers = vec![input, layer];
    let plan = FactorPlan::admit(&layers, "test").unwrap();
    let key = parameters::view_layers(&layers, "test")
        .unwrap()
        .into_iter()
        .find(|row| row.key.role == ParameterRole::InputWeights)
        .unwrap()
        .key;
    let factors = plan
        .get(&key, "test")
        .unwrap()
        .native::<T>(0.1, 0.5, "test")
        .unwrap();
    let mut config = config_with_decay(None);
    config.solver = Solver::Sgdm;
    config.momentum = 0.0;
    let (updated, slot) = run(&[T::one(); 16], &[T::one(); 16], None, &config, &factors, 1);
    assert!(matches!(slot, SolverSlot::Sgdm { .. }));
    // InputWeights is [4 gates × 2 units, 2 input channels], column-major.
    let expected = [1.0, 0.9, 0.6, 0.7];
    for column in 0..2 {
        for (gate, expected) in expected.into_iter().enumerate() {
            for unit in 0..2 {
                let index = column * 8 + gate * 2 + unit;
                assert!((updated[index] - T::from_f64(expected)).abs() < T::from_f64(2.0e-6));
            }
        }
    }
}

#[test]
fn lstm_gate_factors_change_native_single_parameter_step_by_gate() {
    recurrent_step_oracle::<f32>();
}

#[test]
fn lstm_gate_factors_change_native_double_parameter_step_by_gate() {
    recurrent_step_oracle::<f64>();
}

fn near_one_coefficients_oracle<T: Real>() {
    let beta = 0.999_999_999;
    let mut rms = config(Solver::Rmsprop);
    rms.squared_gradient_decay_factor = beta;
    let factors = native_factors::<T>(&rms, 1.0, 0.0);
    let (_, slot) = run(&[T::one()], &[T::from_f64(2.0)], None, &rms, &factors, 1);
    let SolverSlot::Rmsprop { squared } = slot else {
        panic!("RMSProp slot")
    };
    let expected_squared = 4.0 * (1.0 - beta);
    assert!(squared[0] > T::zero());
    assert!(
        (squared[0] - T::from_f64(expected_squared)).abs() < T::from_f64(expected_squared * 1.0e-5)
    );

    let mut adam = config(Solver::Adam);
    adam.gradient_decay_factor = beta;
    adam.squared_gradient_decay_factor = beta;
    let factors = native_factors::<T>(&adam, 1.0, 0.0);
    let (updated, slot) = run(&[T::one()], &[T::from_f64(2.0)], None, &adam, &factors, 1);
    let SolverSlot::Adam { mean, squared } = slot else {
        panic!("Adam slot")
    };
    assert!(
        (mean[0] - T::from_f64(2.0 * (1.0 - beta))).abs()
            < T::from_f64(2.0 * (1.0 - beta) * 1.0e-5)
    );
    assert!(
        (squared[0] - T::from_f64(expected_squared)).abs() < T::from_f64(expected_squared * 1.0e-5)
    );
    assert!((updated[0] - T::from_f64(0.9)).abs() < T::from_f64(2.0e-6));
}

#[test]
fn near_one_solver_controls_preserve_native_single_complements() {
    near_one_coefficients_oracle::<f32>();
}

#[test]
fn near_one_solver_controls_preserve_native_double_complements() {
    near_one_coefficients_oracle::<f64>();
}
