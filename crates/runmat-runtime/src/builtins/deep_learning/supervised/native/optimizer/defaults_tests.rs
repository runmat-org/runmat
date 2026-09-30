use super::*;
use crate::builtins::deep_learning::{
    layers::{lstm_layer_builtin, sequence_input_layer_builtin},
    model::{
        execution::host::numeric::Real,
        parameters::{self, ParameterRole},
    },
    supervised::native::factors::{FactorPlan, Factors},
    training::training_options_builtin,
};
use futures::executor::block_on;
use runmat_value::Tensor;

fn config_with_decay(override_decay: Option<f64>) -> TrainingConfig {
    let rest = override_decay.map_or_else(Vec::new, |value| {
        vec![
            Value::String("SquaredGradientDecayFactor".into()),
            Value::Num(value),
        ]
    });
    let options = block_on(training_options_builtin(
        Value::String("rmsprop".into()),
        rest,
    ))
    .unwrap();
    TrainingConfig::from_value(&options, "trainnet").unwrap()
}

fn two_steps<T: Real>(config: &TrainingConfig) -> (T, T, T) {
    let factors = Factors::uniform(1.0, 0.0)
        .native::<T>(config.learn_rate, config.l2_regularization, "test")
        .unwrap();
    let mut first = [];
    let mut second = [T::zero()];
    let first_value = step_values(
        &[T::one()],
        &[T::from_f64(2.0)],
        &mut first,
        &mut second,
        config,
        &factors,
        1,
        "test",
    )
    .unwrap()[0];
    let second_value = step_values(
        &[first_value],
        &[T::from_f64(-1.0)],
        &mut first,
        &mut second,
        config,
        &factors,
        2,
        "test",
    )
    .unwrap()[0];
    (first_value, second_value, second[0])
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
    let input = block_on(sequence_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let learn = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).unwrap();
    let l2 = Tensor::new(vec![1.0, 0.0, 2.0, 0.0], vec![1, 4]).unwrap();
    let layer = block_on(lstm_layer_builtin(
        Value::Num(2.0),
        vec![
            Value::String("InputWeightsLearnRateFactor".into()),
            Value::Tensor(learn),
            Value::String("InputWeightsL2Factor".into()),
            Value::Tensor(l2),
        ],
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
    let mut first = vec![T::zero(); 16];
    let mut second = [];
    let updated = step_values(
        &[T::one(); 16],
        &[T::one(); 16],
        &mut first,
        &mut second,
        &config,
        &factors,
        1,
        "test",
    )
    .unwrap();
    // InputWeights is [4 gates × 2 units, 2 input channels], column-major.
    // Each gate factor applies to both hidden units and both columns.
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
