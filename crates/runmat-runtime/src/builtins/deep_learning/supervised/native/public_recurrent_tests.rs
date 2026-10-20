//! Public one-step recurrent training oracles, independent of optimizer internals.

use futures::executor::block_on;
use runmat_value::{CellArray, NumericDType, ObjectInstance, Tensor, Value};

use crate::builtins::deep_learning::{
    layers::{bilstm_layer_builtin, lstm_layer_builtin, sequence_input_layer_builtin},
    model::{
        dlnetwork_builtin,
        parameters::{self, ParameterRole},
        state_storage_property,
    },
    supervised::{train_network_builtin, trainnet_builtin},
    training::{dlarray_builtin, training_options_builtin},
};

fn native(values: Vec<f64>, shape: Vec<usize>, single: bool) -> Tensor {
    if single {
        Tensor::from_f32(
            values.into_iter().map(|value| value as f32).collect(),
            shape,
        )
        .unwrap()
    } else {
        Tensor::new(values, shape).unwrap()
    }
}

fn parameter(network: &ObjectInstance, role: ParameterRole) -> &Tensor {
    let row = parameters::view(network, "test")
        .unwrap()
        .into_iter()
        .find(|row| row.key.role == role)
        .unwrap();
    let Some(Value::Tensor(value)) = row.value else {
        panic!("expected native recurrent parameter")
    };
    value
}

fn candidate_output(gate_bias: f64) -> f64 {
    // With all other parameters and initial State zero, input/output gates
    // are 1/2 and one recurrent step produces this scalar hidden output.
    0.5 * (0.5 * gate_bias.tanh()).tanh()
}

fn candidate_gradient(channels: usize, step: f64) -> f64 {
    let loss = |bias: f64| (candidate_output(bias) - 1.0).powi(2) / channels as f64;
    (loss(step) - loss(-step)) / (2.0 * step)
}

fn one_public_step(single: bool, bidirectional: bool, trainnet: bool) {
    let units = 2;
    let directions = if bidirectional { 2 } else { 1 };
    let rows = 4 * units * directions;
    let channels = units * directions;
    let rate = 0.1;
    let mut gate_factors = vec![0.0; 4 * directions];
    gate_factors[2] = 2.0;
    if bidirectional {
        gate_factors[6] = 3.0;
    }
    let recurrent_options = vec![
        Value::String("OutputMode".into()),
        Value::String("last".into()),
        Value::String("InputWeights".into()),
        Value::Tensor(native(vec![0.0; rows * 2], vec![rows, 2], single)),
        Value::String("RecurrentWeights".into()),
        Value::Tensor(native(vec![0.0; rows * units], vec![rows, units], single)),
        Value::String("Bias".into()),
        Value::Tensor(native(vec![0.0; rows], vec![rows, 1], single)),
        Value::String("InputWeightsLearnRateFactor".into()),
        Value::Num(0.0),
        Value::String("RecurrentWeightsLearnRateFactor".into()),
        Value::Num(0.0),
        Value::String("BiasLearnRateFactor".into()),
        Value::Tensor(Tensor::new(gate_factors.clone(), vec![1, gate_factors.len()]).unwrap()),
    ];
    let input_layer = block_on(sequence_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let recurrent = if bidirectional {
        block_on(bilstm_layer_builtin(
            Value::Num(units as f64),
            crate::builtins::common::test_support::builtin_arguments(recurrent_options),
        ))
        .unwrap()
    } else {
        block_on(lstm_layer_builtin(
            Value::Num(units as f64),
            crate::builtins::common::test_support::builtin_arguments(recurrent_options),
        ))
        .unwrap()
    };
    let network = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input_layer, recurrent], 2, 1).unwrap(),
    )]))
    .unwrap();
    let Value::Object(initial) = network.value() else {
        panic!("expected dlnetwork")
    };
    let initial_state = initial
        .properties()
        .get(state_storage_property())
        .unwrap()
        .clone();
    let input = Value::Tensor(native(
        vec![0.2, -0.3, 0.5, 0.7, -0.4, 0.9],
        vec![2, 3, 1],
        single,
    ));
    let predictor = block_on(dlarray_builtin(
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CBT".into())]),
    ))
    .unwrap();
    // Unformatted regression targets are observation-major; the trainer
    // projects this geometry into the formatted network output axes.
    let target = Value::Tensor(native(vec![1.0; channels * 3], vec![3, channels], single));
    let options = block_on(training_options_builtin(
        Value::String("sgdm".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxEpochs".into()),
            Value::Num(1.0),
            Value::String("MiniBatchSize".into()),
            Value::Num(3.0),
            Value::String("InitialLearnRate".into()),
            Value::Num(rate),
            Value::String("Momentum".into()),
            Value::Num(0.0),
            Value::String("Shuffle".into()),
            Value::String("never".into()),
        ]),
    ))
    .unwrap();
    let trained = if trainnet {
        block_on(trainnet_builtin(vec![
            predictor,
            target,
            network.value().clone(),
            Value::String("mse".into()),
            options,
        ]))
    } else {
        block_on(train_network_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                predictor,
                target,
                network.value().clone(),
                options,
            ]),
        ))
    }
    .unwrap();
    let Value::Object(trained) = trained.value() else {
        panic!("expected trained network")
    };
    assert_eq!(
        trained.properties().get(state_storage_property()),
        Some(&initial_state)
    );
    let Some(Value::Struct(info)) = trained.properties().get("TrainingInfo") else {
        panic!("expected public training information")
    };
    assert_eq!(info.fields.get("Iteration"), Some(&Value::Num(1.0)));
    let Some(Value::Tensor(history)) = info.fields.get("TrainingLoss") else {
        panic!("expected one-step loss history")
    };
    assert!((history.materialize_f64()[0] - 1.0).abs() < 1.0e-6);
    let expected_dtype = if single {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    for role in [ParameterRole::InputWeights, ParameterRole::RecurrentWeights] {
        let value = parameter(trained, role);
        assert_eq!(value.numeric_dtype(), expected_dtype);
        assert!(value.materialize_f64().iter().all(|entry| *entry == 0.0));
    }
    let bias = parameter(trained, ParameterRole::Bias);
    assert_eq!(bias.numeric_dtype(), expected_dtype);
    assert_eq!(bias.shape(), &[rows, 1]);
    let actual = bias.materialize_f64();
    let finite_difference = candidate_gradient(channels, if single { 0.002 } else { 1.0e-5 });
    let tolerance = if single { 2.0e-5 } else { 2.0e-9 };
    assert!((finite_difference + 0.5 / channels as f64).abs() < tolerance);
    for (index, value) in actual.iter().enumerate() {
        let gate = index / units;
        let expected = -rate * gate_factors[gate] * finite_difference;
        assert!(
            (value - expected).abs() < tolerance,
            "{expected_dtype:?} {directions} directions, row {index}: {value} != {expected}"
        );
    }
}

#[test]
fn public_lstm_single_trainnet_applies_candidate_gate_factor() {
    one_public_step(true, false, true);
}

#[test]
fn public_lstm_double_train_network_applies_candidate_gate_factor() {
    one_public_step(false, false, false);
}

#[test]
fn public_bilstm_single_train_network_applies_directional_gate_factors() {
    one_public_step(true, true, false);
}

#[test]
fn public_bilstm_double_trainnet_applies_directional_gate_factors() {
    one_public_step(false, true, true);
}

#[test]
fn public_lstm_single_train_network_applies_candidate_gate_factor() {
    one_public_step(true, false, false);
}

#[test]
fn public_lstm_double_trainnet_applies_candidate_gate_factor() {
    one_public_step(false, false, true);
}

#[test]
fn public_bilstm_single_trainnet_applies_directional_gate_factors() {
    one_public_step(true, true, true);
}

#[test]
fn public_bilstm_double_train_network_applies_directional_gate_factors() {
    one_public_step(false, true, false);
}
