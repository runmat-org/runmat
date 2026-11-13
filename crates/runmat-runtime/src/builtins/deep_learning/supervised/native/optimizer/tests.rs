use super::*;
use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;
use crate::builtins::deep_learning::{
    layers::feature_input_layer_builtin,
    model::execution::host::numeric::Real,
    model::{dlnetwork_builtin, parameters::ParameterRole},
    supervised::{
        native::factors::{FactorPlan, Factors, NativeFactors},
        ShuffleMode, Solver,
    },
};
use futures::executor::block_on;
use runmat_value::CellArray;

pub(super) fn config(solver: Solver) -> TrainingConfig {
    TrainingConfig {
        solver,
        max_epochs: 1,
        mini_batch_size: 1,
        learn_rate: 0.1,
        momentum: 0.9,
        gradient_decay_factor: 0.9,
        squared_gradient_decay_factor: 0.999,
        epsilon: 1.0e-8,
        l2_regularization: 0.0001,
        shuffle: ShuffleMode::Never,
        verbose: false,
    }
}

pub(super) fn native_factors<T: Real>(
    config: &TrainingConfig,
    learning: f64,
    l2: f64,
) -> NativeFactors<T> {
    Factors::uniform(learning, l2)
        .native(config.learn_rate, 1.0, "test")
        .unwrap()
}

fn run<T: Real>(
    current: &[T],
    gradient: &[T],
    old: Option<&SolverSlot<T>>,
    config: &TrainingConfig,
    factors: &NativeFactors<T>,
    iteration: usize,
) -> (Vec<T>, SolverSlot<T>) {
    let controls = NativeControls::admit(config, iteration, "test").unwrap();
    step::step(current, gradient, old, controls, factors, "test").unwrap()
}

#[test]
fn l2_regularization_adds_weight_decay_to_native_sgdm_gradient() {
    let config = config(Solver::Sgdm);
    let (updated, slot) = run(
        &[2.0_f64],
        &[0.0_f64],
        None,
        &config,
        &native_factors(&config, 1.0, 0.5),
        1,
    );
    assert_eq!(updated, vec![1.9]);
    let SolverSlot::Sgdm { velocity } = slot else {
        panic!("SGDM slot")
    };
    assert_eq!(velocity, vec![-0.1]);
}

#[test]
fn rmsprop_updates_from_native_squared_gradient_without_bias_correction() {
    let mut config = config(Solver::Rmsprop);
    config.squared_gradient_decay_factor = 0.0;
    config.epsilon = 1.0;
    let (updated, slot) = run(
        &[2.0_f32],
        &[3.0_f32],
        None,
        &config,
        &native_factors(&config, 1.0, 0.0),
        1,
    );
    let SolverSlot::Rmsprop { squared } = slot else {
        panic!("RMSProp slot")
    };
    assert_eq!(squared, vec![9.0]);
    assert!((updated[0] - 1.925).abs() < 1.0e-6);
}

#[test]
fn native_single_sgdm_and_adam_follow_independent_nonzero_oracles() {
    let value = [2.0_f32];
    let gradient = [3.0_f32];
    let sgdm = config(Solver::Sgdm);
    let (updated, slot) = run(
        &value,
        &gradient,
        None,
        &sgdm,
        &native_factors(&sgdm, 1.0, 0.0),
        1,
    );
    assert!((updated[0] - 1.7).abs() < 1.0e-6);
    let SolverSlot::Sgdm { velocity } = slot else {
        panic!("SGDM slot")
    };
    assert!((velocity[0] + 0.3).abs() < 1.0e-6);

    let adam = config(Solver::Adam);
    let (updated, slot) = run(
        &value,
        &gradient,
        None,
        &adam,
        &native_factors(&adam, 1.0, 0.0),
        1,
    );
    assert!((updated[0] - 1.9).abs() < 1.0e-6);
    let SolverSlot::Adam { mean, squared } = slot else {
        panic!("Adam slot")
    };
    assert!((mean[0] - 0.3).abs() < 1.0e-6);
    assert!((squared[0] - 0.009).abs() < 1.0e-6);

    let (updated, _) = run(
        &value,
        &gradient,
        None,
        &sgdm,
        &native_factors(&sgdm, 2.0, 0.5),
        1,
    );
    assert!((updated[0] - 1.2).abs() < 1.0e-6);
}

#[test]
fn rejected_later_gradient_does_not_advance_earlier_private_slots() {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(2.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.0; 2], vec![1, 2]).unwrap()),
        ],
    ))
    .unwrap();
    let network = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, dense], 2, 1).unwrap(),
    )]))
    .unwrap();
    let Value::Object(network) = network.value() else {
        panic!("network object")
    };
    let Value::Cell(layers) = network.properties().get("Layers").unwrap() else {
        panic!("canonical Layers")
    };
    let factors = FactorPlan::admit(layers.data(), "test").unwrap();
    let rows = parameters::view(network, "test").unwrap();
    let prepared = rows
        .iter()
        .map(|row| {
            let Some(Value::Tensor(value)) = row.value else {
                panic!("native parameter")
            };
            (row.key.clone(), value.clone())
        })
        .collect::<Vec<_>>();
    let gradients = |invalid_bias: bool| {
        rows.iter()
            .map(|row| {
                let Some(Value::Tensor(value)) = row.value else {
                    panic!("native parameter")
                };
                let entries = if invalid_bias && row.key.role == ParameterRole::Bias {
                    vec![f64::INFINITY; value.len()]
                } else {
                    vec![1.0; value.len()]
                };
                (
                    row.key.clone(),
                    Tensor::new(entries, value.shape().to_vec()).unwrap(),
                )
            })
            .collect::<Vec<_>>()
    };
    let mut optimizer = NativeOptimizer::new(factors);
    let config = config(Solver::Sgdm);
    assert_eq!(
        optimizer
            .propose(network, &prepared, gradients(false), &config, 1, "test")
            .unwrap()
            .len(),
        2
    );
    let before = optimizer
        .slots
        .iter()
        .map(|(key, slot)| {
            let Slot::Double(SolverSlot::Sgdm { velocity }) = slot else {
                panic!("double SGDM slot")
            };
            (key.clone(), velocity.clone())
        })
        .collect::<HashMap<_, _>>();
    assert!(optimizer
        .propose(network, &prepared, gradients(true), &config, 2, "test")
        .is_err());
    assert_eq!(optimizer.slots.len(), before.len());
    for (key, slot) in &optimizer.slots {
        let Slot::Double(SolverSlot::Sgdm { velocity }) = slot else {
            panic!("double SGDM slot")
        };
        assert_eq!(velocity, &before[key]);
    }
}

#[path = "defaults_tests.rs"]
mod defaults_tests;
