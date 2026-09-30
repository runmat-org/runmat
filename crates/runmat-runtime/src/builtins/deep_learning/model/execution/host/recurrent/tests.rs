use runmat_builtins::DeepLearningAxisLayout;
use runmat_value::{NumericDType, NumericStorage, Tensor};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::*;

fn tensor(data: Vec<f64>, shape: &[usize]) -> Tensor {
    Tensor::from_numeric_storage(NumericStorage::F64(data), shape.to_vec()).unwrap()
}

fn axis(format: &str, shape: &[usize]) -> AxisPlan {
    AxisPlan::new(
        DeepLearningAxisLayout::parse(format, shape.len()).unwrap(),
        shape,
    )
    .unwrap()
}

fn options(bidirectional: bool, output_mode: OutputMode) -> RecurrentOptions {
    RecurrentOptions {
        hidden_size: 1,
        bidirectional,
        output_mode,
        state_activation: Activation::Tanh,
        gate_activation: GateActivation::Sigmoid,
    }
}

fn run(
    x: &Tensor,
    w: &Tensor,
    r: &Tensor,
    b: &Tensor,
    h0: Option<&Tensor>,
    c0: Option<&Tensor>,
    axes: &AxisPlan,
    options: RecurrentOptions,
    capture: bool,
) -> RecurrentOutput {
    forward(x, w, r, b, h0, c0, axes, options, capture).unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() < tolerance,
        "{actual} != {expected}"
    );
}

#[test]
fn two_step_hand_computed_gate_order_and_state() {
    let x = tensor(vec![7.0, -3.0], &[1, 1, 2]);
    let w = tensor(vec![0.0; 4], &[4, 1]);
    let r = tensor(vec![0.0; 4], &[4, 1]);
    let b = tensor(vec![0.0, 0.0, 0.2_f64.atanh(), 0.0], &[4, 1]);
    let axes = axis("CBT", &x.shape);
    let result = run(
        &x,
        &w,
        &r,
        &b,
        None,
        None,
        &axes,
        options(false, OutputMode::Sequence),
        true,
    );
    let y = result.tensor.as_f64_slice().unwrap();
    close(y[0], 0.5 * 0.1_f64.tanh(), 1e-12);
    close(y[1], 0.5 * 0.15_f64.tanh(), 1e-12);
    close(result.cell_state.as_f64_slice().unwrap()[0], 0.15, 1e-12);
    assert_eq!(result.axes.layout().labels(), "CBT");
}

#[test]
fn bidirectional_terminal_states_are_not_the_same_as_y_at_last_time() {
    let x = tensor(vec![0.25, 1.0, 2.0], &[1, 1, 3]);
    let mut weights = vec![0.0; 8];
    weights[2] = 1.0; // Forward candidate.
    weights[6] = 2.0; // Backward candidate differs deliberately.
    let w = tensor(weights, &[8, 1]);
    let r = tensor(vec![0.0; 8], &[8, 1]);
    let b = tensor(vec![0.0; 8], &[8, 1]);
    let axes = axis("CBT", &x.shape);
    let sequence = run(
        &x,
        &w,
        &r,
        &b,
        None,
        None,
        &axes,
        options(true, OutputMode::Sequence),
        false,
    );
    let last = run(
        &x,
        &w,
        &r,
        &b,
        None,
        None,
        &axes,
        options(true, OutputMode::Last),
        false,
    );
    let full = sequence.tensor.as_f64_slice().unwrap();
    let terminal = last.tensor.as_f64_slice().unwrap();
    close(terminal[0], full[4], 1e-12); // Forward t=last.
    close(terminal[1], full[1], 1e-12); // Backward t=first.
    assert!((terminal[1] - full[5]).abs() > 1e-4);
    assert_eq!(last.axes.layout().labels(), "CB");
    assert_eq!(last.hidden_state.shape, [2, 1]);
}

#[test]
fn explicit_tbc_and_cbt_with_nonsquare_geometry_agree() {
    let mut cbt = vec![0.0; 2 * 3 * 4];
    let mut tbc = vec![0.0; cbt.len()];
    for time in 0..4 {
        for batch in 0..3 {
            for channel in 0..2 {
                let value = (100 * time + 10 * batch + channel) as f64 / 17.0;
                cbt[channel + 2 * batch + 6 * time] = value;
                tbc[time + 4 * batch + 12 * channel] = value;
            }
        }
    }
    let x_cbt = tensor(cbt, &[2, 3, 4]);
    let x_tbc = tensor(tbc, &[4, 3, 2]);
    let w = tensor(vec![0.0, 0.0, 0.2, 0.0, 0.0, 0.0, -0.3, 0.0], &[4, 2]);
    let r = tensor(vec![0.0; 4], &[4, 1]);
    let b = tensor(vec![0.0; 4], &[4, 1]);
    let first = run(
        &x_cbt,
        &w,
        &r,
        &b,
        None,
        None,
        &axis("CBT", &x_cbt.shape),
        options(false, OutputMode::Sequence),
        false,
    );
    let second = run(
        &x_tbc,
        &w,
        &r,
        &b,
        None,
        None,
        &axis("TBC", &x_tbc.shape),
        options(false, OutputMode::Sequence),
        false,
    );
    assert_eq!(first.tensor.as_f64_slice(), second.tensor.as_f64_slice());
    assert_eq!(first.tensor.shape, [1, 3, 4]);
}

#[test]
fn backward_matches_independent_finite_differences_for_input_weights_and_state() {
    let x = tensor(vec![0.4, -0.7], &[1, 1, 2]);
    let w = tensor(vec![0.1, -0.2, 0.3, 0.4], &[4, 1]);
    let r = tensor(vec![0.2, -0.1, 0.05, 0.07], &[4, 1]);
    let b = tensor(vec![0.03, -0.04, 0.05, 0.06], &[4, 1]);
    let h0 = tensor(vec![0.13], &[1, 1]);
    let c0 = tensor(vec![-0.17], &[1, 1]);
    let dy = tensor(vec![0.7, -0.2], &[1, 1, 2]);
    let axes = axis("CBT", &x.shape);
    let opts = options(false, OutputMode::Sequence);
    let output = run(&x, &w, &r, &b, Some(&h0), Some(&c0), &axes, opts, true);
    let grads = backward(
        &x,
        &w,
        &r,
        &b,
        Some(&h0),
        Some(&c0),
        &axes,
        opts,
        &output,
        &dy,
        None,
        None,
    )
    .unwrap();
    let loss = |x: &Tensor, w: &Tensor, r: &Tensor, b: &Tensor, h0: &Tensor, c0: &Tensor| {
        let output = run(x, w, r, b, Some(h0), Some(c0), &axes, opts, false);
        output
            .tensor
            .as_f64_slice()
            .unwrap()
            .iter()
            .zip(dy.as_f64_slice().unwrap())
            .map(|(value, gradient)| value * gradient)
            .sum::<f64>()
    };
    let epsilon = 1e-6;
    let derivative = |base: &Tensor, index: usize, rebuild: &dyn Fn(Tensor) -> f64| {
        let mut plus = base.as_f64_slice().unwrap().to_vec();
        let mut minus = plus.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let plus = tensor(plus, &base.shape);
        let minus = tensor(minus, &base.shape);
        (rebuild(plus) - rebuild(minus)) / (2.0 * epsilon)
    };
    for index in 0..2 {
        let expected = derivative(&x, index, &|candidate| {
            loss(&candidate, &w, &r, &b, &h0, &c0)
        });
        close(grads.input.as_f64_slice().unwrap()[index], expected, 1e-8);
    }
    for index in 0..4 {
        let expected = derivative(&w, index, &|candidate| {
            loss(&x, &candidate, &r, &b, &h0, &c0)
        });
        close(
            grads.input_weights.as_f64_slice().unwrap()[index],
            expected,
            1e-8,
        );
        let expected = derivative(&r, index, &|candidate| {
            loss(&x, &w, &candidate, &b, &h0, &c0)
        });
        close(
            grads.recurrent_weights.as_f64_slice().unwrap()[index],
            expected,
            1e-8,
        );
        let expected = derivative(&b, index, &|candidate| {
            loss(&x, &w, &r, &candidate, &h0, &c0)
        });
        close(grads.bias.as_f64_slice().unwrap()[index], expected, 1e-8);
    }
    close(
        grads.initial_hidden.as_f64_slice().unwrap()[0],
        derivative(&h0, 0, &|candidate| loss(&x, &w, &r, &b, &candidate, &c0)),
        1e-8,
    );
    close(
        grads.initial_cell.as_f64_slice().unwrap()[0],
        derivative(&c0, 0, &|candidate| loss(&x, &w, &r, &b, &h0, &candidate)),
        1e-8,
    );
}

#[test]
fn single_precision_and_malformed_state_rejection() {
    let x =
        Tensor::from_numeric_storage(NumericStorage::F32(vec![1.0, 2.0]), vec![1, 1, 2]).unwrap();
    let w = Tensor::from_numeric_storage(NumericStorage::F32(vec![0.0; 4]), vec![4, 1]).unwrap();
    let r = Tensor::from_numeric_storage(NumericStorage::F32(vec![0.0; 4]), vec![4, 1]).unwrap();
    let b = Tensor::from_numeric_storage(NumericStorage::F32(vec![0.0; 4]), vec![4, 1]).unwrap();
    let axes = axis("CBT", &x.shape);
    let result = run(
        &x,
        &w,
        &r,
        &b,
        None,
        None,
        &axes,
        options(false, OutputMode::Last),
        true,
    );
    assert_eq!(result.tensor.numeric_dtype(), NumericDType::F32);
    assert_eq!(result.hidden_state.numeric_dtype(), NumericDType::F32);
    let bad_state =
        Tensor::from_numeric_storage(NumericStorage::F32(vec![0.0; 2]), vec![2, 1]).unwrap();
    assert!(forward(
        &x,
        &w,
        &r,
        &b,
        Some(&bad_state),
        Some(&bad_state),
        &axes,
        options(false, OutputMode::Last),
        false
    )
    .is_err());
}

#[test]
fn spatial_channels_flatten_by_coordinate_without_relabeling_input() {
    // S=2, C=2, B=2, T=2; first parameter column sees the first spatial
    // coordinate, then the second spatial coordinate, then the next channel.
    let mut values = vec![0.0; 16];
    values[1] = 1.0;
    let x = tensor(values, &[2, 2, 2, 2]);
    let mut weights = vec![0.0; 16];
    weights[2 + 4] = 1.0;
    let w = tensor(weights, &[4, 4]);
    let r = tensor(vec![0.0; 4], &[4, 1]);
    let b = tensor(vec![0.0; 4], &[4, 1]);
    let result = run(
        &x,
        &w,
        &r,
        &b,
        None,
        None,
        &axis("SCBT", &x.shape),
        options(false, OutputMode::Sequence),
        false,
    );
    assert_eq!(result.tensor.shape, [1, 2, 2]);
    assert_eq!(result.axes.layout().labels(), "CBT");
    let output = result.tensor.as_f64_slice().unwrap();
    assert!(output[0] > 0.0);
    close(output[1], 0.0, 1e-12);
}

#[test]
fn broadcast_initial_state_accumulates_gradients_across_batches() {
    let x = tensor(vec![0.0, 0.0], &[1, 2, 1]);
    let w = tensor(vec![0.0; 4], &[4, 1]);
    let r = tensor(vec![0.0, 0.0, 0.2, 0.0], &[4, 1]);
    let b = tensor(vec![0.0; 4], &[4, 1]);
    let h0 = tensor(vec![0.3], &[1, 1]);
    let c0 = tensor(vec![0.1], &[1, 1]);
    let dy = tensor(vec![1.0, 1.0], &[1, 2, 1]);
    let axes = axis("CBT", &x.shape);
    let opts = options(false, OutputMode::Sequence);
    let output = run(&x, &w, &r, &b, Some(&h0), Some(&c0), &axes, opts, true);
    let grads = backward(
        &x,
        &w,
        &r,
        &b,
        Some(&h0),
        Some(&c0),
        &axes,
        opts,
        &output,
        &dy,
        None,
        None,
    )
    .unwrap();
    let dh = grads.initial_hidden.as_f64_slice().unwrap()[0];
    let dc = grads.initial_cell.as_f64_slice().unwrap()[0];
    assert!(dh.abs() > 0.0);
    assert!(dc.abs() > 0.0);
    assert_eq!(grads.initial_hidden.shape, [1, 1]);
    assert_eq!(grads.initial_cell.shape, [1, 1]);
}

#[test]
fn admitted_alternative_activations_have_checked_backward_rule() {
    let x = tensor(vec![0.4, -0.1], &[1, 1, 2]);
    let w = tensor(vec![0.2, -0.3, 0.7, 0.1], &[4, 1]);
    let r = tensor(vec![0.0; 4], &[4, 1]);
    let b = tensor(vec![0.1, -0.2, 0.3, 0.4], &[4, 1]);
    let axes = axis("CBT", &x.shape);
    let mut opts = options(false, OutputMode::Last);
    opts.state_activation = Activation::Softsign;
    opts.gate_activation = GateActivation::HardSigmoid;
    let output = run(&x, &w, &r, &b, None, None, &axes, opts, true);
    let dy = tensor(vec![1.0], &[1, 1]);
    let grads = backward(
        &x, &w, &r, &b, None, None, &axes, opts, &output, &dy, None, None,
    )
    .unwrap();
    let epsilon = 1e-6;
    for index in 0..2 {
        let mut plus = x.as_f64_slice().unwrap().to_vec();
        let mut minus = plus.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let plus = tensor(plus, &x.shape);
        let minus = tensor(minus, &x.shape);
        let y_plus = run(&plus, &w, &r, &b, None, None, &axes, opts, false);
        let y_minus = run(&minus, &w, &r, &b, None, None, &axes, opts, false);
        let finite = (y_plus.tensor.as_f64_slice().unwrap()[0]
            - y_minus.tensor.as_f64_slice().unwrap()[0])
            / (2.0 * epsilon);
        close(grads.input.as_f64_slice().unwrap()[index], finite, 1e-8);
    }
}

#[test]
fn terminal_state_adjoint_works_without_output_adjoint() {
    let x = tensor(vec![0.3, -0.8], &[1, 1, 2]);
    let w = tensor(vec![0.1, 0.2, -0.3, 0.4], &[4, 1]);
    let r = tensor(vec![0.05, -0.06, 0.07, 0.08], &[4, 1]);
    let b = tensor(vec![0.0; 4], &[4, 1]);
    let axes = axis("CBT", &x.shape);
    let opts = options(false, OutputMode::Sequence);
    let output = run(&x, &w, &r, &b, None, None, &axes, opts, true);
    let dy = tensor(vec![0.0, 0.0], &[1, 1, 2]);
    let dh_terminal = tensor(vec![0.7], &[1, 1]);
    let dc_terminal = tensor(vec![-0.2], &[1, 1]);
    let grads = backward(
        &x,
        &w,
        &r,
        &b,
        None,
        None,
        &axes,
        opts,
        &output,
        &dy,
        Some(&dh_terminal),
        Some(&dc_terminal),
    )
    .unwrap();
    let epsilon = 1e-6;
    let mut plus = x.as_f64_slice().unwrap().to_vec();
    let mut minus = plus.clone();
    plus[0] += epsilon;
    minus[0] -= epsilon;
    let loss = |data: Vec<f64>| {
        let x = tensor(data, &x.shape);
        let output = run(&x, &w, &r, &b, None, None, &axes, opts, false);
        0.7 * output.hidden_state.as_f64_slice().unwrap()[0]
            - 0.2 * output.cell_state.as_f64_slice().unwrap()[0]
    };
    close(
        grads.input.as_f64_slice().unwrap()[0],
        (loss(plus) - loss(minus)) / (2.0 * epsilon),
        1e-8,
    );
}

#[test]
fn backward_direction_and_last_summary_have_their_own_adjoint() {
    let x = tensor(vec![0.2, 0.8, -0.5], &[1, 1, 3]);
    let mut weights = vec![0.0; 8];
    weights[2] = 0.7;
    weights[6] = -0.9;
    let w = tensor(weights, &[8, 1]);
    let r = tensor(vec![0.0; 8], &[8, 1]);
    let b = tensor(vec![0.0; 8], &[8, 1]);
    let axes = axis("CBT", &x.shape);
    let opts = options(true, OutputMode::Last);
    let output = run(&x, &w, &r, &b, None, None, &axes, opts, true);
    let dy = tensor(vec![0.3, 0.7], &[2, 1]);
    let grads = backward(
        &x, &w, &r, &b, None, None, &axes, opts, &output, &dy, None, None,
    )
    .unwrap();
    let epsilon = 1e-6;
    for index in 0..3 {
        let mut plus = x.as_f64_slice().unwrap().to_vec();
        let mut minus = plus.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let loss = |values: Vec<f64>| {
            let input = tensor(values, &x.shape);
            let result = run(&input, &w, &r, &b, None, None, &axes, opts, false);
            let y = result.tensor.as_f64_slice().unwrap();
            0.3 * y[0] + 0.7 * y[1]
        };
        close(
            grads.input.as_f64_slice().unwrap()[index],
            (loss(plus) - loss(minus)) / (2.0 * epsilon),
            1e-8,
        );
    }
}
