use runmat_builtins::DeepLearningAxisLayout;
use runmat_value::{NumericDType, NumericStorage, Tensor};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::{backward, forward};

fn axes(labels: &str, shape: &[usize]) -> AxisPlan {
    AxisPlan::new(
        DeepLearningAxisLayout::parse(labels, shape.len()).unwrap(),
        shape,
    )
    .unwrap()
}

#[test]
fn non_square_bc_and_cb_match_independent_linear_oracle() {
    // W = [2 3; 5 7], b = [11;13]; columns are input channels.
    let w = Tensor::new(vec![2., 5., 3., 7.], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![11., 13.], vec![2, 1]).unwrap();
    let bc = Tensor::new(vec![1., 4., 2., 5., 3., 6.], vec![3, 2]).unwrap();
    let cb = Tensor::new(vec![1., 3., 4., 5., 2., 6.], vec![2, 3]).unwrap();
    // BC samples are (1,5), (4,3), (2,6). CB samples are (1,3), (4,5), (2,6).
    let y_bc = forward(&bc, &w, &b, &axes("BC", bc.shape())).unwrap();
    let y_cb = forward(&cb, &w, &b, &axes("CB", cb.shape())).unwrap();
    assert_eq!(y_bc.tensor.shape(), &[3, 2]);
    assert_eq!(
        y_bc.tensor.as_f64_slice().unwrap(),
        &[28., 28., 33., 53., 54., 65.]
    );
    assert_eq!(y_cb.tensor.shape(), &[2, 3]);
    assert_eq!(
        y_cb.tensor.as_f64_slice().unwrap(),
        &[22., 39., 34., 68., 33., 65.]
    );
}

#[test]
fn cbt_f32_forward_backward_are_native_and_nonzero() {
    let x = Tensor::from_f32(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]).unwrap();
    let w = Tensor::from_f32(vec![2., 3.], vec![1, 2]).unwrap();
    let b = Tensor::from_f32(vec![1.], vec![1, 1]).unwrap();
    let plan = axes("CBT", x.shape());
    let result = forward(&x, &w, &b, &plan).unwrap();
    assert_eq!(
        result.tensor.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![9., 19., 29., 39.])
    );
    let dy = Tensor::from_f32(vec![1., 2., 3., 4.], vec![1, 2, 2]).unwrap();
    let gradients = backward(&x, &w, &b, &plan, &dy).unwrap();
    assert_eq!(gradients.input.numeric_dtype(), NumericDType::F32);
    assert_eq!(
        gradients.input.as_f32_slice().unwrap(),
        &[2., 3., 4., 6., 6., 9., 8., 12.]
    );
    assert_eq!(gradients.weights.as_f32_slice().unwrap(), &[50., 60.]);
    assert_eq!(gradients.bias.as_f32_slice().unwrap(), &[10.]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn backward_rejects_equal_length_but_different_output_axes_shape() {
    let input = Tensor::from_f32(vec![1.0; 8], vec![2, 2, 2]).unwrap();
    let weights = Tensor::from_f32(vec![2.0, 3.0], vec![1, 2]).unwrap();
    let bias = Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap();
    let plan = axes("CBT", input.shape());
    let wrong_shape = Tensor::from_f32(vec![1.0; 4], vec![2, 1, 2]).unwrap();
    assert!(matches!(
        backward(&input, &weights, &bias, &plan, &wrong_shape),
        Err(super::geometry::DenseError::Invalid(
            "fullyConnectedLayer output gradient shape or class differs"
        ))
    ));
}

#[test]
fn empty_batch_has_correct_output_and_zero_parameter_gradients() {
    let x = Tensor::new(vec![], vec![2, 0]).unwrap();
    let w = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![5., 6.], vec![2, 1]).unwrap();
    let plan = axes("CB", x.shape());
    let y = forward(&x, &w, &b, &plan).unwrap();
    assert_eq!(y.tensor.shape(), &[2, 0]);
    let dy = Tensor::new(vec![], vec![2, 0]).unwrap();
    let g = backward(&x, &w, &b, &plan, &dy).unwrap();
    assert_eq!(g.weights.as_f64_slice().unwrap(), &[0.; 4]);
    assert_eq!(g.bias.as_f64_slice().unwrap(), &[0.; 2]);
}

#[test]
fn non_square_bc_adjoint_matches_independent_finite_difference() {
    let x = Tensor::new(vec![0.7, -0.2, 1.1, 0.5, -0.9, 0.4], vec![3, 2]).unwrap();
    let w = Tensor::new(vec![0.3, -0.6, 1.4, 0.8], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![0.2, -0.1], vec![2, 1]).unwrap();
    let plan = axes("BC", x.shape());
    let dy = Tensor::new(vec![0.4, -0.7, 1.3, -0.5, 0.9, -1.1], vec![3, 2]).unwrap();
    let analytic = backward(&x, &w, &b, &plan, &dy).unwrap();
    let h = 1e-6;
    let objective = |x: &Tensor, w: &Tensor, b: &Tensor| -> f64 {
        forward(x, w, b, &plan)
            .unwrap()
            .tensor
            .as_f64_slice()
            .unwrap()
            .iter()
            .zip(dy.as_f64_slice().unwrap())
            .map(|(value, weight)| value * weight)
            .sum()
    };
    for (original, gradient, kind) in [
        (
            x.as_f64_slice().unwrap().to_vec(),
            analytic.input.as_f64_slice().unwrap().to_vec(),
            0,
        ),
        (
            w.as_f64_slice().unwrap().to_vec(),
            analytic.weights.as_f64_slice().unwrap().to_vec(),
            1,
        ),
        (
            b.as_f64_slice().unwrap().to_vec(),
            analytic.bias.as_f64_slice().unwrap().to_vec(),
            2,
        ),
    ] {
        for index in 0..original.len() {
            let mut positive = original.clone();
            let mut negative = original.clone();
            positive[index] += h;
            negative[index] -= h;
            let evaluate = |values: Vec<f64>| match kind {
                0 => objective(&Tensor::new(values, x.shape().to_vec()).unwrap(), &w, &b),
                1 => objective(&x, &Tensor::new(values, w.shape().to_vec()).unwrap(), &b),
                _ => objective(&x, &w, &Tensor::new(values, b.shape().to_vec()).unwrap()),
            };
            let numerical = (evaluate(positive) - evaluate(negative)) / (2.0 * h);
            assert!(
                (numerical - gradient[index]).abs() < 1e-8,
                "kind={kind} index={index}"
            );
        }
    }
}
