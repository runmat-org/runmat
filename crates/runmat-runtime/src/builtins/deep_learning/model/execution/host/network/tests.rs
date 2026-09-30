use runmat_builtins::DeepLearningAxisLayout;
use runmat_value::{NumericDType, Tensor};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::{backward, execute, HostStep, HostValue, StepGradients};

fn axes(labels: &str, shape: &[usize]) -> AxisPlan {
    AxisPlan::new(
        DeepLearningAxisLayout::parse(labels, shape.len()).unwrap(),
        shape,
    )
    .unwrap()
}

#[test]
fn cb_sequence_preserves_axes_and_backpropagates_through_shared_kernels() {
    let input = Tensor::from_f32(vec![1., -2., 3., -4., 5., -6.], vec![2, 1, 3]).unwrap();
    let weights = Tensor::from_f32(vec![2., 0., 0., 3.], vec![2, 2]).unwrap();
    let bias = Tensor::from_f32(vec![1., 1.], vec![2, 1]).unwrap();
    let steps = [
        HostStep::Input,
        HostStep::FullyConnected {
            weights: &weights,
            bias: &bias,
        },
        HostStep::Relu,
        HostStep::RegressionOutput,
    ];
    let result = execute(
        &steps,
        HostValue::Real(input),
        axes("CBT", &[2, 1, 3]),
        true,
    )
    .unwrap();
    assert_eq!(result.axes.layout().labels(), "CBT");
    assert!(result.next_state.is_empty());
    let HostValue::Real(output) = &result.value else {
        panic!("real output expected")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.as_f32_slice().unwrap(), &[3., 0., 7., 0., 11., 0.]);
    let gradient = HostValue::Real(Tensor::from_f32(vec![1.; 6], vec![2, 1, 3]).unwrap());
    let reverse = backward(result.trace.as_ref().unwrap(), gradient, &[]).unwrap();
    let HostValue::Real(dx) = reverse.input_gradient else {
        panic!("real gradient expected")
    };
    assert_eq!(dx.as_f32_slice().unwrap(), &[2., 0., 2., 0., 2., 0.]);
    assert_eq!(reverse.parameter_gradients.len(), 2);
    let (
        index,
        StepGradients::FullyConnected {
            weights: dw,
            bias: db,
        },
    ) = &reverse.parameter_gradients[0]
    else {
        panic!("dense gradient expected")
    };
    assert_eq!(*index, 1);
    assert_eq!(dw.as_f32_slice().unwrap(), &[9., 0., -12., 0.]);
    assert_eq!(db.as_f32_slice().unwrap(), &[3., 0.]);
}

#[test]
fn reverse_uses_forward_parameter_snapshot_after_caller_drops_steps() {
    let input = Tensor::new(vec![1., 2.], vec![2, 1]).unwrap();
    let result = {
        let weights = Tensor::new(vec![2., 0., 0., 3.], vec![2, 2]).unwrap();
        let bias = Tensor::new(vec![0., 0.], vec![2, 1]).unwrap();
        let steps = [
            HostStep::Input,
            HostStep::FullyConnected {
                weights: &weights,
                bias: &bias,
            },
        ];
        execute(&steps, HostValue::Real(input), axes("CB", &[2, 1]), true).unwrap()
    };
    let _unrelated_weights = Tensor::new(vec![99., 0., 0., 99.], vec![2, 2]).unwrap();
    let gradient = Tensor::new(vec![1., 1.], vec![2, 1]).unwrap();
    let reverse = backward(
        result.trace.as_ref().unwrap(),
        HostValue::Real(gradient.clone()),
        &[],
    )
    .unwrap();
    let second = backward(
        result.trace.as_ref().unwrap(),
        HostValue::Real(gradient),
        &[],
    )
    .unwrap();
    let HostValue::Real(dx) = reverse.input_gradient else {
        panic!("real gradient expected")
    };
    assert_eq!(dx.as_f64_slice().unwrap(), &[2., 3.]);
    let HostValue::Real(second_dx) = second.input_gradient else {
        panic!("real gradient expected")
    };
    assert_eq!(second_dx.as_f64_slice().unwrap(), &[2., 3.]);
}

#[test]
fn identity_only_route_rejects_mismatched_input_axes() {
    let input = Tensor::new(vec![1., 2.], vec![2, 1]).unwrap();
    let wrong = axes("BC", &[1, 2]);
    assert!(execute(&[HostStep::Input], HostValue::Real(input), wrong, false).is_err());
}
