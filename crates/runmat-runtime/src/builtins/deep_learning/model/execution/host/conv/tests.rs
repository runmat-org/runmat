use runmat_builtins::DeepLearningInputKind;
use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::*;

fn axes(format: &str, shape: &[usize]) -> AxisPlan {
    let kind = if format.contains('S') {
        if format.contains('T') {
            DeepLearningInputKind::SpatialSequence1d
        } else {
            DeepLearningInputKind::Spatial1d
        }
    } else {
        DeepLearningInputKind::VectorSequence
    };
    AxisPlan::for_input(format, kind, shape).unwrap()
}

fn valid() -> ConvOptions {
    ConvOptions {
        stride: 1,
        dilation: 1,
        padding: Padding::Valid,
        padding_value: PaddingValue::Constant(0.0),
    }
}

fn values(tensor: &Tensor) -> Vec<f64> {
    match tensor.numeric_dtype() {
        runmat_value::NumericDType::F32 => tensor
            .as_f32_slice()
            .unwrap()
            .iter()
            .map(|v| f64::from(*v))
            .collect(),
        runmat_value::NumericDType::F64 => tensor.as_f64_slice().unwrap().to_vec(),
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn multichannel_cbt_and_scb_have_independent_nonzero_oracles() {
    let weights = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2, 1]).unwrap();
    let bias = Tensor::new(vec![5.], vec![1, 1]).unwrap();
    let cbt = Tensor::new(
        vec![1., 10., 4., 40., 2., 20., 5., 50., 3., 30., 6., 60.],
        vec![2, 2, 3],
    )
    .unwrap();
    let result = forward(&cbt, &weights, &bias, &axes("CBT", cbt.shape()), valid()).unwrap();
    assert_eq!(result.axes.layout().labels(), "CBT");
    assert_eq!(result.tensor.shape(), &[1, 2, 2]);
    assert_eq!(values(&result.tensor), vec![120., 339., 193., 412.]);

    let scb = Tensor::new(
        vec![1., 2., 3., 10., 20., 30., 4., 5., 6., 40., 50., 60.],
        vec![3, 2, 2],
    )
    .unwrap();
    let result = forward(&scb, &weights, &bias, &axes("SCB", scb.shape()), valid()).unwrap();
    assert_eq!(result.axes.layout().labels(), "SCB");
    assert_eq!(result.tensor.shape(), &[2, 1, 2]);
    assert_eq!(values(&result.tensor), vec![120., 193., 339., 412.]);
}

#[test]
fn scbt_keeps_time_independent_and_single_precision_native() {
    let input = Tensor::from_f32(vec![1., 2., 3., 2., 4., 6.], vec![3, 1, 1, 2]).unwrap();
    let weights = Tensor::from_f32(vec![2., 3.], vec![2, 1, 1]).unwrap();
    let bias = Tensor::from_f32(vec![1.], vec![1, 1]).unwrap();
    let result = forward(
        &input,
        &weights,
        &bias,
        &axes("SCBT", input.shape()),
        valid(),
    )
    .unwrap();
    assert_eq!(result.axes.layout().labels(), "SCBT");
    assert_eq!(result.tensor.shape(), &[2, 1, 1, 2]);
    assert_eq!(
        result.tensor.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![9., 14., 17., 27.])
    );
}

#[test]
fn stride_dilation_padding_and_boundary_modes_are_numerical() {
    let input = Tensor::new(vec![1., 2., 3., 4.], vec![1, 1, 4]).unwrap();
    let weights = Tensor::new(vec![2., 3.], vec![2, 1, 1]).unwrap();
    let bias = Tensor::new(vec![0.], vec![1, 1]).unwrap();
    let axes = axes("CBT", input.shape());
    let mut options = valid();
    options.stride = 2;
    options.dilation = 2;
    options.padding = Padding::Same;
    assert_eq!(
        values(
            &forward(&input, &weights, &bias, &axes, options)
                .unwrap()
                .tensor
        ),
        vec![11., 6.]
    );
    options.padding = Padding::Causal;
    assert_eq!(
        values(
            &forward(&input, &weights, &bias, &axes, options)
                .unwrap()
                .tensor
        ),
        vec![3., 11.]
    );
    options.padding = Padding::Explicit { left: 1, right: 0 };
    options.stride = 1;
    options.dilation = 1;
    options.padding_value = PaddingValue::Replicate;
    assert_eq!(
        values(
            &forward(&input, &weights, &bias, &axes, options)
                .unwrap()
                .tensor
        ),
        vec![5., 8., 13., 18.]
    );
    options.padding_value = PaddingValue::SymmetricExcludeEdge;
    assert_eq!(
        values(
            &forward(&input, &weights, &bias, &axes, options)
                .unwrap()
                .tensor
        ),
        vec![7., 8., 13., 18.]
    );
}

#[test]
fn resident_descriptor_uses_the_exact_host_admitted_padding_and_axes() {
    use runmat_accelerate_api::DeepLearningConv1DPaddingValue as PhysicalPadding;

    let axes = axes("SCBT", &[4, 2, 3, 2]);
    let options = ConvOptions {
        stride: 2,
        dilation: 2,
        padding: Padding::Same,
        padding_value: PaddingValue::SymmetricIncludeEdge,
    };
    let physical = resident_request(
        &axes,
        &[2, 2, 5],
        &[1, 5],
        runmat_value::NumericDType::F32,
        options,
    )
    .unwrap();
    assert_eq!(physical.input_shape, vec![4, 2, 3, 2]);
    assert_eq!(physical.output_shape, vec![2, 5, 3, 2]);
    assert_eq!(physical.weights_shape, [2, 2, 5]);
    assert_eq!(physical.bias_shape, [1, 5]);
    assert_eq!((physical.span_axis, physical.channel_axis), (0, 1));
    assert_eq!(
        (physical.batch_axis, physical.time_axis),
        (Some(2), Some(3))
    );
    assert_eq!((physical.stride, physical.dilation), (2, 2));
    assert_eq!((physical.padding_left, physical.padding_right), (0, 1));
    assert_eq!(
        physical.padding_value,
        PhysicalPadding::SymmetricIncludeEdge
    );

    let causal = resident_request(
        &axes,
        &[2, 2, 5],
        &[1, 5],
        runmat_value::NumericDType::F64,
        ConvOptions {
            padding: Padding::Causal,
            ..options
        },
    )
    .unwrap();
    assert_eq!((causal.padding_left, causal.padding_right), (2, 0));
    assert_eq!(causal.output_shape, vec![2, 5, 3, 2]);
}

fn objective(x: &[f64], w: &[f64], b: f64, dy: &[f64]) -> f64 {
    let input = Tensor::new(x.to_vec(), vec![1, 1, 3]).unwrap();
    let weights = Tensor::new(w.to_vec(), vec![2, 1, 1]).unwrap();
    let bias = Tensor::new(vec![b], vec![1, 1]).unwrap();
    let y = forward(
        &input,
        &weights,
        &bias,
        &axes("CBT", input.shape()),
        valid(),
    )
    .unwrap()
    .tensor;
    values(&y).iter().zip(dy).map(|(y, g)| y * g).sum()
}

#[test]
fn all_adjoint_components_match_independent_finite_differences() {
    let x = [1.2, -0.7, 2.5];
    let w = [0.3, -1.1];
    let b = 0.4;
    let dy = [0.8, -0.2];
    let input = Tensor::new(x.to_vec(), vec![1, 1, 3]).unwrap();
    let weights = Tensor::new(w.to_vec(), vec![2, 1, 1]).unwrap();
    let bias = Tensor::new(vec![b], vec![1, 1]).unwrap();
    let gradient = Tensor::new(dy.to_vec(), vec![1, 1, 2]).unwrap();
    let result = backward(
        &input,
        &weights,
        &bias,
        &gradient,
        &axes("CBT", input.shape()),
        valid(),
    )
    .unwrap();
    let epsilon = 1.0e-6;
    for index in 0..x.len() {
        let mut plus = x;
        plus[index] += epsilon;
        let mut minus = x;
        minus[index] -= epsilon;
        let numerical =
            (objective(&plus, &w, b, &dy) - objective(&minus, &w, b, &dy)) / (2. * epsilon);
        assert!((values(&result.input)[index] - numerical).abs() < 1.0e-8);
    }
    for index in 0..w.len() {
        let mut plus = w;
        plus[index] += epsilon;
        let mut minus = w;
        minus[index] -= epsilon;
        let numerical =
            (objective(&x, &plus, b, &dy) - objective(&x, &minus, b, &dy)) / (2. * epsilon);
        assert!((values(&result.weights)[index] - numerical).abs() < 1.0e-8);
    }
    let numerical = (objective(&x, &w, b + epsilon, &dy) - objective(&x, &w, b - epsilon, &dy))
        / (2. * epsilon);
    assert!((values(&result.bias)[0] - numerical).abs() < 1.0e-8);
}

#[test]
fn f32_constant_padding_adjoint_has_exact_nonzero_components() {
    let x = Tensor::from_f32(vec![1., 2.], vec![1, 1, 2]).unwrap();
    let w = Tensor::from_f32(vec![4., 5.], vec![2, 1, 1]).unwrap();
    let b = Tensor::from_f32(vec![0.], vec![1, 1]).unwrap();
    let dy = Tensor::from_f32(vec![2., 3.], vec![1, 1, 2]).unwrap();
    let options = ConvOptions {
        padding: Padding::Explicit { left: 1, right: 0 },
        padding_value: PaddingValue::Constant(3.),
        ..valid()
    };
    let axes = axes("CBT", x.shape());
    assert_eq!(
        values(&forward(&x, &w, &b, &axes, options).unwrap().tensor),
        vec![17., 14.]
    );
    let gradients = backward(&x, &w, &b, &dy, &axes, options).unwrap();
    assert_eq!(
        gradients.input.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![22., 15.])
    );
    assert_eq!(
        gradients.weights.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![9., 8.])
    );
    assert_eq!(
        gradients.bias.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![5.])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn backward_rejects_equal_length_but_different_convolution_output_shape() {
    let input = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 1, 3]).unwrap();
    let weights = Tensor::from_f32(vec![4.0, 5.0], vec![2, 1, 1]).unwrap();
    let bias = Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap();
    let wrong_shape = Tensor::from_f32(vec![1.0, 1.0], vec![2, 1, 1]).unwrap();
    let plan = axes("CBT", input.shape());
    assert!(matches!(
        backward(&input, &weights, &bias, &wrong_shape, &plan, valid()),
        Err(ConvError::Invalid(
            "output gradient shape must equal convolution output shape"
        ))
    ));
}

#[test]
fn empty_overflow_and_precision_mismatch_reject_before_allocation() {
    let x = Tensor::new(vec![], vec![1, 1, 0]).unwrap();
    let w = Tensor::new(vec![1.], vec![1, 1, 1]).unwrap();
    let b = Tensor::new(vec![0.], vec![1, 1]).unwrap();
    assert!(matches!(
        forward(&x, &w, &b, &axes("CBT", x.shape()), valid()),
        Err(ConvError::Invalid(_))
    ));
    let x = Tensor::new(vec![1.], vec![1, 1, 1]).unwrap();
    let single_w = Tensor::from_f32(vec![1.], vec![1, 1, 1]).unwrap();
    assert!(matches!(
        forward(&x, &single_w, &b, &axes("CBT", x.shape()), valid()),
        Err(ConvError::UnsupportedPrecision)
    ));
    let mut options = valid();
    options.padding = Padding::Explicit {
        left: usize::MAX,
        right: 1,
    };
    assert!(matches!(
        forward(&x, &w, &b, &axes("CBT", x.shape()), options),
        Err(ConvError::ShapeOverflow)
    ));
}

#[test]
fn multichannel_spatial_sequence_stride_dilation_and_repeated_padding_adjoints() {
    let cases = [
        (
            "SCBT",
            vec![3, 2, 2, 2],
            ConvOptions {
                stride: 1,
                dilation: 1,
                padding: Padding::Explicit { left: 2, right: 2 },
                padding_value: PaddingValue::Replicate,
            },
        ),
        (
            "SCBT",
            vec![3, 2, 2, 2],
            ConvOptions {
                stride: 1,
                dilation: 1,
                padding: Padding::Explicit { left: 2, right: 2 },
                padding_value: PaddingValue::SymmetricIncludeEdge,
            },
        ),
        (
            "SCBT",
            vec![3, 2, 2, 2],
            ConvOptions {
                stride: 1,
                dilation: 1,
                padding: Padding::Explicit { left: 2, right: 2 },
                padding_value: PaddingValue::SymmetricExcludeEdge,
            },
        ),
        (
            "CBT",
            vec![2, 2, 5],
            ConvOptions {
                stride: 2,
                dilation: 2,
                padding: Padding::Same,
                padding_value: PaddingValue::Constant(0.7),
            },
        ),
    ];
    for (format, shape, options) in cases {
        let x: Vec<f64> = (0..shape.iter().product::<usize>())
            .map(|i| (i as f64 * 0.23) - 1.4)
            .collect();
        let w: Vec<f64> = (0..12).map(|i| (i as f64 * 0.17) - 0.8).collect();
        let b = vec![0.3, -0.4];
        let axes = axes(format, &shape);
        let input = Tensor::new(x.clone(), shape.clone()).unwrap();
        let weights = Tensor::new(w.clone(), vec![3, 2, 2]).unwrap();
        let bias = Tensor::new(b.clone(), vec![1, 2]).unwrap();
        let output = forward(&input, &weights, &bias, &axes, options).unwrap();
        let output_shape = output.tensor.shape().to_vec();
        let dy: Vec<f64> = (0..output.tensor.len())
            .map(|i| (i as f64 * 0.11) - 0.7)
            .collect();
        let gradient = Tensor::new(dy.clone(), output_shape).unwrap();
        let adjoints = backward(&input, &weights, &bias, &gradient, &axes, options).unwrap();
        let objective = |x: &[f64], w: &[f64], b: &[f64]| -> f64 {
            let input = Tensor::new(x.to_vec(), shape.clone()).unwrap();
            let weights = Tensor::new(w.to_vec(), vec![3, 2, 2]).unwrap();
            let bias = Tensor::new(b.to_vec(), vec![1, 2]).unwrap();
            forward(&input, &weights, &bias, &axes, options)
                .unwrap()
                .tensor
                .as_f64_slice()
                .unwrap()
                .iter()
                .zip(&dy)
                .map(|(y, grad)| y * grad)
                .sum()
        };
        let h = 1.0e-6;
        for index in 0..x.len() {
            let mut plus = x.clone();
            plus[index] += h;
            let mut minus = x.clone();
            minus[index] -= h;
            let expected = (objective(&plus, &w, &b) - objective(&minus, &w, &b)) / (2. * h);
            assert!(
                (adjoints.input.as_f64_slice().unwrap()[index] - expected).abs() < 1.0e-7,
                "{format} input index {index}"
            );
        }
        for index in 0..w.len() {
            let mut plus = w.clone();
            plus[index] += h;
            let mut minus = w.clone();
            minus[index] -= h;
            let expected = (objective(&x, &plus, &b) - objective(&x, &minus, &b)) / (2. * h);
            assert!(
                (adjoints.weights.as_f64_slice().unwrap()[index] - expected).abs() < 1.0e-7,
                "{format} weight index {index}"
            );
        }
        for index in 0..b.len() {
            let mut plus = b.clone();
            plus[index] += h;
            let mut minus = b.clone();
            minus[index] -= h;
            let expected = (objective(&x, &w, &plus) - objective(&x, &w, &minus)) / (2. * h);
            assert!(
                (adjoints.bias.as_f64_slice().unwrap()[index] - expected).abs() < 1.0e-7,
                "{format} bias index {index}"
            );
        }
    }
}

#[test]
fn empty_batch_retains_zero_output_and_zero_parameter_gradients() {
    let input = Tensor::new(vec![], vec![3, 2, 0]).unwrap();
    let weights = Tensor::new(vec![1.; 12], vec![3, 2, 2]).unwrap();
    let bias = Tensor::new(vec![0., 0.], vec![1, 2]).unwrap();
    let axes = axes("SCB", input.shape());
    let result = forward(&input, &weights, &bias, &axes, valid()).unwrap();
    assert_eq!(result.tensor.shape(), &[1, 2, 0]);
    let gradient = Tensor::new(vec![], result.tensor.shape().to_vec()).unwrap();
    let adjoints = backward(&input, &weights, &bias, &gradient, &axes, valid()).unwrap();
    assert_eq!(adjoints.input.shape(), input.shape());
    assert!(adjoints
        .weights
        .as_f64_slice()
        .unwrap()
        .iter()
        .all(|value| *value == 0.));
    assert_eq!(adjoints.bias.as_f64_slice().unwrap(), &[0., 0.]);
}
