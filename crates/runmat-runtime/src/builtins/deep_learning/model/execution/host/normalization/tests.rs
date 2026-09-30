use runmat_builtins::{DeepLearningInputKind, DeepLearningNormalizationAxes};
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
    } else if format.contains('T') {
        DeepLearningInputKind::VectorSequence
    } else {
        DeepLearningInputKind::Feature
    };
    AxisPlan::for_input(format, kind, shape).unwrap()
}
fn options(operation: DeepLearningNormalizationAxes) -> NormalizationOptions {
    NormalizationOptions {
        epsilon: 1.0e-5,
        operation,
    }
}

#[test]
fn auto_normalizes_channels_independently_per_batch_and_time() {
    let mut values = Vec::new();
    for time in 0..2 {
        for batch in 0..3 {
            values.push((10 * time + 2 * batch + 1) as f64);
            values.push((10 * time + 2 * batch + 3) as f64);
        }
    }
    let x = Tensor::new(values, vec![2, 3, 2]).unwrap();
    let gamma = Tensor::new(vec![2., -1.], vec![2, 1]).unwrap();
    let beta = Tensor::new(vec![0.5, 1.5], vec![2, 1]).unwrap();
    let y = forward(
        &x,
        &gamma,
        &beta,
        &axes("CBT", &x.shape),
        options(DeepLearningNormalizationAxes::Auto),
    )
    .unwrap();
    assert_eq!(y.axes.layout().labels(), "CBT");
    assert_eq!(y.tensor.shape, vec![2, 3, 2]);
    for chunk in y.tensor.as_f64_slice().unwrap().chunks_exact(2) {
        assert!((chunk[0] + 1.5).abs() < 1.0e-4);
        assert!((chunk[1] - 0.5).abs() < 1.0e-4);
    }
}

fn objective(x: &[f64], gamma: &[f64], beta: &[f64], dy: &[f64]) -> f64 {
    let x = Tensor::new(x.to_vec(), vec![3, 2, 2]).unwrap();
    let gamma = Tensor::new(gamma.to_vec(), vec![3, 1]).unwrap();
    let beta = Tensor::new(beta.to_vec(), vec![3, 1]).unwrap();
    let y = forward(
        &x,
        &gamma,
        &beta,
        &axes("CBT", &x.shape),
        options(DeepLearningNormalizationAxes::BatchExcluded),
    )
    .unwrap()
    .tensor;
    y.as_f64_slice()
        .unwrap()
        .iter()
        .zip(dy)
        .map(|(a, b)| a * b)
        .sum()
}

#[test]
fn all_norm_adjoint_components_match_finite_differences() {
    let x = [
        0.2, 1.1, -0.7, 2.0, 0.3, 1.4, 0.8, -0.2, 1.7, -1.3, 0.4, 2.1,
    ];
    let gamma = [1.2, -0.6, 0.9];
    let beta = [0.3, 0.5, -0.2];
    let dy = [
        0.4, 0.7, -0.3, 0.5, -0.6, 0.8, 0.1, -0.9, 0.2, 0.3, 0.4, -0.7,
    ];
    let input = Tensor::new(x.to_vec(), vec![3, 2, 2]).unwrap();
    let scale = Tensor::new(gamma.to_vec(), vec![3, 1]).unwrap();
    let offset = Tensor::new(beta.to_vec(), vec![3, 1]).unwrap();
    let gradient = Tensor::new(dy.to_vec(), vec![3, 2, 2]).unwrap();
    let axes = axes("CBT", &input.shape);
    let opts = options(DeepLearningNormalizationAxes::BatchExcluded);
    let forward_result = forward(&input, &scale, &offset, &axes, opts).unwrap();
    let result = backward(
        &input,
        &scale,
        &offset,
        &gradient,
        &axes,
        opts,
        &forward_result.stats,
    )
    .unwrap();
    let epsilon = 1.0e-6;
    for index in 0..x.len() {
        let mut plus = x;
        plus[index] += epsilon;
        let mut minus = x;
        minus[index] -= epsilon;
        let numerical = (objective(&plus, &gamma, &beta, &dy)
            - objective(&minus, &gamma, &beta, &dy))
            / (2. * epsilon);
        assert!((result.input.as_f64_slice().unwrap()[index] - numerical).abs() < 1.0e-8);
    }
    for index in 0..3 {
        let mut plus = gamma;
        plus[index] += epsilon;
        let mut minus = gamma;
        minus[index] -= epsilon;
        let numerical =
            (objective(&x, &plus, &beta, &dy) - objective(&x, &minus, &beta, &dy)) / (2. * epsilon);
        assert!((result.scale.as_f64_slice().unwrap()[index] - numerical).abs() < 1.0e-8);
        let mut plus = beta;
        plus[index] += epsilon;
        let mut minus = beta;
        minus[index] -= epsilon;
        let numerical = (objective(&x, &gamma, &plus, &dy) - objective(&x, &gamma, &minus, &dy))
            / (2. * epsilon);
        assert!((result.offset.as_f64_slice().unwrap()[index] - numerical).abs() < 1.0e-8);
    }
}

#[test]
fn spatial_channel_mode_and_native_single_are_distinct_from_channel_only() {
    let input = Tensor::from_f32(vec![1., 3., 5., 7., 2., 4., 6., 8.], vec![2, 2, 1, 2]).unwrap();
    let scale = Tensor::from_f32(vec![1., 1.], vec![1, 2]).unwrap();
    let offset = Tensor::from_f32(vec![0., 0.], vec![1, 2]).unwrap();
    let axes = axes("SCBT", &input.shape);
    let y = forward(
        &input,
        &scale,
        &offset,
        &axes,
        options(DeepLearningNormalizationAxes::SpatialChannel),
    )
    .unwrap();
    assert_eq!(y.tensor.shape, input.shape);
    let NumericStorage::F32(values) = y.tensor.into_numeric_storage().unwrap() else {
        panic!("single was widened")
    };
    assert!((values[0] + 1.34164).abs() < 0.0001);
    let y = forward(
        &input,
        &scale,
        &offset,
        &axes,
        options(DeepLearningNormalizationAxes::ChannelOnly),
    )
    .unwrap();
    let NumericStorage::F32(values) = y.tensor.into_numeric_storage().unwrap() else {
        panic!("single was widened")
    };
    assert!((values[0] + 1.0).abs() < 0.0001);
}

#[test]
fn invalid_epsilon_empty_reduction_and_dtype_reject() {
    let input = Tensor::new(vec![1., 2.], vec![2, 1]).unwrap();
    let scale = Tensor::new(vec![1., 1.], vec![2, 1]).unwrap();
    let offset = Tensor::new(vec![0., 0.], vec![2, 1]).unwrap();
    let axes = axes("CB", &input.shape);
    assert!(matches!(
        forward(
            &input,
            &scale,
            &offset,
            &axes,
            NormalizationOptions {
                epsilon: 0.,
                operation: DeepLearningNormalizationAxes::Auto
            }
        ),
        Err(NormalizationError::InvalidEpsilon)
    ));
    let single_scale = Tensor::from_f32(vec![1., 1.], vec![2, 1]).unwrap();
    assert!(matches!(
        forward(
            &input,
            &single_scale,
            &offset,
            &axes,
            options(DeepLearningNormalizationAxes::Auto)
        ),
        Err(NormalizationError::UnsupportedPrecision)
    ));
    let empty = Tensor::new(vec![], vec![0, 1]).unwrap();
    assert!(matches!(
        forward(
            &empty,
            &Tensor::new(vec![], vec![0, 1]).unwrap(),
            &Tensor::new(vec![], vec![0, 1]).unwrap(),
            &axes("CB", &empty.shape),
            options(DeepLearningNormalizationAxes::Auto)
        ),
        Err(NormalizationError::EmptyReduction)
    ));
}

#[test]
fn zero_variance_is_finite_and_f32_adjoint_matches_finite_difference() {
    let axes = axes("CB", &[3, 1]);
    let scale = Tensor::from_f32(vec![1.2, 0.7, -0.3], vec![3, 1]).unwrap();
    let offset = Tensor::from_f32(vec![0.5, -1.0, 2.0], vec![3, 1]).unwrap();
    let equal = Tensor::from_f32(vec![3.0, 3.0, 3.0], vec![3, 1]).unwrap();
    let opts = options(DeepLearningNormalizationAxes::Auto);
    let result = forward(&equal, &scale, &offset, &axes, opts).unwrap();
    assert_eq!(
        result.tensor.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![0.5, -1.0, 2.0])
    );

    let original = [1.0_f32, 2.0, 4.0];
    let input = Tensor::from_f32(original.to_vec(), vec![3, 1]).unwrap();
    let dy = Tensor::from_f32(vec![0.4, -0.2, 0.8], vec![3, 1]).unwrap();
    let result = forward(&input, &scale, &offset, &axes, opts).unwrap();
    let gradient = backward(&input, &scale, &offset, &dy, &axes, opts, &result.stats).unwrap();
    let epsilon = 0.001_f32;
    let objective = |values: [f32; 3]| -> f32 {
        let x = Tensor::from_f32(values.to_vec(), vec![3, 1]).unwrap();
        let y = forward(&x, &scale, &offset, &axes, opts).unwrap().tensor;
        y.as_f32_slice()
            .unwrap()
            .iter()
            .zip(dy.as_f32_slice().unwrap())
            .map(|(a, b)| a * b)
            .sum()
    };
    for index in 0..3 {
        let mut plus = original;
        plus[index] += epsilon;
        let mut minus = original;
        minus[index] -= epsilon;
        let numerical = (objective(plus) - objective(minus)) / (2.0 * epsilon);
        assert!((gradient.input.as_f32_slice().unwrap()[index] - numerical).abs() < 0.001);
    }
}
