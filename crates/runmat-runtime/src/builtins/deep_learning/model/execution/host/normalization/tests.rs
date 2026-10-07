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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn channel_only_normalization_preserves_rank_one_single_and_exact_admission() {
    let input = Tensor::from_f32(vec![1., 3.], vec![2]).unwrap();
    let scale = Tensor::from_f32(vec![1., 1.], vec![2, 1]).unwrap();
    let offset = Tensor::from_f32(vec![0., 0.], vec![2, 1]).unwrap();
    let admitted = axes("C", input.shape());
    let opts = options(DeepLearningNormalizationAxes::ChannelOnly);
    let output = forward(&input, &scale, &offset, &admitted, opts).unwrap();
    assert_eq!(output.tensor.shape(), &[2]);
    assert_eq!(
        output.tensor.numeric_dtype(),
        runmat_value::NumericDType::F32
    );
    let expected = 1.0_f32 / (1.0_f32 + 1.0e-5_f32).sqrt();
    let actual = output.tensor.as_f32_slice().unwrap();
    assert!((actual[0] + expected).abs() < 1.0e-6);
    assert!((actual[1] - expected).abs() < 1.0e-6);

    let mismatched = Tensor::from_f32(vec![1., 3.], vec![1, 2]).unwrap();
    assert!(matches!(
        forward(&mismatched, &scale, &offset, &admitted, opts),
        Err(NormalizationError::ShapeMismatch)
    ));
    let row_scale = Tensor::from_f32(vec![1., 1.], vec![1, 2]).unwrap();
    assert!(matches!(
        forward(&input, &row_scale, &offset, &admitted, opts),
        Err(NormalizationError::ShapeMismatch)
    ));
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
        &axes("CBT", x.shape()),
        options(DeepLearningNormalizationAxes::Auto),
    )
    .unwrap();
    assert_eq!(y.axes.layout().labels(), "CBT");
    assert_eq!(y.tensor.shape(), &[2, 3, 2]);
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
        &axes("CBT", x.shape()),
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
    let axes = axes("CBT", input.shape());
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
fn batch_excluded_full_output_uses_each_batchs_channels_and_times_only() {
    let input = Tensor::new(
        vec![1., 2., 4., 10., 20., 40., 3., 5., 7., 30., 50., 70.],
        vec![3, 2, 2],
    )
    .unwrap();
    let scale = Tensor::new(vec![1., 2., -1.], vec![3, 1]).unwrap();
    let offset = Tensor::new(vec![0.5, -0.5, 1.], vec![3, 1]).unwrap();
    let axes = axes("CBT", input.shape());
    let actual = forward(
        &input,
        &scale,
        &offset,
        &axes,
        options(DeepLearningNormalizationAxes::BatchExcluded),
    )
    .unwrap()
    .tensor;
    let x = input.as_f64_slice().unwrap();
    let gamma = scale.as_f64_slice().unwrap();
    let beta = offset.as_f64_slice().unwrap();
    for batch in 0..2 {
        let positions: Vec<usize> = (0..2)
            .flat_map(|time| (0..3).map(move |channel| channel + 3 * batch + 6 * time))
            .collect();
        let mean = positions.iter().map(|&i| x[i]).sum::<f64>() / 6.;
        let variance = positions
            .iter()
            .map(|&i| (x[i] - mean).powi(2))
            .sum::<f64>()
            / 6.;
        for index in positions {
            let channel = index % 3;
            let expected =
                (x[index] - mean) / (variance + 1.0e-5).sqrt() * gamma[channel] + beta[channel];
            assert!((actual.as_f64_slice().unwrap()[index] - expected).abs() < 1.0e-12);
        }
    }
}

#[test]
fn spatial_channel_mode_and_native_single_are_distinct_from_channel_only() {
    let input = Tensor::from_f32(vec![1., 3., 5., 7., 2., 4., 6., 8.], vec![2, 2, 1, 2]).unwrap();
    let scale = Tensor::from_f32(vec![1., 1.], vec![1, 2]).unwrap();
    let offset = Tensor::from_f32(vec![0., 0.], vec![1, 2]).unwrap();
    let axes = axes("SCBT", input.shape());
    let y = forward(
        &input,
        &scale,
        &offset,
        &axes,
        options(DeepLearningNormalizationAxes::SpatialChannel),
    )
    .unwrap();
    assert_eq!(y.tensor.shape(), input.shape());
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
    let transposed_scale = Tensor::from_f32(vec![1., 1.], vec![2, 1]).unwrap();
    assert!(matches!(
        forward(
            &input,
            &transposed_scale,
            &offset,
            &axes,
            options(DeepLearningNormalizationAxes::SpatialChannel),
        ),
        Err(NormalizationError::ShapeMismatch)
    ));
}

#[test]
fn invalid_epsilon_empty_reduction_and_dtype_reject() {
    let input = Tensor::new(vec![1., 2.], vec![2, 1]).unwrap();
    let scale = Tensor::new(vec![1., 1.], vec![2, 1]).unwrap();
    let offset = Tensor::new(vec![0., 0.], vec![2, 1]).unwrap();
    let axes = axes("CB", input.shape());
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
            &self::axes("CB", empty.shape()),
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

#[test]
fn extreme_finite_inputs_do_not_overflow_the_mean_or_scaled_variance() {
    {
        let (positive, negative) = (2.0e38_f32, -2.0e38_f32);
        let axes = axes("CB", &[2, 1]);
        let scale = Tensor::from_f32(vec![1., 1.], vec![2, 1]).unwrap();
        let offset = Tensor::from_f32(vec![0.25, -0.5], vec![2, 1]).unwrap();
        let equal = Tensor::from_f32(vec![positive, positive], vec![2, 1]).unwrap();
        let result = forward(
            &equal,
            &scale,
            &offset,
            &axes,
            options(DeepLearningNormalizationAxes::Auto),
        )
        .unwrap();
        assert_eq!(result.tensor.as_f32_slice().unwrap(), &[0.25, -0.5]);
        let opposed = Tensor::from_f32(vec![positive, negative], vec![2, 1]).unwrap();
        let result = forward(
            &opposed,
            &scale,
            &offset,
            &axes,
            options(DeepLearningNormalizationAxes::Auto),
        )
        .unwrap();
        assert_eq!(result.tensor.as_f32_slice().unwrap(), &[1.25, -1.5]);
    }
    let huge = 1.0e308_f64;
    let axes = axes("CB", &[2, 1]);
    let scale = Tensor::new(vec![1., 1.], vec![2, 1]).unwrap();
    let offset = Tensor::new(vec![0., 0.], vec![2, 1]).unwrap();
    let input = Tensor::new(vec![huge, -huge], vec![2, 1]).unwrap();
    let result = forward(
        &input,
        &scale,
        &offset,
        &axes,
        options(DeepLearningNormalizationAxes::Auto),
    )
    .unwrap();
    assert_eq!(result.tensor.as_f64_slice().unwrap(), &[1., -1.]);
}

#[test]
fn spatial_channel_normalization_has_full_nonsquare_output_and_adjoint_oracles() {
    let shape = [2, 3, 2, 2];
    let axes = axes("SCBT", &shape);
    let opts = options(DeepLearningNormalizationAxes::SpatialChannel);
    let x: Vec<f64> = (0..24).map(|i| (i as f64 * 0.37) - 2.4).collect();
    let gamma = vec![1.3, -0.7, 0.5];
    let beta = vec![0.2, 0.4, -0.3];
    let dy: Vec<f64> = (0..24).map(|i| (i as f64 * 0.13) - 1.1).collect();
    let input = Tensor::new(x.clone(), shape.to_vec()).unwrap();
    let scale = Tensor::new(gamma.clone(), vec![1, 3]).unwrap();
    let offset = Tensor::new(beta.clone(), vec![1, 3]).unwrap();
    let output_gradient = Tensor::new(dy.clone(), shape.to_vec()).unwrap();
    let result = forward(&input, &scale, &offset, &axes, opts).unwrap();
    let values = result.tensor.as_f64_slice().unwrap();
    for batch in 0..2 {
        for time in 0..2 {
            let base = 6 * batch + 12 * time;
            let group = &x[base..base + 6];
            let mean = group.iter().sum::<f64>() / 6.;
            let variance = group.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 6.;
            for (index, value) in group.iter().enumerate() {
                let channel = index / 2;
                let expected =
                    (value - mean) / (variance + 1.0e-5).sqrt() * gamma[channel] + beta[channel];
                assert!((values[base + index] - expected).abs() < 1.0e-12);
            }
        }
    }
    let gradients = backward(
        &input,
        &scale,
        &offset,
        &output_gradient,
        &axes,
        opts,
        &result.stats,
    )
    .unwrap();
    let objective = |x: &[f64], gamma: &[f64], beta: &[f64]| -> f64 {
        let input = Tensor::new(x.to_vec(), shape.to_vec()).unwrap();
        let scale = Tensor::new(gamma.to_vec(), vec![1, 3]).unwrap();
        let offset = Tensor::new(beta.to_vec(), vec![1, 3]).unwrap();
        forward(&input, &scale, &offset, &axes, opts)
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
        let expected =
            (objective(&plus, &gamma, &beta) - objective(&minus, &gamma, &beta)) / (2. * h);
        assert!((gradients.input.as_f64_slice().unwrap()[index] - expected).abs() < 1.0e-8);
    }
    for index in 0..gamma.len() {
        let mut plus = gamma.clone();
        plus[index] += h;
        let mut minus = gamma.clone();
        minus[index] -= h;
        let expected = (objective(&x, &plus, &beta) - objective(&x, &minus, &beta)) / (2. * h);
        assert!((gradients.scale.as_f64_slice().unwrap()[index] - expected).abs() < 1.0e-8);
        let mut plus = beta.clone();
        plus[index] += h;
        let mut minus = beta.clone();
        minus[index] -= h;
        let expected = (objective(&x, &gamma, &plus) - objective(&x, &gamma, &minus)) / (2. * h);
        assert!((gradients.offset.as_f64_slice().unwrap()[index] - expected).abs() < 1.0e-8);
    }
}

#[test]
fn empty_batch_preserves_shape_without_touching_reduction_coordinates() {
    let input = Tensor::new(vec![], vec![2, 0]).unwrap();
    let scale = Tensor::new(vec![1., 2.], vec![2, 1]).unwrap();
    let offset = Tensor::new(vec![0., 0.], vec![2, 1]).unwrap();
    let axes = axes("CB", input.shape());
    let opts = options(DeepLearningNormalizationAxes::Auto);
    let result = forward(&input, &scale, &offset, &axes, opts).unwrap();
    assert_eq!(result.tensor.shape(), &[2, 0]);
    let dy = Tensor::new(vec![], vec![2, 0]).unwrap();
    let gradients = backward(&input, &scale, &offset, &dy, &axes, opts, &result.stats).unwrap();
    assert_eq!(gradients.input.shape(), &[2, 0]);
    assert_eq!(gradients.scale.as_f64_slice().unwrap(), &[0., 0.]);
    assert_eq!(gradients.offset.as_f64_slice().unwrap(), &[0., 0.]);
}
