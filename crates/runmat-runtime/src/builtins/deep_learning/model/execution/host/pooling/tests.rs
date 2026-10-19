use runmat_builtins::{DeepLearningAxisLayout, DeepLearningInputKind};
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

#[test]
fn cbt_non_square_pool_has_independent_channel_batch_oracle() {
    let mut values = Vec::new();
    for t in 0..4 {
        for b in 0..3 {
            for c in 0..2 {
                values.push((100 * c + 10 * b + t) as f64);
            }
        }
    }
    let x = Tensor::new(values, vec![2, 3, 4]).unwrap();
    let result = forward(&x, &axes("CBT", x.shape())).unwrap();
    assert_eq!(result.axes.layout().labels(), "CB");
    assert_eq!(result.tensor.shape(), &[2, 3]);
    assert_eq!(
        result.tensor.as_f64_slice().unwrap(),
        &[1.5, 101.5, 11.5, 111.5, 21.5, 121.5]
    );
    let dy = Tensor::new(vec![2., 4., 6., 8., 10., 12.], vec![2, 3]).unwrap();
    let dx = backward(&dy, &axes("CBT", x.shape())).unwrap();
    assert_eq!(dx.shape(), x.shape());
    assert_eq!(
        &dx.as_f64_slice().unwrap()[0..6],
        &[0.5, 1., 1.5, 2., 2.5, 3.]
    );
    assert_eq!(
        &dx.as_f64_slice().unwrap()[18..24],
        &[0.5, 1., 1.5, 2., 2.5, 3.]
    );
}

#[test]
fn spatial_pool_preserves_s_and_time_while_ct_becomes_cu() {
    let scb = Tensor::from_f32(vec![1., 3., 5., 2., 4., 6.], vec![3, 1, 2]).unwrap();
    let result = forward(&scb, &axes("SCB", scb.shape())).unwrap();
    assert_eq!(result.axes.layout().labels(), "SCB");
    assert_eq!(result.tensor.shape(), &[1, 1, 2]);
    assert_eq!(
        result.tensor.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![3., 4.])
    );
    let dy = Tensor::from_f32(vec![6., 9.], vec![1, 1, 2]).unwrap();
    let dx = backward(&dy, &axes("SCB", scb.shape())).unwrap();
    assert_eq!(
        dx.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![2., 2., 2., 3., 3., 3.])
    );

    let scbt = Tensor::from_f32(vec![1., 3., 5., 2., 4., 6.], vec![3, 1, 1, 2]).unwrap();
    let result = forward(&scbt, &axes("SCBT", scbt.shape())).unwrap();
    assert_eq!(result.axes.layout().labels(), "SCBT");
    assert_eq!(result.tensor.shape(), &[1, 1, 1, 2]);
    assert_eq!(
        result.tensor.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![3., 4.])
    );

    let ct = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();
    let result = forward(&ct, &axes("CT", ct.shape())).unwrap();
    assert_eq!(result.axes.layout().labels(), "CU");
    assert_eq!(result.tensor.shape(), &[2, 1]);
    assert_eq!(result.tensor.as_f64_slice().unwrap(), &[3., 4.]);
}

#[test]
fn pool_adjoint_matches_finite_difference_and_rejects_empty_axis() {
    let x = [1., 2., 4., 7., 9., 11.];
    let shape = [2, 1, 3];
    let axes = axes("CBT", &shape);
    let dy = Tensor::new(vec![0.7, -0.4], vec![2, 1]).unwrap();
    let dx = backward(&dy, &axes).unwrap();
    let epsilon = 1.0e-6;
    for index in 0..x.len() {
        let mut plus = x;
        plus[index] += epsilon;
        let mut minus = x;
        minus[index] -= epsilon;
        let objective = |data: [f64; 6]| {
            let tensor = Tensor::new(data.to_vec(), shape.to_vec()).unwrap();
            let y = forward(&tensor, &axes).unwrap().tensor;
            y.as_f64_slice()
                .unwrap()
                .iter()
                .zip(dy.as_f64_slice().unwrap())
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };
        let numerical = (objective(plus) - objective(minus)) / (2. * epsilon);
        assert!((dx.as_f64_slice().unwrap()[index] - numerical).abs() < 1.0e-9);
    }
    let empty = Tensor::new(vec![], vec![2, 1, 0]).unwrap();
    assert!(matches!(
        forward(&empty, &self::axes("CBT", empty.shape())),
        Err(PoolError::EmptyPool)
    ));
}

#[test]
fn documented_no_channel_formats_pool_their_spatial_or_time_axis() {
    for (format, shape, expected_format, expected_shape, expected) in [
        ("SB", vec![3, 2], "SB", vec![1, 2], vec![2., 5.]),
        ("BT", vec![2, 3], "BU", vec![2, 1], vec![3., 4.]),
        (
            "SBT",
            vec![3, 2, 2],
            "SBT",
            vec![1, 2, 2],
            vec![2., 5., 8., 11.],
        ),
    ] {
        let count = shape.iter().product();
        let input = Tensor::new((1..=count).map(|v| v as f64).collect(), shape.clone()).unwrap();
        let layout = DeepLearningAxisLayout::parse(format, shape.len()).unwrap();
        let axes = AxisPlan::new(layout, &shape).unwrap();
        let result = forward(&input, &axes).unwrap();
        assert_eq!(result.axes.layout().labels(), expected_format);
        assert_eq!(result.tensor.shape(), expected_shape.as_slice());
        assert_eq!(result.tensor.as_f64_slice().unwrap(), expected);
        let dy = Tensor::new(vec![1.; expected.len()], expected_shape).unwrap();
        let dx = backward(&dy, &axes).unwrap();
        assert!(dx.as_f64_slice().unwrap().iter().all(|v| *v == 1. / 3.));
    }
}

#[test]
fn finite_extreme_means_remain_finite_without_widening_or_raw_sum_overflow() {
    let axes = axes("CT", &[1, 2]);
    for (input, expected) in [
        (vec![2.0e38_f32, 2.0e38_f32], 2.0e38_f32),
        (vec![2.0e38_f32, -2.0e38_f32], 0.0_f32),
    ] {
        let tensor = Tensor::from_f32(input, vec![1, 2]).unwrap();
        let result = forward(&tensor, &axes).unwrap();
        assert_eq!(result.tensor.as_f32_slice().unwrap(), &[expected]);
    }
    let tensor = Tensor::new(vec![1.0e308, -1.0e308], vec![1, 2]).unwrap();
    assert_eq!(
        forward(&tensor, &axes)
            .unwrap()
            .tensor
            .as_f64_slice()
            .unwrap(),
        &[0.]
    );
}

#[test]
fn nonsquare_spatial_sequence_and_unbatched_adjoint_match_independent_finite_differences() {
    for (format, shape) in [("SCT", vec![3, 2, 4]), ("SCBT", vec![3, 2, 2, 4])] {
        let count: usize = shape.iter().product();
        let x: Vec<f64> = (0..count).map(|index| index as f64 * 0.3 - 2.0).collect();
        let axes = axes(format, &shape);
        let input = Tensor::new(x.clone(), shape.clone()).unwrap();
        let result = forward(&input, &axes).unwrap();
        let output_shape = result.tensor.shape().to_vec();
        let dy_values: Vec<f64> = (0..result.tensor.len())
            .map(|i| i as f64 * 0.2 + 0.5)
            .collect();
        let dy = Tensor::new(dy_values.clone(), output_shape).unwrap();
        let dx = backward(&dy, &axes).unwrap();
        // S is the leading axis in both cases, so column-major groups of
        // three inputs each contribute exactly one third of their output's
        // cotangent. This oracle is independent of the pooling geometry code.
        for (index, &value) in dx.as_f64_slice().unwrap().iter().enumerate() {
            assert_eq!(value, dy_values[index / shape[0]] / shape[0] as f64);
        }
        // The pooled objective is linear in each input. A larger central
        // difference step has no truncation penalty here and avoids losing
        // small perturbations when the full objective is accumulated.
        let h = 1.0e-4;
        for index in 0..count {
            let mut plus = x.clone();
            plus[index] += h;
            let mut minus = x.clone();
            minus[index] -= h;
            let objective = |values: Vec<f64>| -> f64 {
                let tensor = Tensor::new(values, shape.clone()).unwrap();
                forward(&tensor, &axes)
                    .unwrap()
                    .tensor
                    .as_f64_slice()
                    .unwrap()
                    .iter()
                    .zip(&dy_values)
                    .map(|(value, grad)| value * grad)
                    .sum()
            };
            let expected = (objective(plus) - objective(minus)) / (2. * h);
            assert!((dx.as_f64_slice().unwrap()[index] - expected).abs() < 1.0e-9);
        }
    }
}

#[test]
fn empty_batch_retains_zero_elements_without_dividing_by_zero() {
    let input = Tensor::new(vec![], vec![2, 0, 3]).unwrap();
    let axes = axes("CBT", input.shape());
    let result = forward(&input, &axes).unwrap();
    assert_eq!(result.tensor.shape(), &[2, 0]);
    let dy = Tensor::new(vec![], vec![2, 0]).unwrap();
    assert_eq!(backward(&dy, &axes).unwrap().shape(), input.shape());
}
