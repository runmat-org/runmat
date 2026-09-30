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
    let result = forward(&x, &axes("CBT", &x.shape)).unwrap();
    assert_eq!(result.axes.layout().labels(), "CB");
    assert_eq!(result.tensor.shape, vec![2, 3]);
    assert_eq!(
        result.tensor.as_f64_slice().unwrap(),
        &[1.5, 101.5, 11.5, 111.5, 21.5, 121.5]
    );
    let dy = Tensor::new(vec![2., 4., 6., 8., 10., 12.], vec![2, 3]).unwrap();
    let dx = backward(&dy, &axes("CBT", &x.shape)).unwrap();
    assert_eq!(dx.shape, x.shape);
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
    let result = forward(&scb, &axes("SCB", &scb.shape)).unwrap();
    assert_eq!(result.axes.layout().labels(), "SCB");
    assert_eq!(result.tensor.shape, vec![1, 1, 2]);
    assert_eq!(
        result.tensor.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![3., 4.])
    );
    let dy = Tensor::from_f32(vec![6., 9.], vec![1, 1, 2]).unwrap();
    let dx = backward(&dy, &axes("SCB", &scb.shape)).unwrap();
    assert_eq!(
        dx.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![2., 2., 2., 3., 3., 3.])
    );

    let scbt = Tensor::from_f32(vec![1., 3., 5., 2., 4., 6.], vec![3, 1, 1, 2]).unwrap();
    let result = forward(&scbt, &axes("SCBT", &scbt.shape)).unwrap();
    assert_eq!(result.axes.layout().labels(), "SCBT");
    assert_eq!(result.tensor.shape, vec![1, 1, 1, 2]);
    assert_eq!(
        result.tensor.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![3., 4.])
    );

    let ct = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();
    let result = forward(&ct, &axes("CT", &ct.shape)).unwrap();
    assert_eq!(result.axes.layout().labels(), "CU");
    assert_eq!(result.tensor.shape, vec![2, 1]);
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
        forward(&empty, &axes("CBT", &empty.shape)),
        Err(PoolError::EmptyPool)
    ));
}
