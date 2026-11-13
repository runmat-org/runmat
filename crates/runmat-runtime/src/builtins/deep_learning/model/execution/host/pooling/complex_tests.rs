use runmat_builtins::{DeepLearningAxisLayout, DeepLearningInputKind};
use runmat_value::{
    ComplexElement, ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericDType,
};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::*;

fn axes(format: &str, shape: &[usize]) -> AxisPlan {
    let layout = DeepLearningAxisLayout::parse(format, shape.len()).unwrap();
    AxisPlan::new(layout, shape).unwrap()
}

#[test]
fn complex_spatial_sequence_forward_has_independent_full_output_oracle() {
    let shape = vec![3, 2, 2, 2];
    let values: Vec<(f64, f64)> = (0..24)
        .map(|i| (i as f64 * 0.4 - 3.0, 7.0 - i as f64 * 0.3))
        .collect();
    let input = ComplexTensor::new(values.clone(), shape.clone()).unwrap();
    let result = forward_complex(&input, &axes("SCBT", &shape)).unwrap();
    assert_eq!(result.axes.layout().labels(), "SCBT");
    assert_eq!(result.tensor.shape(), &[1, 2, 2, 2]);
    for (out_index, &ComplexElement(real, imaginary)) in
        result.tensor.as_f64_slice().unwrap().iter().enumerate()
    {
        let base = out_index * 3;
        let expected_real = (0..3).map(|j| values[base + j].0).sum::<f64>() / 3.;
        let expected_imaginary = (0..3).map(|j| values[base + j].1).sum::<f64>() / 3.;
        assert!((real - expected_real).abs() < 1.0e-13);
        assert!((imaginary - expected_imaginary).abs() < 1.0e-13);
    }
}

#[test]
fn real_linear_complex_adjoint_matches_both_component_finite_differences() {
    // L = Re(sum(conj(dY) .* Y)); this tests the host linear map's real and
    // imaginary adjoint. Public tape conventions require separate integration.
    let shape = vec![3, 2, 2, 2];
    let x: Vec<(f64, f64)> = (0..24)
        .map(|i| (i as f64 * 0.2 - 1.4, i as f64 * -0.3 + 2.1))
        .collect();
    let input = ComplexTensor::new(x.clone(), shape.clone()).unwrap();
    let axes = axes("SCBT", &shape);
    let result = forward_complex(&input, &axes).unwrap();
    let dy: Vec<(f64, f64)> = (0..result.tensor.len())
        .map(|i| (i as f64 * 0.17 + 0.4, i as f64 * -0.11 - 0.2))
        .collect();
    let gradient = ComplexTensor::new(dy.clone(), result.tensor.shape().to_vec()).unwrap();
    let dx = backward_complex(&input, &gradient, &axes).unwrap();
    let objective = |values: Vec<(f64, f64)>| -> f64 {
        let tensor = ComplexTensor::new(values, shape.clone()).unwrap();
        forward_complex(&tensor, &axes)
            .unwrap()
            .tensor
            .as_f64_slice()
            .unwrap()
            .iter()
            .zip(&dy)
            .map(|(value, gradient)| value.0 * gradient.0 + value.1 * gradient.1)
            .sum()
    };
    let h = 1.0e-6;
    for index in 0..x.len() {
        let mut plus = x.clone();
        plus[index].0 += h;
        let mut minus = x.clone();
        minus[index].0 -= h;
        let expected_real = (objective(plus) - objective(minus)) / (2. * h);
        let mut plus = x.clone();
        plus[index].1 += h;
        let mut minus = x.clone();
        minus[index].1 -= h;
        let expected_imag = (objective(plus) - objective(minus)) / (2. * h);
        let actual = dx.as_f64_slice().unwrap()[index];
        assert!((actual.0 - expected_real).abs() < 1.0e-8);
        assert!((actual.1 - expected_imag).abs() < 1.0e-8);
    }
}

#[test]
fn single_complex_extreme_components_stay_native_and_finite() {
    let input =
        ComplexTensor::from_f32(vec![(2.0e38, 2.0e38), (2.0e38, -2.0e38)], vec![1, 2]).unwrap();
    let axes = axes("CT", input.shape());
    let result = forward_complex(&input, &axes).unwrap();
    assert_eq!(result.axes.layout().labels(), "CU");
    assert_eq!(result.tensor.numeric_dtype(), NumericDType::F32);
    assert_eq!(
        result.tensor.as_f32_slice().unwrap(),
        &[ComplexElement(2.0e38, 0.)]
    );
    let gradient = ComplexTensor::from_f32(vec![(2., -4.)], vec![1, 1]).unwrap();
    let dx = backward_complex(&input, &gradient, &axes).unwrap();
    assert_eq!(dx.as_f32_slice().unwrap(), &[ComplexElement(1., -2.); 2]);

    let double =
        ComplexTensor::new(vec![(1.0e308, -1.0e308), (1.0e308, 1.0e308)], vec![1, 2]).unwrap();
    let result = forward_complex(&double, &axes).unwrap();
    assert_eq!(result.tensor.numeric_dtype(), NumericDType::F64);
    assert_eq!(
        result.tensor.as_f64_slice().unwrap(),
        &[ComplexElement(1.0e308, 0.)]
    );
}

#[test]
fn no_channel_layout_and_empty_batch_keep_checked_complex_geometry() {
    let bt = ComplexTensor::new(
        vec![
            (1., 10.),
            (2., 20.),
            (3., 30.),
            (4., 40.),
            (5., 50.),
            (6., 60.),
        ],
        vec![2, 3],
    )
    .unwrap();
    let result = forward_complex(&bt, &axes("BT", bt.shape())).unwrap();
    assert_eq!(result.axes.layout().labels(), "BU");
    assert_eq!(
        result.tensor.as_f64_slice().unwrap(),
        &[ComplexElement(3., 30.), ComplexElement(4., 40.)]
    );

    for (format, shape, expected) in [
        (
            "SB",
            vec![3, 2],
            vec![ComplexElement(2., 20.), ComplexElement(5., 50.)],
        ),
        (
            "SBT",
            vec![3, 2, 2],
            vec![
                ComplexElement(2., 20.),
                ComplexElement(5., 50.),
                ComplexElement(8., 80.),
                ComplexElement(11., 110.),
            ],
        ),
    ] {
        let count = shape.iter().product::<usize>();
        let values = (1..=count)
            .map(|value| (value as f64, (value * 10) as f64))
            .collect();
        let tensor = ComplexTensor::new(values, shape.clone()).unwrap();
        let result = forward_complex(&tensor, &axes(format, &shape)).unwrap();
        assert_eq!(result.axes.layout().labels(), format);
        assert_eq!(result.tensor.as_f64_slice().unwrap(), expected);
    }

    let layout =
        DeepLearningAxisLayout::for_input("CBT", DeepLearningInputKind::VectorSequence, 3).unwrap();
    let axes = AxisPlan::new(layout, &[2, 0, 3]).unwrap();
    let empty = ComplexTensor::new(vec![], vec![2, 0, 3]).unwrap();
    let result = forward_complex(&empty, &axes).unwrap();
    assert_eq!(result.tensor.shape(), &[2, 0]);
    let empty_gradient = ComplexTensor::new(vec![], vec![2, 0]).unwrap();
    assert_eq!(
        backward_complex(&empty, &empty_gradient, &axes)
            .unwrap()
            .shape(),
        empty.shape()
    );
}

#[test]
fn complex_gradient_class_and_shape_must_match_forward_contract() {
    let input = ComplexTensor::from_f32(vec![(1., 2.), (3., 4.)], vec![1, 2]).unwrap();
    let axes = axes("CT", input.shape());
    let wrong_class = ComplexTensor::new(vec![(1., 1.)], vec![1, 1]).unwrap();
    assert!(matches!(
        backward_complex(&input, &wrong_class, &axes),
        Err(PoolError::UnsupportedPrecision)
    ));
    let wrong_shape = ComplexTensor::from_f32(vec![(1., 1.)], vec![1]).unwrap();
    assert!(matches!(
        backward_complex(&input, &wrong_shape, &axes),
        Err(PoolError::ShapeMismatch)
    ));
    let integer = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I32(vec![3, 4]),
        )
        .unwrap(),
        vec![1, 2],
    )
    .unwrap();
    assert!(matches!(
        forward_complex(&integer, &axes),
        Err(PoolError::UnsupportedPrecision)
    ));
}
