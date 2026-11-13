use runmat_value::{ComplexTensor, NumericDType, Tensor};

use super::*;

fn axes(format: &str, shape: &[usize]) -> AxisPlan {
    AxisPlan::new(
        DeepLearningAxisLayout::parse(format, shape.len()).unwrap(),
        shape,
    )
    .unwrap()
}

#[test]
fn observation_major_moves_batch_first_with_nonsquare_native_single_payload() {
    let source = axes("CBT", &[2, 3, 2]);
    let target = target_axes(&source, NetworkOutputLayout::ObservationMajor, "predict").unwrap();
    assert_eq!(
        target.layout().axes(),
        &[
            DeepLearningAxisLabel::Batch,
            DeepLearningAxisLabel::Channel,
            DeepLearningAxisLabel::Time,
        ]
    );
    assert_eq!(target.shape(), &[3, 2, 2]);
    let input = Tensor::from_f32(
        vec![
            1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0,
        ],
        vec![2, 3, 2],
    )
    .unwrap();
    let HostValue::Real(result) =
        present_host(HostValue::Real(input), &source, &target, "predict").unwrap()
    else {
        panic!("real result expected")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);
    assert_eq!(result.shape(), &[3, 2, 2]);
    assert_eq!(
        result.as_f32_slice().unwrap(),
        &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0]
    );
}

#[test]
fn observation_major_preserves_both_complex_components_and_class() {
    let source = axes("CB", &[2, 3]);
    let target = target_axes(&source, NetworkOutputLayout::ObservationMajor, "predict").unwrap();
    let input = ComplexTensor::from_f32(
        vec![
            (1.0, -1.0),
            (10.0, -10.0),
            (2.0, -2.0),
            (20.0, -20.0),
            (3.0, -3.0),
            (30.0, -30.0),
        ],
        vec![2, 3],
    )
    .unwrap();
    let HostValue::Complex(result) =
        present_host(HostValue::Complex(input), &source, &target, "predict").unwrap()
    else {
        panic!("complex result expected")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);
    assert_eq!(result.shape(), &[3, 2]);
    let values = result.as_f32_slice().unwrap();
    let pairs = values
        .iter()
        .map(|value| (value.0, value.1))
        .collect::<Vec<_>>();
    assert_eq!(
        pairs,
        vec![
            (1.0, -1.0),
            (2.0, -2.0),
            (3.0, -3.0),
            (10.0, -10.0),
            (20.0, -20.0),
            (30.0, -30.0),
        ]
    );
}

#[test]
fn unbatched_and_already_observation_major_outputs_are_noops() {
    for (format, shape) in [("CT", vec![2, 3]), ("BC", vec![3, 2])] {
        let source = axes(format, &shape);
        let target =
            target_axes(&source, NetworkOutputLayout::ObservationMajor, "predict").unwrap();
        assert_eq!(target, source);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn observation_major_rejects_same_size_host_output_with_wrong_admitted_shape() {
    let source = axes("CB", &[2, 3]);
    let target = target_axes(&source, NetworkOutputLayout::ObservationMajor, "predict").unwrap();
    let input = Tensor::from_f32(vec![1.0; 6], vec![1, 6]).unwrap();
    let Err(error) = present_host(HostValue::Real(input), &source, &target, "predict") else {
        panic!("mismatched host output shape must reject before reordering")
    };
    assert!(error
        .message()
        .contains("host output shape differs from admitted axes"));
}
