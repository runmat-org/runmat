use super::*;
use runmat_value::{ComplexTensor, NumericStorage};

#[test]
fn sqrt_preserves_native_single_real_complex_negative_and_empty_storage() {
    let tensor = Tensor::from_f32(vec![0.0, 4.0], vec![2, 1]).unwrap();
    let Value::Tensor(output) = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt") else {
        panic!("expected single real tensor");
    };
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![0.0, 2.0])
    );

    let tensor = Tensor::from_f32(vec![-4.0, 9.0], vec![1, 2]).unwrap();
    let Value::ComplexTensor(output) = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt") else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        output.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(0.0, 2.0), (3.0, 0.0)])
    );
    let complex = ComplexTensor::from_f32(vec![(3.0, 4.0)], vec![1, 1]).unwrap();
    let Value::ComplexTensor(output) = sqrt_builtin(Value::ComplexTensor(complex)).expect("sqrt")
    else {
        panic!("one-element complex single must retain class");
    };
    assert_eq!(
        output.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![sqrt_complex_parts_f32(3.0, 4.0)])
    );
    let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let Value::ComplexTensor(output) = sqrt_builtin(Value::ComplexTensor(empty)).expect("sqrt")
    else {
        panic!("expected empty complex single tensor");
    };
    assert_eq!(output.shape(), &[0, 2]);
    assert_eq!(output.as_f32_slice(), Some(&[][..]));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn sqrt_low_rank_inputs_preserve_existing_scalarization_and_class_policy() {
    let positive = Tensor::new(vec![4.0], Vec::new()).expect("rank-zero positive");
    assert_eq!(
        sqrt_builtin(Value::Tensor(positive)).expect("positive sqrt"),
        Value::Num(2.0)
    );

    let negative = Tensor::new(vec![-4.0], Vec::new()).expect("rank-zero negative");
    assert_eq!(
        sqrt_builtin(Value::Tensor(negative)).expect("negative sqrt"),
        Value::Complex(0.0, 2.0)
    );

    let positive = Tensor::from_f32(vec![4.0], Vec::new()).expect("single positive");
    let Value::Tensor(positive) = sqrt_builtin(Value::Tensor(positive)).expect("single sqrt")
    else {
        panic!("single scalar retains tensor class")
    };
    assert!(positive.shape().is_empty());
    assert_eq!(positive.numeric_dtype(), runmat_value::NumericDType::F32);

    let negative = Tensor::from_f32(vec![-4.0], Vec::new()).expect("single negative");
    let Value::ComplexTensor(negative) =
        sqrt_builtin(Value::Tensor(negative)).expect("single complex sqrt")
    else {
        panic!("single negative scalar retains complex tensor class")
    };
    assert!(negative.shape().is_empty());
    assert_eq!(negative.numeric_dtype(), runmat_value::NumericDType::F32);

    let vector =
        ComplexTensor::new(vec![(3.0, 4.0), (-4.0, 0.0)], vec![2]).expect("complex vector");
    let Value::ComplexTensor(vector) =
        sqrt_builtin(Value::ComplexTensor(vector)).expect("complex sqrt")
    else {
        panic!("rank-one complex output remains a tensor")
    };
    assert_eq!(vector.shape(), &[2]);
    for ((actual_real, actual_imag), (expected_real, expected_imag)) in vector
        .materialize_f64()
        .into_iter()
        .zip([(2.0, 1.0), (0.0, 2.0)])
    {
        assert!((actual_real - expected_real).abs() < 1e-12);
        assert!((actual_imag - expected_imag).abs() < 1e-12);
    }
}
