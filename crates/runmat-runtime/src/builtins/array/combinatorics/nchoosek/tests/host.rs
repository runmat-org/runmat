use super::*;
use crate::builtins::common::test_support;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, LogicalArray, NumericDType, NumericStorage,
};

#[test]
fn coefficients_preserve_compatible_classes_and_limits() {
    assert_eq!(
        call(Value::Num(5.0), Value::Num(4.0)).unwrap(),
        Value::Num(5.0)
    );
    assert_eq!(
        call(Value::Int(IntValue::U16(5)), Value::Int(IntValue::U16(2))).unwrap(),
        Value::Int(IntValue::U16(10))
    );
    assert_eq!(
        call(Value::Num(5.0), Value::Int(IntValue::U16(2))).unwrap(),
        Value::Int(IntValue::U16(10))
    );
    let selection = Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap();
    let Value::Tensor(output) = call(Value::Num(5.0), Value::Tensor(selection)).unwrap() else {
        panic!("expected single coefficient")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.materialize_f64(), vec![10.0]);

    let mismatch = call(Value::Int(IntValue::U8(5)), Value::Int(IntValue::U16(2)))
        .expect_err("mismatched integer classes");
    assert_eq!(mismatch.identifier(), Some("RunMat:nchoosek:InvalidInput"));
    let overflow =
        call(Value::Int(IntValue::U8(20)), Value::Num(10.0)).expect_err("coefficient overflow");
    assert_eq!(overflow.identifier(), Some("RunMat:nchoosek:TooLarge"));
    assert_eq!(
        call(Value::Num(3.0), Value::Num(5.0)).unwrap(),
        Value::Num(0.0)
    );
}

#[test]
fn numeric_combinations_have_compatible_order_shape_and_empty_forms() {
    let vector = Tensor::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], vec![1, 5]).unwrap();
    let Value::Tensor(output) = call(Value::Tensor(vector), Value::Num(4.0)).unwrap() else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape, vec![5, 4]);
    assert_eq!(
        output.materialize_f64(),
        vec![
            2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 10.0,
            10.0, 10.0, 10.0
        ]
    );

    let vector = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let Value::Tensor(empty) = call(Value::Tensor(vector.clone()), Value::Num(0.0)).unwrap() else {
        panic!("expected tensor")
    };
    assert_eq!(empty.shape, vec![1, 0]);
    let Value::Tensor(empty) = call(Value::Tensor(vector), Value::Num(4.0)).unwrap() else {
        panic!("expected tensor")
    };
    assert_eq!(empty.shape, vec![0, 4]);

    let single = Tensor::from_f32(vec![2.0, 4.0, 6.0], vec![1, 3]).unwrap();
    let Value::Tensor(single) = call(Value::Tensor(single), Value::Num(2.0)).unwrap() else {
        panic!("expected tensor")
    };
    assert_eq!(
        single.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![2.0, 2.0, 4.0, 4.0, 6.0, 6.0])
    );
}

#[test]
fn logical_character_and_complex_vectors_are_supported() {
    let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
    let Value::LogicalArray(output) = call(Value::LogicalArray(logical), Value::Num(2.0)).unwrap()
    else {
        panic!("expected logical")
    };
    assert_eq!(output.data, vec![1, 1, 0, 0, 1, 1]);

    let Value::CharArray(output) = call(
        Value::CharArray(CharArray::new_row("abcd")),
        Value::Num(2.0),
    )
    .unwrap() else {
        panic!("expected characters")
    };
    assert_eq!(
        output.data,
        vec!['a', 'b', 'a', 'c', 'a', 'd', 'b', 'c', 'b', 'd', 'c', 'd']
    );

    let complex =
        ComplexTensor::from_f32(vec![(1.0, 0.5), (2.0, -0.5), (3.0, 1.0)], vec![1, 3]).unwrap();
    let Value::ComplexTensor(output) =
        call(Value::ComplexTensor(complex), Value::Num(2.0)).unwrap()
    else {
        panic!("expected complex tensor")
    };
    assert!(matches!(output.complex_storage(), ComplexStorage::F32(_)));
}

#[test]
fn invalid_shapes_values_and_provider_inputs_return_stable_errors() {
    let invalid = call(Value::Num(2.5), Value::Num(1.0)).expect_err("fractional n");
    assert_eq!(invalid.identifier(), Some("RunMat:nchoosek:InvalidInput"));
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let invalid = call(Value::Tensor(matrix), Value::Num(2.0)).expect_err("matrix");
    assert_eq!(invalid.identifier(), Some("RunMat:nchoosek:InvalidInput"));

    test_support::with_test_provider(|provider| {
        let vector = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &vector.materialize_f64(),
                shape: &vector.shape,
            })
            .unwrap();
        let invalid = call(Value::GpuTensor(handle), Value::Num(2.0)).expect_err("gpu input");
        assert_eq!(invalid.identifier(), Some("RunMat:nchoosek:InvalidInput"));
    });
}
