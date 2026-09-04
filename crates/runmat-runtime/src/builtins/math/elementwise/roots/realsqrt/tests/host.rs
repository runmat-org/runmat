use super::*;
use runmat_builtins::{
    REALSQRT_DESCRIPTOR, REALSQRT_ERROR_DOMAIN as ERROR_DOMAIN,
    REALSQRT_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT,
};
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType, NumericStorage,
    SparseTensor, Tensor,
};

#[test]
fn descriptor_covers_core_form() {
    assert_eq!(REALSQRT_DESCRIPTOR.signatures[0].label, "Y = realsqrt(X)");
}

#[test]
fn double_scalar_nonnegative_value_returns_real_root() {
    match call(Value::Num(9.0)).unwrap() {
        Value::Num(value) => assert!((value - 3.0).abs() < 1.0e-12),
        other => panic!("expected numeric scalar, got {other:?}"),
    }
}

#[test]
fn double_scalar_negative_zero_is_canonicalized() {
    let result = call(Value::Num(-0.0)).expect("realsqrt");
    let Value::Num(value) = result else {
        panic!("expected numeric scalar, got {result:?}");
    };
    assert_eq!(value.to_bits(), 0.0f64.to_bits());
}

#[test]
fn scalar_negative_errors_instead_of_promoting_complex() {
    let err = call(Value::Num(-1.0)).unwrap_err();
    assert_eq!(err.identifier(), ERROR_DOMAIN.identifier);
}

#[test]
fn nan_and_infinity_follow_ieee_real_sqrt() {
    match call(Value::Tensor(
        Tensor::new(vec![f64::NAN, f64::INFINITY], vec![1, 2]).unwrap(),
    ))
    .unwrap()
    {
        Value::Tensor(tensor) => {
            assert!(tensor.materialize_f64()[0].is_nan());
            assert!(tensor.materialize_f64()[1].is_infinite());
            assert!(tensor.materialize_f64()[1].is_sign_positive());
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn dense_tensor_preserves_shape() {
    let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![2, 2]).unwrap();
    match call(Value::Tensor(tensor)).unwrap() {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![2, 2]);
            assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 2.0, 3.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn dense_typed_integer_tensor_is_rejected_by_class() {
    let tensor = Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 4, 9]), vec![2, 2]).unwrap();

    let err = call(Value::Tensor(tensor)).unwrap_err();
    assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn dense_negative_typed_integer_is_rejected_by_class_before_domain() {
    let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, -4]), vec![1, 2]).unwrap();

    let err = call(Value::Tensor(tensor)).unwrap_err();
    assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn single_tensor_preserves_dtype() {
    let tensor = Tensor::from_f32(vec![2.0, 9.0], vec![1, 2]).unwrap();
    match call(Value::Tensor(tensor)).unwrap() {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![1, 2]);
            assert_eq!(out.numeric_dtype(), NumericDType::F32);
            assert_eq!(
                out.into_numeric_storage().unwrap(),
                NumericStorage::F32(vec![2.0f32.sqrt(), 3.0])
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn dense_tensor_with_negative_errors() {
    let tensor = Tensor::new(vec![1.0, -4.0], vec![1, 2]).unwrap();
    let err = call(Value::Tensor(tensor)).unwrap_err();
    assert_eq!(err.identifier(), ERROR_DOMAIN.identifier);
}

#[test]
fn integer_logical_and_char_inputs_are_rejected() {
    let integer = call(Value::Int(IntValue::I32(16))).unwrap_err();
    assert_eq!(integer.identifier(), ERROR_INVALID_INPUT.identifier);
    let boolean = call(Value::Bool(true)).unwrap_err();
    assert_eq!(boolean.identifier(), ERROR_INVALID_INPUT.identifier);

    let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
    let logical = call(Value::LogicalArray(logical)).unwrap_err();
    assert_eq!(logical.identifier(), ERROR_INVALID_INPUT.identifier);

    let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
    let chars = call(Value::CharArray(chars)).unwrap_err();
    assert_eq!(chars.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn complex_inputs_error() {
    let err = call(Value::Complex(1.0, 0.0)).unwrap_err();
    assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);

    let tensor = ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
    let err = call(Value::ComplexTensor(tensor)).unwrap_err();
    assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn string_inputs_error() {
    let err = call(Value::String("9".to_string())).unwrap_err();
    assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn sparse_inputs_preserve_sparsity() {
    let sparse =
        SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![4.0, 9.0, 16.0]).unwrap();
    match call(Value::SparseTensor(sparse)).unwrap() {
        Value::SparseTensor(out) => {
            assert_eq!(out.rows, 3);
            assert_eq!(out.cols, 2);
            assert_eq!(out.col_ptrs, vec![0, 2, 3]);
            assert_eq!(out.row_indices, vec![0, 2, 1]);
            assert_eq!(out.materialize_f64(), vec![2.0, 3.0, 4.0]);
        }
        other => panic!("expected sparse tensor, got {other:?}"),
    }
}

#[test]
fn sparse_single_inputs_preserve_native_single_storage() {
    let sparse =
        SparseTensor::new_f32(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![4.0, 9.0, 16.0]).unwrap();
    let Value::SparseTensor(output) = call(Value::SparseTensor(sparse)).unwrap() else {
        panic!("expected sparse tensor");
    };
    assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
    assert_eq!(output.as_f32_slice(), Some(&[2.0, 3.0, 4.0][..]));
}

#[test]
fn sparse_complex_inputs_reject_without_discarding_imaginary_storage() {
    let sparse =
        SparseTensor::new_complex_f32(2, 1, vec![0, 1], vec![1], vec![(4.0, 3.0)]).unwrap();

    let err = call(Value::SparseTensor(sparse)).unwrap_err();
    assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn sparse_typed_integer_values_are_rejected_by_class() {
    let sparse = SparseTensor::new_integer(
        3,
        2,
        vec![0, 2, 3],
        vec![0, 2, 1],
        IntegerStorage::U16(vec![4, 9, 16]),
    )
    .unwrap();

    let err = call(Value::SparseTensor(sparse)).unwrap_err();
    assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn sparse_negative_typed_integer_is_rejected_by_class_before_domain() {
    let sparse =
        SparseTensor::new_integer(2, 1, vec![0, 1], vec![1], IntegerStorage::I16(vec![-4]))
            .unwrap();

    let err = call(Value::SparseTensor(sparse)).unwrap_err();
    assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn sparse_negative_value_errors() {
    let sparse = SparseTensor::new(2, 1, vec![0, 1], vec![1], vec![-4.0]).unwrap();
    let err = call(Value::SparseTensor(sparse)).unwrap_err();
    assert_eq!(err.identifier(), ERROR_DOMAIN.identifier);
}
