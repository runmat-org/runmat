use super::*;
use runmat_builtins::{LOG1P_DESCRIPTOR, LOG1P_ERROR_INVALID_INPUT};
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType,
    NumericStorage, SparseTensor, Tensor,
};

#[test]
fn descriptor_invalid_input_and_compatibility_are_catalog_owned() {
    assert_eq!(LOG1P_DESCRIPTOR.signatures[0].label, "Y = log1p(X)");
    assert_eq!(
        call(Value::from("bad"))
            .expect_err("string must fail")
            .identifier(),
        LOG1P_ERROR_INVALID_INPUT.identifier
    );
    assert_eq!(
        call_matlab(Value::Int(IntValue::I8(1)))
            .expect_err("integer extension must be gated")
            .identifier(),
        Some("RunMat:compatibility:Log1pIntegerInputExtension")
    );
}

#[test]
fn real_domain_preserves_accuracy_shape_and_precision() {
    let Value::Num(zero) = call(Value::Num(0.0)).expect("log1p zero") else {
        panic!("expected scalar")
    };
    assert_eq!(zero, 0.0);
    let Value::Num(boundary) = call(Value::Num(-1.0)).expect("log1p boundary") else {
        panic!("expected scalar")
    };
    assert!(boundary.is_infinite() && boundary.is_sign_negative());
    let Value::Complex(real, imag) = call(Value::Num(-2.0)).expect("complex promotion") else {
        panic!("expected complex scalar")
    };
    assert!(real.abs() < 1e-12);
    assert!((imag - std::f64::consts::PI).abs() < 1e-12);

    let input = Tensor::from_f32(vec![0.0, 0.5], vec![2, 1]).expect("single input");
    let Value::Tensor(output) = call(Value::Tensor(input)).expect("single log1p") else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape, vec![2, 1]);
    assert_eq!(
        output.into_numeric_storage().expect("storage"),
        NumericStorage::F32(vec![0.0, 0.5_f32.ln_1p()])
    );
}

#[test]
fn complex_input_retains_near_zero_accuracy_and_single_class() {
    let input = 1.0e-20;
    let Value::Complex(real, imag) = call(Value::Complex(input, input)).expect("complex log1p")
    else {
        panic!("expected complex scalar")
    };
    assert!((real - input).abs() <= 1.0e-35, "real={real}");
    assert!((imag - input).abs() <= 1.0e-35, "imag={imag}");

    let input = ComplexTensor::from_f32(vec![(1.0, 1.0)], vec![1, 1]).expect("complex single");
    let Value::ComplexTensor(output) = call(Value::ComplexTensor(input)).expect("log1p") else {
        panic!("expected complex tensor")
    };
    assert!(matches!(
        output.into_complex_storage(),
        ComplexStorage::F32(_)
    ));
}

#[test]
fn exact_integer_boundary_covers_all_native_classes() {
    for storage in [
        IntegerStorage::I8(vec![-1, 1]),
        IntegerStorage::I16(vec![-1, 1]),
        IntegerStorage::I32(vec![-1, 1]),
        IntegerStorage::I64(vec![-9_007_199_254_740_992, 9_007_199_254_740_992]),
        IntegerStorage::U8(vec![0, 1]),
        IntegerStorage::U16(vec![0, 1]),
        IntegerStorage::U32(vec![0, 1]),
        IntegerStorage::U64(vec![0, 9_007_199_254_740_992]),
    ] {
        let input = Tensor::new_integer(storage, vec![1, 2]).expect("integer input");
        assert!(matches!(
            call(Value::Tensor(input)).expect("integer log1p"),
            Value::Tensor(_) | Value::ComplexTensor(_)
        ));
    }
    let input = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
        .expect("wide integer");
    assert_eq!(
        call(Value::Tensor(input))
            .expect_err("inexact integer must fail")
            .identifier(),
        LOG1P_ERROR_INVALID_INPUT.identifier
    );
}

#[test]
fn logical_character_sparse_and_table_contracts_are_distinct() {
    let logical = LogicalArray::new(vec![0, 1], vec![2, 1]).expect("logical input");
    let Value::Tensor(output) = call(Value::LogicalArray(logical)).expect("logical log1p") else {
        panic!("expected tensor")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F64);

    let chars = CharArray::new(vec!['A'], 1, 1).expect("character input");
    assert!(matches!(
        call(Value::CharArray(chars)),
        Ok(Value::Num(_)) | Ok(Value::Tensor(_))
    ));

    let sparse = SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![1.0]).expect("sparse");
    assert_eq!(
        call(Value::SparseTensor(sparse))
            .expect_err("sparse input must fail")
            .identifier(),
        LOG1P_ERROR_INVALID_INPUT.identifier
    );

    let table = crate::builtins::table::table_from_columns(
        vec!["X".into()],
        vec![Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap())],
    )
    .expect("table");
    assert_eq!(
        call(table)
            .expect_err("log1p has no tabular overload")
            .identifier(),
        LOG1P_ERROR_INVALID_INPUT.identifier
    );
}
