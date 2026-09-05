use super::*;
use runmat_builtins::{LOG2_DESCRIPTOR, LOG2_ERROR_INVALID_INPUT};
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType,
    NumericStorage, Tensor,
};

#[test]
fn descriptor_invalid_input_and_compatibility_are_catalog_owned() {
    let labels: Vec<_> = LOG2_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert!(labels.contains(&"Y = log2(X)"));
    assert!(labels.contains(&"[F,E] = log2(X)"));
    assert_eq!(
        call(Value::from("bad"))
            .expect_err("string must fail")
            .identifier(),
        LOG2_ERROR_INVALID_INPUT.identifier
    );
    assert_eq!(
        call_matlab(Value::Int(IntValue::I8(2)))
            .expect_err("integer extension must be gated")
            .identifier(),
        Some("RunMat:compatibility:Log2IntegerInputExtension")
    );
}

#[test]
fn scalar_dense_and_single_inputs_preserve_contract() {
    let Value::Num(one) = call(Value::Num(2.0)).expect("log2") else {
        panic!("expected scalar")
    };
    assert!((one - 1.0).abs() < 1e-12);
    let Value::Num(zero) = call(Value::Num(0.0)).expect("log2 zero") else {
        panic!("expected scalar")
    };
    assert!(zero.is_infinite() && zero.is_sign_negative());
    let Value::Complex(real, imag) = call(Value::Num(-4.0)).expect("complex promotion") else {
        panic!("expected complex scalar")
    };
    assert!((real - 2.0).abs() < 1e-12);
    assert!((imag - std::f64::consts::PI * std::f64::consts::LOG2_E).abs() < 1e-12);

    let input = Tensor::from_f32(vec![1.0, 2.0, 4.0], vec![3, 1]).expect("single input");
    let Value::Tensor(output) = call(Value::Tensor(input)).expect("single log2") else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape, vec![3, 1]);
    assert_eq!(
        output.into_numeric_storage().expect("storage"),
        NumericStorage::F32(vec![0.0, 1.0, 2.0])
    );
}

#[test]
fn complex_input_preserves_single_class() {
    let input = ComplexTensor::from_f32(vec![(1.0, 1.0)], vec![1, 1]).expect("complex input");
    let Value::ComplexTensor(output) = call(Value::ComplexTensor(input)).expect("log2") else {
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
            call(Value::Tensor(input)).expect("integer log2"),
            Value::Tensor(_) | Value::ComplexTensor(_)
        ));
    }
    let input = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
        .expect("wide integer");
    assert_eq!(
        call(Value::Tensor(input))
            .expect_err("inexact integer must fail")
            .identifier(),
        LOG2_ERROR_INVALID_INPUT.identifier
    );
}

#[test]
fn logical_character_and_table_inputs_follow_distinct_overloads() {
    let logical = LogicalArray::new(vec![1, 0], vec![2, 1]).expect("logical input");
    let Value::Tensor(output) = call(Value::LogicalArray(logical)).expect("logical log2") else {
        panic!("expected tensor")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F64);

    let chars = CharArray::new(vec!['A'], 1, 1).expect("character input");
    assert!(matches!(
        call(Value::CharArray(chars)),
        Ok(Value::Num(_)) | Ok(Value::Tensor(_))
    ));

    let table = crate::builtins::table::table_from_columns(
        vec!["X".into()],
        vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
        )],
    )
    .expect("table");
    let Value::Object(output) = call(table).expect("table log2") else {
        panic!("expected table")
    };
    assert!(crate::builtins::table::is_tabular_object(&output));
}
