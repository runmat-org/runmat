use super::*;
use runmat_builtins::{EXPM1_DESCRIPTOR, EXPM1_ERROR_INVALID_INPUT};
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType,
    NumericStorage, SparseTensor, Tensor,
};

#[test]
fn descriptor_and_invalid_input_are_catalog_owned() {
    assert_eq!(EXPM1_DESCRIPTOR.signatures[0].label, "Y = expm1(X)");
    let error = call(Value::from("bad")).expect_err("string must be rejected");
    assert_eq!(error.identifier(), EXPM1_ERROR_INVALID_INPUT.identifier);
}

#[test]
fn scalar_accuracy_and_single_storage_are_preserved() {
    let tiny = 1.0e-16;
    let Value::Num(output) = call(Value::Num(tiny)).expect("expm1") else {
        panic!("expected scalar");
    };
    assert_eq!(output, tiny.exp_m1());

    let input = Tensor::from_f32(vec![0.0, 1.0], vec![2, 1]).expect("single input");
    let Value::Tensor(output) = call(Value::Tensor(input)).expect("single expm1") else {
        panic!("expected tensor");
    };
    assert_eq!(
        output.into_numeric_storage().expect("storage"),
        NumericStorage::F32(vec![0.0, 1.0_f32.exp_m1()])
    );
}

#[test]
fn complex_accuracy_precision_and_signed_zero_are_preserved() {
    let Value::Complex(real, imag) = call(Value::Complex(f64::INFINITY, -0.0)).expect("expm1")
    else {
        panic!("expected complex scalar");
    };
    assert_eq!(real, f64::INFINITY);
    assert!(imag.is_sign_negative());

    let input =
        ComplexTensor::from_f32(vec![(1.0e-6, 1.0e-6)], vec![1, 1]).expect("complex single");
    let Value::ComplexTensor(output) = call(Value::ComplexTensor(input)).expect("complex expm1")
    else {
        panic!("expected complex tensor");
    };
    assert!(matches!(
        output.into_complex_storage(),
        ComplexStorage::F32(_)
    ));
}

#[test]
fn integer_extensions_are_exact_and_compatibility_gated() {
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
        let Value::Tensor(output) = call(Value::Tensor(input)).expect("integer expm1") else {
            panic!("expected tensor");
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F64);
    }
    let input = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
        .expect("wide integer");
    assert_eq!(
        call(Value::Tensor(input))
            .expect_err("inexact integer must fail")
            .identifier(),
        EXPM1_ERROR_INVALID_INPUT.identifier
    );
    assert_eq!(
        call_matlab(Value::Int(IntValue::I8(1)))
            .expect_err("compatibility gate")
            .identifier(),
        Some("RunMat:compatibility:Expm1IntegerInputExtension")
    );
}

#[test]
fn sparse_input_preserves_sparse_single_storage_and_implicit_zeros() {
    let input = SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0])
        .expect("sparse single");
    let Value::SparseTensor(output) = call(Value::SparseTensor(input)).expect("sparse expm1")
    else {
        panic!("expected sparse output");
    };
    assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
    assert_eq!(output.nnz(), 2);
}

#[test]
fn table_and_character_inputs_map_elementwise() {
    let chars = CharArray::new(vec!['A'], 1, 1).expect("characters");
    assert!(matches!(
        call(Value::CharArray(chars)),
        Ok(Value::Tensor(_))
    ));

    let table = crate::builtins::table::table_from_columns(
        vec!["Double".into(), "Single".into()],
        vec![
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::from_f32(vec![0.0, 1.0], vec![2, 1]).unwrap()),
        ],
    )
    .expect("table");
    let Value::Object(output) = call(table).expect("table expm1") else {
        panic!("expected table");
    };
    let variables = crate::builtins::table::table_variables(&output).expect("variables");
    assert!(matches!(
        variables.fields.get("Single"),
        Some(Value::Tensor(tensor)) if tensor.numeric_dtype() == NumericDType::F32
    ));
}

#[test]
fn logical_and_character_extensions_have_independent_compatibility_gates() {
    let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical input");
    let Value::Tensor(output) = call(Value::LogicalArray(logical)).expect("logical expm1") else {
        panic!("expected tensor");
    };
    assert_eq!(output.materialize_f64(), &[1.0_f64.exp_m1(), 0.0]);

    let character_error = call_matlab(Value::CharArray(
        CharArray::new(vec!['A'], 1, 1).expect("character input"),
    ))
    .expect_err("character compatibility gate");
    assert_eq!(
        character_error.identifier(),
        Some("RunMat:compatibility:Expm1CharacterInputExtension")
    );
    let logical_error = call_matlab(Value::Bool(true)).expect_err("logical compatibility gate");
    assert_eq!(
        logical_error.identifier(),
        Some("RunMat:compatibility:Expm1LogicalInputExtension")
    );
}
