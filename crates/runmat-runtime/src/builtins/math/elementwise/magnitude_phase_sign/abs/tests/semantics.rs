use super::*;
use runmat_builtins::{ABS_CHARACTER_INPUT_EXTENSION, ABS_LOGICAL_INPUT_EXTENSION};

#[test]
fn abs_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = ABS_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"Y = abs(X)"));
    assert_eq!(ABS_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(ABS_EXTENSIONS.len(), 3);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abs_scalar_negative() {
    let result = abs_builtin(Value::Num(-3.5)).expect("abs");
    match result {
        Value::Num(n) => assert!((n - 3.5).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abs_integer_scalar_preserves_class_and_saturates_signed_minimum() {
    let result = abs_builtin(Value::Int(IntValue::I32(-8))).expect("abs");
    assert_eq!(result, Value::Int(IntValue::I32(8)));
    assert_eq!(
        abs_builtin(Value::Int(IntValue::I64(i64::MIN))).expect("abs"),
        Value::Int(IntValue::I64(i64::MAX))
    );
}

#[test]
fn abs_preserves_native_single_and_rejects_typed_complex_integer() {
    let single = Tensor::from_f32(vec![-2.5, 0.0, 3.25], vec![1, 3]).unwrap();
    let output = abs_tensor(single).unwrap();
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![2.5, 0.0, 3.25])
    );

    let storage = IntegerComplexStorage::new(
        IntegerStorage::I16(vec![3, -4]),
        IntegerStorage::I16(vec![4, 3]),
    )
    .unwrap();
    let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).unwrap();
    let error = abs_builtin(Value::ComplexTensor(tensor)).unwrap_err();
    assert!(error
        .message()
        .contains("complex numbers with integer types"));
}

#[test]
fn abs_complex_single_preserves_native_class_shape_and_empty_storage() {
    let complex = ComplexTensor::from_f32(vec![(3.0, 4.0), (5.0, 12.0)], vec![2, 1]).unwrap();
    let Value::Tensor(output) = abs_builtin(Value::ComplexTensor(complex)).unwrap() else {
        panic!("expected single magnitude tensor");
    };
    assert_eq!(output.shape, vec![2, 1]);
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![5.0, 13.0])
    );

    let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
    let Value::Tensor(output) = abs_builtin(Value::ComplexTensor(empty)).unwrap() else {
        panic!("expected empty single magnitude tensor");
    };
    assert_eq!(output.shape, vec![0, 3]);
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(Vec::new())
    );
}

#[test]
fn abs_preserves_all_typed_integer_array_classes_exactly() {
    let cases = [
        (
            IntegerStorage::I8(vec![i8::MIN, -4, 0, i8::MAX]),
            IntegerStorage::I8(vec![i8::MAX, 4, 0, i8::MAX]),
        ),
        (
            IntegerStorage::I16(vec![i16::MIN, -4, 0, i16::MAX]),
            IntegerStorage::I16(vec![i16::MAX, 4, 0, i16::MAX]),
        ),
        (
            IntegerStorage::I32(vec![i32::MIN, -4, 0, i32::MAX]),
            IntegerStorage::I32(vec![i32::MAX, 4, 0, i32::MAX]),
        ),
        (
            IntegerStorage::I64(vec![i64::MIN, -4, 0, i64::MAX]),
            IntegerStorage::I64(vec![i64::MAX, 4, 0, i64::MAX]),
        ),
        (
            IntegerStorage::U8(vec![0, 4, u8::MAX]),
            IntegerStorage::U8(vec![0, 4, u8::MAX]),
        ),
        (
            IntegerStorage::U16(vec![0, 4, u16::MAX]),
            IntegerStorage::U16(vec![0, 4, u16::MAX]),
        ),
        (
            IntegerStorage::U32(vec![0, 4, u32::MAX]),
            IntegerStorage::U32(vec![0, 4, u32::MAX]),
        ),
        (
            IntegerStorage::U64(vec![0, 4, u64::MAX]),
            IntegerStorage::U64(vec![0, 4, u64::MAX]),
        ),
    ];
    for (input, expected) in cases {
        let input = Tensor::new_integer(input, vec![1, expected.len()]).expect("tensor");
        let Value::Tensor(result) = abs_builtin(Value::Tensor(input)).expect("abs") else {
            panic!("expected tensor");
        };
        assert_eq!(result.integer_storage(), Some(&expected));
    }
}

#[test]
fn abs_sparse_preserves_floating_class_structure_and_exact_integer_extension() {
    let double =
        SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![-2.5, 0.0, -4.0]).unwrap();
    let Value::SparseTensor(double) =
        abs_builtin(Value::SparseTensor(double)).expect("double sparse abs")
    else {
        panic!("expected sparse output");
    };
    assert_eq!(double.col_ptrs, vec![0, 2, 3]);
    assert_eq!(double.row_indices, vec![0, 2, 1]);
    assert_eq!(double.as_f64_slice(), Some(&[2.5, 0.0, 4.0][..]));

    let single = SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![1, 0], vec![-3.0, 4.5]).unwrap();
    let Value::SparseTensor(single) =
        abs_builtin(Value::SparseTensor(single)).expect("single sparse abs")
    else {
        panic!("expected sparse output");
    };
    assert_eq!(single.as_f32_slice(), Some(&[3.0, 4.5][..]));

    let integer = SparseTensor::new_integer(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        IntegerStorage::I64(vec![i64::MIN, -9_007_199_254_740_993]),
    )
    .unwrap();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let Value::SparseTensor(integer) =
        abs_builtin(Value::SparseTensor(integer)).expect("integer sparse abs")
    else {
        panic!("expected sparse output");
    };
    assert_eq!(
        integer.integer_storage(),
        Some(&IntegerStorage::I64(vec![i64::MAX, 9_007_199_254_740_993]))
    );
}

#[test]
fn abs_extensions_and_output_arity_are_enforced_before_execution() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let logical_error = abs_builtin(Value::LogicalArray(
        LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap(),
    ))
    .expect_err("logical input is an extension");
    assert_eq!(
        logical_error.identifier(),
        ABS_LOGICAL_INPUT_EXTENSION.error_identifier
    );
    let char_error = abs_builtin(Value::CharArray(
        CharArray::new("A".chars().collect(), 1, 1).unwrap(),
    ))
    .expect_err("character input is an extension");
    assert_eq!(
        char_error.identifier(),
        ABS_CHARACTER_INPUT_EXTENSION.error_identifier
    );
    let sparse_integer =
        SparseTensor::new_integer(1, 1, vec![0, 1], vec![0], IntegerStorage::I8(vec![-1])).unwrap();
    let sparse_error = abs_builtin(Value::SparseTensor(sparse_integer))
        .expect_err("integer sparse input is an extension");
    assert_eq!(
        sparse_error.identifier(),
        crate::compatibility::SPARSE_INTEGER_EXTENSION.error_identifier
    );

    let _outputs = crate::output_count::push_output_count(Some(2));
    let output_error =
        abs_builtin(Value::Int(IntValue::I8(-1))).expect_err("excess outputs reject");
    assert_eq!(
        output_error.identifier(),
        ABS_ERROR_TOO_MANY_OUTPUTS.identifier
    );
}

#[test]
fn abs_zero_output_still_validates_input() {
    let _outputs = crate::output_count::push_output_count(Some(0));
    let error = abs_builtin(Value::from("not numeric")).expect_err("input must validate");
    assert_eq!(error.identifier(), ABS_ERROR_INVALID_INPUT.identifier);
}
