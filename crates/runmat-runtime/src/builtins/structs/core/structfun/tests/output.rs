use super::*;
use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage, Tensor};

#[test]
fn uniform_output_preserves_single_and_exact_integer_classes() {
    let mut singles = runmat_value::UnboundedStructValueBuilder::new();
    singles.insert(
        "a",
        Value::Tensor(Tensor::from_f32(vec![1.25], vec![1, 1]).unwrap()),
    );
    singles.insert(
        "b",
        Value::Tensor(Tensor::from_f32(vec![2.5], vec![1, 1]).unwrap()),
    );
    let result = call(
        Value::FunctionHandle("single".into()),
        singles.finish(),
        Vec::new(),
    )
    .unwrap();
    let Value::Tensor(single) = result.value() else {
        panic!("expected tensor")
    };
    assert_eq!(single.numeric_dtype(), runmat_value::NumericDType::F32);

    let mut integers = runmat_value::UnboundedStructValueBuilder::new();
    integers.insert("a", Value::Int(runmat_value::IntValue::U64(u64::MAX)));
    integers.insert("b", Value::Int(runmat_value::IntValue::U64(u64::MAX - 1)));
    let result = call(
        Value::FunctionHandle("uint64".into()),
        integers.finish(),
        Vec::new(),
    )
    .unwrap();
    let Value::Tensor(integer) = result.value() else {
        panic!("expected tensor")
    };
    assert_eq!(integer.numeric_dtype(), runmat_value::NumericDType::U64);
    assert_eq!(
        integer.integer_storage().unwrap().value_at(0),
        Some(runmat_value::IntValue::U64(u64::MAX))
    );
}

#[test]
fn uniform_output_preserves_integer_complex_storage() {
    let storage =
        IntegerComplexStorage::new(IntegerStorage::I16(vec![-4]), IntegerStorage::I16(vec![9]))
            .unwrap();
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert(
        "value",
        Value::ComplexTensor(ComplexTensor::new_integer(storage, vec![1, 1]).unwrap()),
    );
    let result = call(
        Value::FunctionHandle("conj".into()),
        structure.finish(),
        Vec::new(),
    )
    .unwrap();
    let Value::ComplexTensor(output) = result.value() else {
        panic!("expected complex tensor")
    };
    assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::I16);
}

#[test]
fn uniform_output_rejects_mixed_classes() {
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert("a", Value::Int(runmat_value::IntValue::I8(1)));
    structure.insert("b", Value::Int(runmat_value::IntValue::I16(2)));
    let error = call(
        Value::FunctionHandle("abs".into()),
        structure.finish(),
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::STRUCTFUN_ERROR_UNIFORM_OUTPUT.identifier
    );
}
