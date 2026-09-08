use super::*;
use runmat_value::{ComplexTensor, IntegerComplexStorage};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn typed_complex_integer_cells_preserve_exact_components() {
    let first = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX]),
            IntegerStorage::U64(vec![5]),
        )
        .expect("storage"),
        vec![1, 1],
    )
    .expect("tensor");
    let second = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![1_u64 << 63]),
            IntegerStorage::U64(vec![6]),
        )
        .expect("storage"),
        vec![1, 1],
    )
    .expect("tensor");
    let cell = crate::make_cell(
        vec![Value::ComplexTensor(first), Value::ComplexTensor(second)],
        1,
        2,
    )
    .expect("cell");

    let result = run(cell).expect("cell2mat");
    assert!(matches!(
        result,
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_storage().as_ref().is_some_and(|storage|
                    storage.real == IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63])
                        && storage.imag == IntegerStorage::U64(vec![5, 6]))
    ));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn typed_complex_integer_cells_reject_mixed_classes() {
    let u64_value = ComplexTensor::new_integer(
        IntegerComplexStorage::new(IntegerStorage::U64(vec![1]), IntegerStorage::U64(vec![2]))
            .expect("storage"),
        vec![1, 1],
    )
    .expect("tensor");
    let i64_value = ComplexTensor::new_integer(
        IntegerComplexStorage::new(IntegerStorage::I64(vec![1]), IntegerStorage::I64(vec![2]))
            .expect("storage"),
        vec![1, 1],
    )
    .expect("tensor");
    let cell = crate::make_cell(
        vec![
            Value::ComplexTensor(u64_value),
            Value::ComplexTensor(i64_value),
        ],
        1,
        2,
    )
    .expect("cell");

    let err = run(cell).expect_err("mixed integer classes must reject");
    assert!(err
        .to_string()
        .contains("must share the same integer class"));
}
