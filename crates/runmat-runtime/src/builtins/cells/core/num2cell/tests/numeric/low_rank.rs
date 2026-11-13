use super::*;
use runmat_value::IntegerComplexStorage;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn low_rank_numeric_inputs_preserve_cell_geometry_and_native_values() {
    let scalar = Tensor::from_f32(vec![1.25], Vec::new()).unwrap();
    let Value::Cell(scalar) = run(Value::Tensor(scalar), Vec::new()) else {
        panic!("scalar cell expected")
    };
    assert_eq!(scalar.shape, vec![1, 1]);
    let Value::Tensor(value) = &scalar.data()[0] else {
        panic!("single scalar retains native storage")
    };
    assert_eq!(value.shape(), &[1, 1]);
    assert_eq!(value.as_f32_slice(), Some(&[1.25][..]));

    let vector =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1 << 63]), vec![2]).unwrap();
    let Value::Cell(vector) = run(Value::Tensor(vector), Vec::new()) else {
        panic!("vector cell expected")
    };
    assert_eq!(vector.shape, vec![2]);
    assert_eq!(
        vector.data(),
        &[
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Int(IntValue::U64(1 << 63))
        ]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rank_zero_numeric_and_complex_arrays_make_one_exact_cell() {
    let Value::Cell(double) = run(
        Value::Tensor(Tensor::new(vec![2.5], vec![]).unwrap()),
        Vec::new(),
    ) else {
        panic!("double scalar cell expected")
    };
    assert_eq!(double.shape, vec![1, 1]);
    assert_eq!(double.data(), vec![Value::Num(2.5)]);

    let Value::Cell(integer) = run(
        Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![]).unwrap()),
        Vec::new(),
    ) else {
        panic!("integer scalar cell expected")
    };
    assert_eq!(integer.shape, vec![1, 1]);
    assert_eq!(integer.data(), vec![Value::Int(IntValue::U64(u64::MAX))]);

    let Value::Cell(complex_double) = run(
        Value::ComplexTensor(ComplexTensor::new(vec![(3.0, -4.0)], vec![]).unwrap()),
        Vec::new(),
    ) else {
        panic!("complex double scalar cell expected")
    };
    assert_eq!(complex_double.shape, vec![1, 1]);
    assert_eq!(complex_double.data(), vec![Value::Complex(3.0, -4.0)]);

    let Value::Cell(complex_single) = run(
        Value::ComplexTensor(ComplexTensor::from_f32(vec![(1.5, -2.0)], vec![]).unwrap()),
        Vec::new(),
    ) else {
        panic!("complex single scalar cell expected")
    };
    assert_eq!(complex_single.shape, vec![1, 1]);
    let Value::ComplexTensor(value) = &complex_single.data()[0] else {
        panic!("complex single retains native storage")
    };
    assert_eq!(value.shape(), &[1, 1]);
    assert!(matches!(value.complex_storage(), ComplexStorage::F32(_)));
    assert_eq!(value.materialize_f64(), vec![(1.5, -2.0)]);

    let components = IntegerComplexStorage::new(
        IntegerStorage::U64(vec![u64::MAX]),
        IntegerStorage::U64(vec![1_u64 << 63]),
    )
    .unwrap();
    let Value::Cell(complex_integer) = run(
        Value::ComplexTensor(ComplexTensor::new_integer(components.clone(), vec![]).unwrap()),
        Vec::new(),
    ) else {
        panic!("complex integer scalar cell expected")
    };
    assert_eq!(complex_integer.shape, vec![1, 1]);
    let Value::ComplexTensor(value) = &complex_integer.data()[0] else {
        panic!("complex integer retains exact component storage")
    };
    assert_eq!(value.shape(), &[1, 1]);
    assert_eq!(value.integer_storage(), Some(&components));
}
