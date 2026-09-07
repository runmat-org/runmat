use super::*;

#[test]
fn bsxfun_output_collector_preserves_complex_single_storage() {
    let mut collector = output::UniformCollector::default();
    for values in [[(1.25_f32, -2.5_f32)], [(3.5_f32, 4.75_f32)]] {
        let tensor = ComplexTensor::from_f32(values.to_vec(), vec![1, 1]).expect("scalar");
        collector
            .push(Value::ComplexTensor(tensor))
            .expect("collect complex single");
    }
    let result = collector
        .finish(
            &[2, 1],
            callback::OutputContract::Complex(runmat_value::NumericDType::F32),
        )
        .expect("finish complex single");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex tensor");
    };
    assert_eq!(
        result.as_f32_slice().expect("single storage"),
        &[(1.25, -2.5).into(), (3.5, 4.75).into()]
    );
}

#[test]
fn bsxfun_output_collector_preserves_complex_integer_storage() {
    let mut collector = output::UniformCollector::default();
    for (real, imaginary) in [(8_i16, -3_i16), (i16::MAX, i16::MIN)] {
        let storage = runmat_value::IntegerComplexStorage::new(
            IntegerStorage::I16(vec![real]),
            IntegerStorage::I16(vec![imaginary]),
        )
        .expect("storage");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("scalar");
        collector
            .push(Value::ComplexTensor(tensor))
            .expect("collect complex integer");
    }
    let result = collector
        .finish(
            &[2, 1],
            callback::OutputContract::Complex(runmat_value::NumericDType::I16),
        )
        .expect("finish complex integer");
    let Value::ComplexTensor(result) = result else {
        panic!("expected complex tensor");
    };
    let storage = result.integer_storage().expect("integer complex storage");
    assert_eq!(storage.real, IntegerStorage::I16(vec![8, i16::MAX]));
    assert_eq!(storage.imag, IntegerStorage::I16(vec![-3, i16::MIN]));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_gt_collects_logical_outputs() {
    let column = Tensor::new(vec![8.0, 17.0, 20.0, 24.0], vec![4, 1]).unwrap();
    let row = Tensor::new(vec![0.0, 10.0, 21.0], vec![1, 3]).unwrap();
    let result = call(
        Value::FunctionHandle("gt".to_string()),
        Value::Tensor(column),
        Value::Tensor(row),
    )
    .expect("bsxfun gt");

    let Value::LogicalArray(array) = result else {
        panic!("expected logical result, got {result:?}");
    };
    assert_eq!(array.shape, vec![4, 3]);
    assert_eq!(array.data, vec![1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_handles_complex_callback_results() {
    let left = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -1.0)], vec![2, 1]).unwrap();
    let result = call(
        Value::FunctionHandle("plus".to_string()),
        Value::ComplexTensor(left),
        Value::Num(10.0),
    )
    .expect("bsxfun complex");

    let Value::ComplexTensor(tensor) = result else {
        panic!("expected complex tensor");
    };
    assert_eq!(tensor.shape, vec![2, 1]);
    assert_eq!(tensor.materialize_f64(), vec![(11.0, 2.0), (13.0, -1.0)]);
}

#[test]
fn bsxfun_callback_classifier_reads_typed_complex_integer_storage_exactly() {
    let storage = runmat_value::IntegerComplexStorage::new(
        runmat_value::IntegerStorage::I16(vec![8]),
        runmat_value::IntegerStorage::I16(vec![-3]),
    )
    .expect("storage");
    let complex = ComplexTensor::new_integer(storage, vec![1, 1]).expect("typed complex");

    assert_eq!(
        classify_value(&Value::ComplexTensor(complex)).expect("classify"),
        ClassifiedValue::Complex(ComplexClassedValue {
            class: runmat_value::NumericDType::I16,
            real: NumericScalar::I16(8),
            imaginary: NumericScalar::I16(-3),
        })
    );
}

#[test]
fn bsxfun_rejects_typed_complex_integer_inputs_before_callback_dispatch() {
    let complex = ComplexTensor::new_integer(
        runmat_value::IntegerComplexStorage::new(
            runmat_value::IntegerStorage::U64(vec![u64::MAX]),
            runmat_value::IntegerStorage::U64(vec![1]),
        )
        .expect("storage"),
        vec![1, 1],
    )
    .expect("tensor");

    let err = call(
        Value::FunctionHandle("plus".to_string()),
        Value::ComplexTensor(complex),
        Value::Num(1.0),
    )
    .expect_err("typed complex integer input must reject");
    assert!(err.message().contains("complex numbers with integer types"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bsxfun_empty_logical_callback_preserves_logical_type() {
    let empty = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let result = call(
        Value::FunctionHandle("gt".to_string()),
        Value::Tensor(empty),
        Value::Tensor(row),
    )
    .expect("bsxfun empty logical");

    let Value::LogicalArray(array) = result else {
        panic!("expected logical result, got {result:?}");
    };
    assert_eq!(array.shape, vec![0, 3]);
    assert!(array.data.is_empty());
}
