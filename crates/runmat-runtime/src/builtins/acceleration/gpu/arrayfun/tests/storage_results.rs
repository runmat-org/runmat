use super::*;

#[test]
fn arrayfun_preserves_native_single_and_complex_single_scalars() {
    let input = ArrayData::Tensor(Tensor::from_f32(vec![0.25, -1.5], vec![1, 2]).expect("single"));
    let first = input.value_at(0).expect("first single");
    let Value::Tensor(first) = first else {
        panic!("single element must remain a scalar tensor");
    };
    assert_eq!(first.as_f32_slice(), Some(&[0.25][..]));

    let mut real_collector = UniformCollector::Pending;
    real_collector
        .push(&Value::Tensor(first))
        .expect("single result");
    real_collector
        .push(&Value::Tensor(
            Tensor::from_f32(vec![-1.5], vec![1, 1]).expect("single scalar"),
        ))
        .expect("second single result");
    let Value::Tensor(real) = real_collector.finish(&[1, 2]).expect("single output") else {
        panic!("expected native single output");
    };
    assert_eq!(real.as_f32_slice(), Some(&[0.25, -1.5][..]));

    let mut complex_collector = UniformCollector::Pending;
    for value in [(1.25_f32, -2.5_f32), (3.5_f32, 4.75_f32)] {
        complex_collector
            .push(&Value::ComplexTensor(
                ComplexTensor::from_f32(vec![value], vec![1, 1]).expect("complex single scalar"),
            ))
            .expect("complex single result");
    }
    let Value::ComplexTensor(complex) = complex_collector.finish(&[1, 2]).expect("complex output")
    else {
        panic!("expected complex single output");
    };
    assert_eq!(
        complex.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(1.25, -2.5), (3.5, 4.75)])
    );
}

#[test]
fn uniform_collector_preserves_exact_complex_integer_storage() {
    let mut collector = UniformCollector::Pending;
    for (real, imag) in [
        (
            IntValue::I64(9_007_199_254_740_993),
            IntValue::I64(-9_007_199_254_740_993),
        ),
        (IntValue::I64(i64::MAX), IntValue::I64(i64::MIN)),
    ] {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::from_scalar(real),
            IntegerStorage::from_scalar(imag),
        )
        .expect("integer complex scalar");
        collector
            .push(&Value::ComplexTensor(
                ComplexTensor::new_integer(storage, vec![1, 1]).expect("integer complex tensor"),
            ))
            .expect("integer complex result");
    }
    let Value::ComplexTensor(output) = collector.finish(&[1, 2]).expect("integer complex output")
    else {
        panic!("expected integer complex tensor");
    };
    let storage = output.integer_storage().expect("exact component storage");
    assert_eq!(
        storage.real,
        IntegerStorage::I64(vec![9_007_199_254_740_993, i64::MAX])
    );
    assert_eq!(
        storage.imag,
        IntegerStorage::I64(vec![-9_007_199_254_740_993, i64::MIN])
    );
}

#[test]
fn uniform_collector_rejects_different_noninteger_classes() {
    for second in [
        Value::Num(1.0),
        Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single")),
        Value::CharArray(CharArray::new(vec!['1'], 1, 1).expect("char")),
    ] {
        let mut collector = UniformCollector::Pending;
        collector.push(&Value::Bool(true)).expect("logical");
        let error = collector
            .push(&second)
            .expect_err("heterogeneous uniform result must reject");
        assert_eq!(
            error.identifier(),
            ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE.identifier
        );
    }
}

#[test]
fn arrayfun_rejects_every_typed_integer_uniform_output_control() {
    let input = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("input"));
    for storage in [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ] {
        for control in [
            Value::Int(storage.value_at(0).expect("scalar")),
            Value::Tensor(Tensor::new_integer(storage.clone(), vec![1, 1]).expect("typed control")),
        ] {
            let error = call(
                Value::FunctionHandle("sin".to_string()),
                vec![input.clone(), Value::from("UniformOutput"), control],
            )
            .expect_err("typed integer control must reject");
            assert_eq!(
                error.identifier(),
                ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION.identifier
            );
        }
    }
}
