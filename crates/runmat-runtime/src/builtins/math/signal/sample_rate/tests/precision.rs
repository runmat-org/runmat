use super::*;

#[test]
fn complex_integer_samples_reject_before_lossy_materialization() {
    let storage = runmat_value::IntegerComplexStorage::new(
        IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]),
        IntegerStorage::U64(vec![7, 0]),
    )
    .expect("paired exact integer components");
    let input = Value::ComplexTensor(
        ComplexTensor::new_integer(storage, vec![1, 2]).expect("complex integer tensor"),
    );
    for up in [false, true] {
        let result = if up {
            block_on(upsample_builtin(input.clone(), Value::Num(2.0), Vec::new()))
        } else {
            block_on(downsample_builtin(
                input.clone(),
                Value::Num(2.0),
                Vec::new(),
            ))
        };
        let error = result.expect_err("integer-complex kernel is not admitted");
        assert_eq!(error.identifier(), SAMPLE_ERROR_INVALID_INPUT.identifier);
        assert!(error.message().contains("exact integer-complex kernel"));
    }
    let Value::ComplexTensor(original) = input else {
        unreachable!()
    };
    assert_eq!(
        original.integer_storage().unwrap().real(),
        &IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63])
    );
}

#[test]
fn host_sample_rate_preserves_exact_uint64_storage() {
    let input = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        vec![1, 2],
    )
    .unwrap();
    let Value::Tensor(upsampled) = call_upsample(vec![Value::Tensor(input), Value::Num(2.0)])
    else {
        panic!("expected integer upsample output");
    };
    assert_eq!(
        upsampled.integer_storage(),
        Some(&IntegerStorage::U64(vec![
            9_007_199_254_740_993,
            0,
            u64::MAX,
            0
        ]))
    );
    let Value::Tensor(downsampled) =
        call_downsample(vec![Value::Tensor(upsampled), Value::Num(2.0)])
    else {
        panic!("expected integer downsample output");
    };
    assert_eq!(
        downsampled.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
    );
}

#[test]
fn downsample_preserves_all_integer_classes_exactly() {
    macro_rules! check {
        ($variant:ident, $values:expr, $expected:expr) => {{
            let input = Value::Tensor(
                Tensor::new_integer(IntegerStorage::$variant($values), vec![1, 5]).unwrap(),
            );
            let output = call_downsample(vec![input, Value::Num(2.0), Value::Num(1.0)]);
            let Value::Tensor(output) = output else {
                panic!("expected integer tensor")
            };
            assert_eq!(
                output.integer_storage(),
                Some(&IntegerStorage::$variant($expected))
            );
            assert_eq!(output.shape(), [1, 2]);
        }};
    }
    check!(I8, vec![-1, 2, -3, 4, -5], vec![2, 4]);
    check!(I16, vec![-1, 2, -3, 4, -5], vec![2, 4]);
    check!(I32, vec![-1, 2, -3, 4, -5], vec![2, 4]);
    check!(I64, vec![-1, 2, -3, 4, -5], vec![2, 4]);
    check!(U8, vec![1, 2, 3, 4, 5], vec![2, 4]);
    check!(U16, vec![1, 2, 3, 4, 5], vec![2, 4]);
    check!(U32, vec![1, 2, 3, 4, 5], vec![2, 4]);
    check!(U64, vec![1, 2, 3, u64::MAX, 5], vec![2, u64::MAX]);
}

#[test]
fn sample_rate_preserves_complex_values() {
    let input =
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap());
    let out = call_upsample(vec![input, Value::Num(2.0)]);
    let Value::ComplexTensor(tensor) = out else {
        panic!("expected complex tensor");
    };
    assert_eq!(tensor.shape(), [1, 4]);
    assert_eq!(
        tensor.materialize_f64(),
        vec![(1.0, 2.0), (0.0, 0.0), (3.0, 4.0), (0.0, 0.0)]
    );
}

#[test]
fn resample_preserves_complex_domain() {
    let input =
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap());
    let out = call_resample(vec![
        input,
        Value::Num(2.0),
        Value::Num(1.0),
        tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
    ]);
    let Value::ComplexTensor(tensor) = out else {
        panic!("expected complex tensor");
    };
    assert_eq!(tensor.shape(), [1, 4]);
    assert_eq!(
        tensor.materialize_f64(),
        vec![(1.0, 2.0), (0.0, 0.0), (3.0, 4.0), (0.0, 0.0)]
    );
}

#[test]
fn single_complex_sample_rate_keeps_physical_class() {
    let input = Value::ComplexTensor(
        ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap(),
    );
    let up = call_upsample(vec![input.clone(), Value::Num(2.0)]);
    let Value::ComplexTensor(up) = up else {
        panic!("expected single complex tensor")
    };
    assert_eq!(up.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(
        up.materialize_f64(),
        vec![(1.0, 2.0), (0.0, 0.0), (3.0, 4.0), (0.0, 0.0)]
    );

    let down = call_downsample(vec![input.clone(), Value::Num(2.0)]);
    let Value::ComplexTensor(down) = down else {
        panic!("expected single complex tensor")
    };
    assert_eq!(down.numeric_dtype(), runmat_value::NumericDType::F32);

    let resampled = call_resample(vec![
        input,
        Value::Num(2.0),
        Value::Num(1.0),
        tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
    ]);
    let Value::ComplexTensor(resampled) = resampled else {
        panic!("expected single complex tensor")
    };
    assert_eq!(resampled.numeric_dtype(), runmat_value::NumericDType::F32);
}

#[test]
fn resample_logical_input_uses_host_double_contract() {
    let output = call_resample(vec![
        Value::Bool(true),
        Value::Num(2.0),
        Value::Num(1.0),
        tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
    ]);
    let Value::Tensor(output) = output else {
        panic!("expected real tensor")
    };
    assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
    assert_eq!(output.materialize_f64(), vec![1.0, 0.0]);
}
