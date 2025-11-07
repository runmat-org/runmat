use runmat_builtins::{CharArray, NumericDType, Tensor, Value};

#[test]
fn zeros_single_uses_f32_dtype() {
    let result = runmat_runtime::call_builtin(
        "zeros",
        &[
            Value::Num(2.0),
            Value::Num(3.0),
            Value::String("single".into()),
        ],
    )
    .expect("zeros single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 3]);
            assert_eq!(t.dtype, NumericDType::F32);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn ones_single_uses_f32_dtype() {
    let result = runmat_runtime::call_builtin(
        "ones",
        &[
            Value::Num(3.0),
            Value::Num(4.0),
            Value::String("single".into()),
        ],
    )
    .expect("ones single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![3, 4]);
            assert_eq!(t.dtype, NumericDType::F32);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn zeros_like_proto_preserves_numeric_dtype() {
    let proto = Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::F32)
        .expect("proto tensor");
    let result = runmat_runtime::call_builtin(
        "zeros",
        &[Value::String("like".into()), Value::Tensor(proto.clone())],
    )
    .expect("zeros like proto");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, proto.shape);
            assert_eq!(t.dtype, NumericDType::F32);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn randn_single_sets_f32_dtype() {
    let result = runmat_runtime::call_builtin(
        "randn",
        &[
            Value::Num(4.0),
            Value::Num(5.0),
            Value::String("single".into()),
        ],
    )
    .expect("randn single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![4, 5]);
            assert_eq!(t.dtype, NumericDType::F32);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn randn_like_proto_preserves_dtype() {
    let proto = Tensor::new_with_dtype(vec![0.0, 0.0, 0.0], vec![3, 1], NumericDType::F32)
        .expect("proto tensor");
    let result = runmat_runtime::call_builtin(
        "randn",
        &[Value::String("like".into()), Value::Tensor(proto.clone())],
    )
    .expect("randn like");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, proto.shape);
            assert_eq!(t.dtype, NumericDType::F32);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn gpu_array_single_roundtrip_preserves_dtype() {
    runmat_accelerate::simple_provider::register_inprocess_provider();
    let host = Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::F32)
        .expect("host tensor");
    let gpu = runmat_runtime::call_builtin(
        "gpuArray",
        &[
            Value::Tensor(host.clone()),
            Value::CharArray(CharArray::new_row("single")),
        ],
    )
    .expect("gpuArray single upload");
    if let Value::GpuTensor(ref handle) = gpu {
        let expected_handle_precision = runmat_accelerate_api::ProviderPrecision::F32;
        let precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or(expected_handle_precision);
        assert_eq!(precision, expected_handle_precision);
        let expected_dtype = match precision {
            runmat_accelerate_api::ProviderPrecision::F32 => NumericDType::F32,
            runmat_accelerate_api::ProviderPrecision::F64 => NumericDType::F64,
        };
        let gathered = runmat_runtime::dispatcher::gather_if_needed(&gpu).expect("gather single");
        match gathered {
            Value::Tensor(t) => {
                assert_eq!(t.shape, host.shape);
                assert_eq!(t.dtype, expected_dtype);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        
        let direct_eval = runmat_runtime::builtins::acceleration::gpu::gather::evaluate(&[gpu.clone()])
            .expect("gather eval")
            .into_first();
        match direct_eval {
            Value::Tensor(t) => {
                assert_eq!(t.shape, host.shape);
                assert_eq!(t.dtype, expected_dtype);
            }
            other => panic!("expected tensor from gather::evaluate, got {other:?}"),
        }

        let builtin_gathered =
            runmat_runtime::call_builtin("gather", &[gpu.clone()]).expect("gather builtin");
        match builtin_gathered {
            Value::Tensor(t) => {
                assert_eq!(t.shape, host.shape);
                assert_eq!(t.dtype, expected_dtype);
            }
            other => panic!("expected tensor result from builtin gather, got {other:?}"),
        }
    } else {
        panic!("expected gpu tensor");
    }
}
