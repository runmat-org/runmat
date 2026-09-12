use super::super::entrypoints::movmad_builtin;
use super::*;

#[test]
fn movmad_computes_centered_median_absolute_deviation() {
    let result = block_on(movmad_builtin(
        tensor(vec![1.0, 2.0, 100.0, 4.0, 5.0], vec![5, 1]),
        Value::Num(3.0),
        Vec::new(),
    ))
    .unwrap();
    assert!(matches!(result, Value::Tensor(t) if t.materialize_f64()[2] == 2.0));
}

#[test]
fn movmad_reads_typed_integer_storage_and_returns_double_output() {
    let input = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 100, 4, 5]), vec![5, 1])
        .expect("integer movmad input");

    let result = block_on(movmad_builtin(
        Value::Tensor(input),
        Value::Num(3.0),
        Vec::new(),
    ))
    .unwrap();

    match result {
        Value::Tensor(tensor) => {
            assert!(tensor.integer_storage().is_none());
            assert_eq!(tensor.materialize_f64(), vec![0.5, 1.0, 2.0, 1.0, 0.5]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn movmad_accepts_all_integer_classes_into_double_output() {
    let storages = vec![
        IntegerStorage::I8(vec![1, 2, 5]),
        IntegerStorage::I16(vec![1, 2, 5]),
        IntegerStorage::I32(vec![1, 2, 5]),
        IntegerStorage::I64(vec![1, 2, 5]),
        IntegerStorage::U8(vec![1, 2, 5]),
        IntegerStorage::U16(vec![1, 2, 5]),
        IntegerStorage::U32(vec![1, 2, 5]),
        IntegerStorage::U64(vec![1, 2, 5]),
    ];
    for storage in storages {
        let input = Tensor::new_integer(storage, vec![3, 1]).unwrap();
        let result = block_on(movmad_builtin(
            Value::Tensor(input),
            Value::Num(3.0),
            Vec::new(),
        ))
        .unwrap();
        assert!(
            matches!(result, Value::Tensor(tensor) if tensor.integer_storage().is_none() && tensor.materialize_f64() == vec![0.5, 1.0, 1.5])
        );
    }
}

#[test]
#[cfg(feature = "wgpu")]
fn movmad_gpu_large_window_follows_compatibility_mode() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new((1..=32).map(f64::from).collect(), vec![32, 1]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input).unwrap();
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(movmad_builtin(
                Value::GpuTensor(handle.clone()),
                Value::Num(32.0),
                Vec::new(),
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:MovmadGpuLargeWindowExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let result = block_on(movmad_builtin(
                Value::GpuTensor(handle.clone()),
                Value::Num(32.0),
                Vec::new(),
            ))
            .unwrap();
            assert!(matches!(result, Value::GpuTensor(_)));
        }
        let _ = provider.free(&handle);
    });
}
