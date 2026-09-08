use super::*;

#[test]
fn arrayfun_gpu_options_are_mode_gated() {
    test_support::with_test_provider(|provider| {
        let handle = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![0.0, 1.0], vec![2, 1]).expect("input"),
        )
        .expect("upload");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call(
            Value::FunctionHandle("sin".to_string()),
            vec![
                Value::GpuTensor(handle.clone()),
                Value::from("UniformOutput"),
                Value::Bool(false),
            ],
        )
        .expect_err("gpu options must reject in compatible mode");
        assert_eq!(
            error.identifier(),
            ARRAYFUN_GPU_OPTIONS_EXTENSION.error_identifier
        );
        let _ = provider.free(&handle);
    });
}

#[test]
fn arrayfun_provider_fallback_preserves_every_integer_class() {
    test_support::with_test_provider(|provider| {
        for (storage, callback) in [
            (IntegerStorage::I8(vec![i8::MIN, i8::MAX]), "int8"),
            (IntegerStorage::I16(vec![i16::MIN, i16::MAX]), "int16"),
            (IntegerStorage::I32(vec![i32::MIN, i32::MAX]), "int32"),
            (IntegerStorage::I64(vec![i64::MIN, i64::MAX]), "int64"),
            (IntegerStorage::U8(vec![0, u8::MAX]), "uint8"),
            (IntegerStorage::U16(vec![0, u16::MAX]), "uint16"),
            (IntegerStorage::U32(vec![0, u32::MAX]), "uint32"),
            (
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                "uint64",
            ),
        ] {
            let input = Tensor::new_integer(storage.clone(), vec![1, 2]).expect("input");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let result = call(
                Value::FunctionHandle(callback.to_string()),
                vec![Value::GpuTensor(handle.clone())],
            )
            .expect("provider arrayfun");
            let Value::GpuTensor(output) = result else {
                panic!("expected resident output");
            };
            let gathered =
                test_support::gather(Value::GpuTensor(output.clone())).expect("gather output");
            assert_eq!(gathered.integer_storage(), Some(&storage));
            let _ = provider.free(&handle);
            let _ = provider.free(&output);
        }
    });
}

#[test]
fn arrayfun_gpu_overload_uses_documented_compatible_size_expansion_exactly() {
    test_support::with_test_provider(|provider| {
        let row_storage = IntegerStorage::U64(vec![
            9_007_199_254_740_993,
            9_007_199_254_740_994,
            9_007_199_254_740_995,
        ]);
        let row = Tensor::new_integer(row_storage, vec![1, 3]).expect("row");
        let row_handle = gpu_helpers::upload_tensor(provider, &row).expect("upload row");
        let column =
            Tensor::new_integer(IntegerStorage::U64(vec![10, 20]), vec![2, 1]).expect("column");
        let result = call(
            Value::FunctionHandle("plus".to_string()),
            vec![Value::GpuTensor(row_handle.clone()), Value::Tensor(column)],
        )
        .expect("compatible gpu arrayfun");
        let Value::GpuTensor(output) = result else {
            panic!("expected resident output");
        };
        let gathered = test_support::gather(Value::GpuTensor(output.clone())).expect("gather");
        assert_eq!(gathered.shape, vec![2, 3]);
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_741_003,
                9_007_199_254_741_013,
                9_007_199_254_741_004,
                9_007_199_254_741_014,
                9_007_199_254_741_005,
                9_007_199_254_741_015,
            ]))
        );
        let _ = provider.free(&row_handle);
        let _ = provider.free(&output);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_uniform_false_gpu_returns_cell() {
    test_support::with_test_provider(|provider| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = call(
            Value::FunctionHandle("sin".to_string()),
            vec![
                Value::GpuTensor(handle),
                Value::String("UniformOutput".into()),
                Value::Bool(false),
            ],
        )
        .expect("arrayfun");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 2);
                assert_eq!(cell.cols, 1);
                let first = cell.get(0, 0).expect("first cell");
                let second = cell.get(1, 0).expect("second cell");
                match (first, second) {
                    (Value::Num(a), Value::Num(b)) => {
                        assert!((a - 0.0f64.sin()).abs() < 1e-12);
                        assert!((b - 1.0f64.sin()).abs() < 1e-12);
                    }
                    other => panic!("expected numeric cells, got {other:?}"),
                }
            }
            other => panic!("expected cell, got {other:?}"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn arrayfun_gpu_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = call(
            Value::FunctionHandle("sin".to_string()),
            vec![Value::GpuTensor(handle)],
        )
        .expect("arrayfun");
        match result {
            Value::GpuTensor(gpu) => {
                let gathered = test_support::gather(Value::GpuTensor(gpu.clone())).unwrap();
                let expected: Vec<f64> = values(&tensor).into_iter().map(f64::sin).collect();
                assert_eq!(values(&gathered), expected);
                let _ = provider.free(&gpu);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    });
}
