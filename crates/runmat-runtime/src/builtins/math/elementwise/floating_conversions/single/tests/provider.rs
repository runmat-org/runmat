use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_gpu_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = single_builtin(Value::GpuTensor(handle), Vec::new()).expect("single");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                let expected: Vec<f64> = [0.1f32, 0.2, 0.3, 0.4]
                    .into_iter()
                    .map(|v| v as f64)
                    .collect();
                assert_eq!(gathered.shape, vec![2, 2]);
                assert!(gathered.as_f32_slice().is_some());
                assert_eq!(gathered.materialize_f64(), expected);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_like_host_prototype() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = single_builtin(
        Value::Tensor(tensor.clone()),
        vec![Value::from("like"), Value::Num(0.0)],
    )
    .expect("single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, tensor.shape);
            let expected: Vec<f64> = [1.0f32, 2.0, 3.0, 4.0]
                .into_iter()
                .map(|v| v as f64)
                .collect();
            assert_eq!(t.materialize_f64(), expected);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_like_requires_runmat_compatibility_mode() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = single_builtin(
        Value::Int(IntValue::U64(u64::MAX)),
        vec![Value::from("like"), Value::Num(0.0)],
    )
    .expect_err("single like extension must reject in strict mode");
    assert_eq!(
        error.identifier(),
        SINGLE_LIKE_OUTPUT_EXTENSION.error_identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_like_gpu_prototype() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2]).unwrap();
        let proto_view = HostTensorView {
            data: &[0.0],
            shape: &[1, 1],
        };
        let proto_handle = provider.upload(&proto_view).expect("upload");
        let result = single_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("like"), Value::GpuTensor(proto_handle)],
        )
        .expect("single");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                let expected: Vec<f64> = [0.5f32, 1.5, 2.5, 3.5]
                    .into_iter()
                    .map(|v| v as f64)
                    .collect();
                assert_eq!(gathered.shape, vec![2, 2]);
                assert_eq!(gathered.materialize_f64(), expected);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    });
}

#[test]
fn single_like_validates_physical_output_and_protects_the_prototype() {
    test_support::with_test_provider(|provider| {
        let make_output = || {
            let tensor = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).expect("single tensor");
            gpu_helpers::upload_tensor(provider, &tensor).expect("single upload")
        };
        let prototype = {
            let tensor = Tensor::from_f32(vec![0.0], vec![1, 1]).expect("prototype tensor");
            gpu_helpers::upload_tensor(provider, &tensor).expect("prototype upload")
        };

        let valid = make_output();
        assert!(valid_single_like_output(
            &valid,
            &prototype,
            provider,
            &[2, 1],
        ));
        provider.free(&valid).expect("free valid output");

        let mut wrong_shape = make_output();
        wrong_shape.shape = vec![1, 2];
        assert!(!valid_single_like_output(
            &wrong_shape,
            &prototype,
            provider,
            &[2, 1],
        ));
        provider.free(&wrong_shape).expect("free wrong shape");

        let mut wrong_precision = make_output();
        wrong_precision.descriptor.element_type =
            Some(runmat_accelerate_api::NumericElementType::F64);
        assert!(!valid_single_like_output(
            &wrong_precision,
            &prototype,
            provider,
            &[2, 1],
        ));
        provider
            .free(&wrong_precision)
            .expect("free wrong precision");

        let mut wrong_storage = make_output();
        wrong_storage.descriptor.storage =
            Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
        assert!(!valid_single_like_output(
            &wrong_storage,
            &prototype,
            provider,
            &[2, 1],
        ));
        provider.free(&wrong_storage).expect("free wrong storage");

        assert!(!valid_single_like_output(
            &prototype,
            &prototype,
            provider,
            &[1, 1],
        ));
        free_rejected_single_handle(&prototype, &[&prototype]);
        assert!(block_on(provider.download_numeric(&prototype)).is_ok());
        provider.free(&prototype).expect("free prototype");
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_like_case_insensitive() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new(vec![1.5, 2.25], vec![2, 1]).unwrap();
    let result = single_builtin(
        Value::Tensor(tensor.clone()),
        vec![
            Value::CharArray(CharArray::new_row("LIKE")),
            Value::Num(0.0),
        ],
    )
    .expect("single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, tensor.shape);
            let expected: Vec<f64> = [1.5f32, 2.25f32].into_iter().map(|v| v as f64).collect();
            assert_eq!(t.materialize_f64(), expected);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}
