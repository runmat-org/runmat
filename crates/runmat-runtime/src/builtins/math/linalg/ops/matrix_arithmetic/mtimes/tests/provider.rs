use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_scalar_matrix_product() {
    test_support::with_test_provider(|provider| {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &matrix.materialize_f64(),
            shape: &matrix.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result =
            mtimes_builtin(Value::Num(2.0), Value::GpuTensor(handle)).expect("gpu scalar matmul");
        let gathered = match result {
            Value::GpuTensor(out) => test_support::gather(Value::GpuTensor(out)).expect("gather"),
            other => panic!("expected gpu tensor, got {other:?}"),
        };
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(gathered.materialize_f64(), vec![2.0, 4.0, 6.0, 8.0]);
    });
}

#[test]
fn gpu_nonscalar_floating_partner_rejects_typed_integer_scalar() {
    test_support::with_test_provider(|provider| {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &matrix.materialize_f64(),
            shape: &matrix.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let scalar = Tensor::new_integer(IntegerStorage::U8(vec![3]), vec![1, 1]).expect("scalar");

        let error = mtimes_builtin(Value::Tensor(scalar), Value::GpuTensor(handle))
            .expect_err("nonscalar floating GPU partner must reject");
        assert_eq!(error.identifier(), MTIMES_ERROR_INVALID_INPUT.identifier);
        assert!(error
            .message()
            .contains("integer arrays can only be combined with scalar double values"));
    });
}

#[test]
fn resident_integer_scalar_mtimes_preserves_all_classes_and_residency() {
    test_support::with_test_provider(|provider| {
        for (array, scalar, expected) in integer_scalar_mtimes_cases() {
            let array = Tensor::new_integer(array, vec![1, 3]).expect("resident integer array");
            let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
            let result = mtimes_builtin(Value::GpuTensor(array_handle), Value::Int(scalar.clone()))
                .expect("resident integer array times host scalar");
            let Value::GpuTensor(result_handle) = &result else {
                panic!("expected resident integer result, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(result_handle),
                Some(integer_element_type(&expected))
            );
            let gathered = test_support::gather(result).expect("gather integer result");
            assert_eq!(gathered.integer_storage(), Some(&expected));

            let scalar_tensor =
                Tensor::new_integer(IntegerStorage::from_scalar(scalar), vec![1, 1])
                    .expect("resident scalar");
            let scalar_handle =
                gpu_helpers::upload_tensor(provider, &scalar_tensor).expect("upload scalar");
            let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
            let result = mtimes_builtin(
                Value::GpuTensor(scalar_handle),
                Value::GpuTensor(array_handle),
            )
            .expect("resident scalar times resident integer array");
            let Value::GpuTensor(result_handle) = &result else {
                panic!("expected resident integer result, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(result_handle),
                Some(integer_element_type(&expected))
            );
            let gathered = test_support::gather(result).expect("gather integer result");
            assert_eq!(gathered.integer_storage(), Some(&expected));
        }

        let scalar =
            Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1, 1]).expect("scalar");
        let scalar_handle = gpu_helpers::upload_tensor(provider, &scalar).expect("upload");
        let array =
            Tensor::new_integer(IntegerStorage::U16(vec![3, 4]), vec![1, 2]).expect("array");
        let result = mtimes_builtin(Value::GpuTensor(scalar_handle), Value::Tensor(array))
            .expect("singleton-N-D resident scalar mtimes");
        let gathered = test_support::gather(result).expect("gather integer result");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U16(vec![6, 8]))
        );
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn wgpu_integer_scalar_mtimes_preserves_all_classes_and_residency() {
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    for (array, scalar, expected) in integer_scalar_mtimes_cases() {
        let array = Tensor::new_integer(array, vec![1, 3]).expect("wgpu integer array");
        let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
        let result = mtimes_builtin(Value::GpuTensor(array_handle), Value::Int(scalar.clone()))
            .expect("wgpu integer scalar mtimes");
        let Value::GpuTensor(result_handle) = &result else {
            panic!("expected resident integer result, got {result:?}");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(result_handle),
            Some(integer_element_type(&expected))
        );
        let gathered = test_support::gather(result).expect("gather wgpu integer result");
        assert_eq!(gathered.integer_storage(), Some(&expected));

        let scalar =
            Tensor::new_integer(IntegerStorage::from_scalar(scalar), vec![1, 1]).expect("scalar");
        let scalar_handle = gpu_helpers::upload_tensor(provider, &scalar).expect("upload");
        let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
        let result = mtimes_builtin(
            Value::GpuTensor(scalar_handle),
            Value::GpuTensor(array_handle),
        )
        .expect("wgpu scalar times integer array");
        let Value::GpuTensor(result_handle) = &result else {
            panic!("expected resident integer result, got {result:?}");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(result_handle),
            Some(integer_element_type(&expected))
        );
        let gathered = test_support::gather(result).expect("gather wgpu integer result");
        assert_eq!(gathered.integer_storage(), Some(&expected));
    }
}
