use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_round_trip_matches_cpu() {
    test_support::with_test_provider(|provider| {
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let cpu = mrdivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone()))
            .expect("cpu mrdivide");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");

        let view_a = HostTensorView {
            data: &a.materialize_f64(),
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.materialize_f64(),
            shape: &b.shape,
        };
        let ha = provider
            .upload(&view_a)
            .expect("upload A")
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let hb = provider.upload(&view_b).expect("upload B");
        let result = mrdivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
            .expect("gpu mrdivide");
        let Value::GpuTensor(output) = &result else {
            panic!("explicit mrdivide must remain resident");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(output));
        let gathered = test_support::gather(result).expect("gather");
        let _ = provider.free(&ha);
        let _ = provider.free(&hb);

        assert_eq!(gathered.shape, cpu_tensor.shape);
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-12);
        }
    });
}

#[test]
fn provider_telemetry_records_gpu_host_reupload_path() {
    test_support::with_test_provider(|provider| {
        provider.reset_telemetry();
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: &a.shape,
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: &b.shape,
            })
            .expect("upload B");

        let _ = mrdivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
            .expect("gpu mrdivide");

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.mrdivide.count, 1);
        assert!(telemetry.upload_bytes > 0);
        assert!(telemetry.download_bytes > 0);
        assert_eq!(fallback_count(&telemetry, "mrdivide:host_reupload"), 1);

        let _ = provider.free(&ha);
        let _ = provider.free(&hb);
    });
}

#[test]
fn scalar_gpu_rhs_falls_back_without_provider_solve_dispatch() {
    test_support::with_test_provider(|provider| {
        provider.reset_telemetry();
        let matrix = Tensor::new(vec![2.0, 4.0, 6.0], vec![1, 3]).unwrap();
        let scalar = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let hm = provider
            .upload(&HostTensorView {
                data: &matrix.materialize_f64(),
                shape: &matrix.shape,
            })
            .expect("upload matrix");
        let hs = provider
            .upload(&HostTensorView {
                data: &scalar.materialize_f64(),
                shape: &scalar.shape,
            })
            .expect("upload scalar");

        let result = mrdivide_eval(&Value::GpuTensor(hm.clone()), &Value::GpuTensor(hs.clone()))
            .expect("fallback mrdivide");
        let gathered = test_support::gather(result).expect("gather fallback");
        assert_eq!(gathered.materialize_f64(), vec![1.0, 2.0, 3.0]);

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.mrdivide.count, 0);
        assert_eq!(fallback_count(&telemetry, "mrdivide:host_reupload"), 0);
        assert!(telemetry.download_bytes > 0);

        let _ = provider.free(&hm);
        let _ = provider.free(&hs);
    });
}

#[test]
fn resident_integer_scalar_mrdivide_preserves_all_classes_and_residency() {
    test_support::with_test_provider(|provider| {
        for (array, scalar, expected) in integer_scalar_mrdivide_cases() {
            let array = Tensor::new_integer(array, vec![1, 3]).expect("resident integer array");
            let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
            let result =
                mrdivide_builtin(Value::GpuTensor(array_handle), Value::Int(scalar.clone()))
                    .expect("resident integer array divided by host scalar");
            let Value::GpuTensor(result_handle) = &result else {
                panic!("expected resident integer result, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(result_handle),
                Some(integer_element_type(&expected))
            );
            let gathered = test_support::gather(result).expect("gather integer result");
            assert_eq!(gathered.integer_storage(), Some(&expected));

            let scalar = Tensor::new_integer(IntegerStorage::from_scalar(scalar), vec![1, 1])
                .expect("resident scalar");
            let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
            let scalar_handle = gpu_helpers::upload_tensor(provider, &scalar).expect("upload");
            let result = mrdivide_builtin(
                Value::GpuTensor(array_handle),
                Value::GpuTensor(scalar_handle),
            )
            .expect("resident integer array divided by resident scalar");
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

        let array =
            Tensor::new_integer(IntegerStorage::U16(vec![6, 4]), vec![1, 2]).expect("array");
        let scalar =
            Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1, 1]).expect("scalar");
        let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
        let scalar_handle = gpu_helpers::upload_tensor(provider, &scalar).expect("upload");
        let result = mrdivide_builtin(
            Value::GpuTensor(array_handle),
            Value::GpuTensor(scalar_handle),
        )
        .expect("singleton-N-D resident scalar mrdivide");
        let gathered = test_support::gather(result).expect("gather integer result");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U16(vec![3, 2]))
        );
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn wgpu_integer_scalar_mrdivide_preserves_all_classes_and_residency() {
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    for (array, scalar, expected) in integer_scalar_mrdivide_cases() {
        let array = Tensor::new_integer(array, vec![1, 3]).expect("wgpu integer array");
        let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
        let result = mrdivide_builtin(Value::GpuTensor(array_handle), Value::Int(scalar.clone()))
            .expect("wgpu integer scalar mrdivide");
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
        let array_handle = gpu_helpers::upload_tensor(provider, &array).expect("upload");
        let scalar_handle = gpu_helpers::upload_tensor(provider, &scalar).expect("upload");
        let result = mrdivide_builtin(
            Value::GpuTensor(array_handle),
            Value::GpuTensor(scalar_handle),
        )
        .expect("wgpu integer array divided by resident scalar");
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
