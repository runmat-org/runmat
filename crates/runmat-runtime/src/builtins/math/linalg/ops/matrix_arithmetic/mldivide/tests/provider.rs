use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_round_trip_matches_cpu() {
    test_support::with_test_provider(|provider| {
        let a = Tensor::new(vec![4.0, 2.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let cpu = mldivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone()))
            .expect("cpu mldivide");
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
        let result = mldivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
            .expect("gpu mldivide");
        let Value::GpuTensor(output) = &result else {
            panic!("explicit mldivide must remain resident");
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
        let a = Tensor::new(vec![4.0, 2.0, 1.0, 3.0], vec![2, 2]).unwrap();
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

        let _ = mldivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
            .expect("gpu mldivide");

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.mldivide.count, 1);
        assert!(telemetry.upload_bytes > 0);
        assert!(telemetry.download_bytes > 0);
        assert_eq!(fallback_count(&telemetry, "mldivide:host_reupload"), 1);

        let _ = provider.free(&ha);
        let _ = provider.free(&hb);
    });
}

#[test]
fn scalar_gpu_input_falls_back_without_provider_solve_dispatch() {
    test_support::with_test_provider(|provider| {
        provider.reset_telemetry();
        let scalar = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let matrix = Tensor::new(vec![2.0, 4.0, 6.0], vec![1, 3]).unwrap();
        let hs = provider
            .upload(&HostTensorView {
                data: &scalar.materialize_f64(),
                shape: &scalar.shape,
            })
            .expect("upload scalar");
        let hm = provider
            .upload(&HostTensorView {
                data: &matrix.materialize_f64(),
                shape: &matrix.shape,
            })
            .expect("upload matrix");

        let result = mldivide_eval(&Value::GpuTensor(hs.clone()), &Value::GpuTensor(hm.clone()))
            .expect("fallback mldivide");
        let gathered = test_support::gather(result).expect("gather fallback");
        assert_eq!(gathered.materialize_f64(), vec![1.0, 2.0, 3.0]);

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.mldivide.count, 0);
        assert_eq!(fallback_count(&telemetry, "mldivide:host_reupload"), 0);
        assert!(telemetry.download_bytes > 0);

        let _ = provider.free(&hs);
        let _ = provider.free(&hm);
    });
}
