use super::*;

#[cfg(feature = "wgpu")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn wgpu_tall_path_avoids_host_reupload_fallback() {
    let _accel_guard = test_support::accel_test_lock();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => panic!("wgpu provider not available"),
    };
    if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
        return;
    }
    provider.reset_telemetry();

    let a = Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], vec![3, 2]).unwrap();
    let b = Tensor::new(vec![1.0, 2.0, 2.0], vec![3, 1]).unwrap();
    let cpu =
        mldivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone())).expect("cpu mldivide");
    let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
    provider.reset_telemetry();

    let view_a = HostTensorView {
        data: &a.materialize_f64(),
        shape: &a.shape,
    };
    let view_b = HostTensorView {
        data: &b.materialize_f64(),
        shape: &b.shape,
    };
    let ha = provider.upload(&view_a).expect("upload A");
    let hb = provider.upload(&view_b).expect("upload B");
    let gpu_value = mldivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
        .expect("gpu mldivide");
    let gathered = test_support::gather(gpu_value).expect("gather");
    let _ = provider.free(&ha);
    let _ = provider.free(&hb);

    assert_eq!(gathered.shape, cpu_tensor.shape);
    for (gpu, cpu) in gathered
        .materialize_f64()
        .iter()
        .zip(cpu_tensor.materialize_f64().iter())
    {
        assert!((gpu - cpu).abs() < 1e-4, "gpu={gpu} cpu={cpu}");
    }

    let telemetry = provider.telemetry_snapshot();
    assert_eq!(telemetry.mldivide.count, 1);
    assert_eq!(fallback_count(&telemetry, "mldivide:host_reupload"), 0);
}

#[cfg(feature = "wgpu")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn wgpu_square_path_avoids_host_reupload_fallback() {
    let _accel_guard = test_support::accel_test_lock();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => panic!("wgpu provider not available"),
    };
    if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
        return;
    }
    provider.reset_telemetry();

    let a = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![7.0, 8.0], vec![2, 1]).unwrap();
    let cpu =
        mldivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone())).expect("cpu mldivide");
    let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
    provider.reset_telemetry();

    let view_a = HostTensorView {
        data: &a.materialize_f64(),
        shape: &a.shape,
    };
    let view_b = HostTensorView {
        data: &b.materialize_f64(),
        shape: &b.shape,
    };
    let ha = provider.upload(&view_a).expect("upload A");
    let hb = provider.upload(&view_b).expect("upload B");
    let gpu_value = mldivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
        .expect("gpu mldivide");
    let gathered = test_support::gather(gpu_value).expect("gather");
    let _ = provider.free(&ha);
    let _ = provider.free(&hb);

    assert_eq!(gathered.shape, cpu_tensor.shape);
    for (gpu, cpu) in gathered
        .materialize_f64()
        .iter()
        .zip(cpu_tensor.materialize_f64().iter())
    {
        assert!((gpu - cpu).abs() < 1e-4, "gpu={gpu} cpu={cpu}");
    }

    let telemetry = provider.telemetry_snapshot();
    assert_eq!(telemetry.mldivide.count, 1);
    assert_eq!(fallback_count(&telemetry, "mldivide:host_reupload"), 0);
}

#[cfg(feature = "wgpu")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn wgpu_round_trip_matches_cpu() {
    let _accel_guard = test_support::accel_test_lock();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => panic!("wgpu provider not available"),
    };
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-10,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-4,
    };

    let a = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    let cpu =
        mldivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone())).expect("cpu mldivide");
    let cpu_tensor = test_support::gather(cpu).expect("cpu gather");

    let view_a = HostTensorView {
        data: &a.materialize_f64(),
        shape: &a.shape,
    };
    let view_b = HostTensorView {
        data: &b.materialize_f64(),
        shape: &b.shape,
    };
    let ha = provider.upload(&view_a).expect("upload A");
    let hb = provider.upload(&view_b).expect("upload B");
    let gpu_value = mldivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
        .expect("gpu mldivide");
    let gathered = test_support::gather(gpu_value).expect("gather");
    let _ = provider.free(&ha);
    let _ = provider.free(&hb);

    assert_eq!(gathered.shape, cpu_tensor.shape);
    for (gpu, cpu) in gathered
        .materialize_f64()
        .iter()
        .zip(cpu_tensor.materialize_f64().iter())
    {
        assert!((gpu - cpu).abs() < tol, "gpu={gpu} cpu={cpu}");
    }
}
