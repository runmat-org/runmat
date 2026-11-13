use super::*;
use crate::builtins::common::test_support;
use runmat_accelerate_api::AccelProvider;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filter_gpu_matches_cpu() {
    test_support::with_test_provider(|provider| {
        let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x = Tensor::new(vec![1.0, 5.0, 2.0, 0.0, 3.0], vec![1, 5]).unwrap();

        let view = runmat_accelerate_api::HostTensorView {
            data: &x.materialize_f64(),
            shape: x.shape(),
        };
        let x_gpu = provider.upload(&view).expect("upload signal");

        let eval_gpu = evaluate(
            Value::Tensor(b.clone()),
            Value::Tensor(a.clone()),
            Value::GpuTensor(x_gpu),
            &[],
        )
        .expect("gpu filter");
        let (y_gpu, _) = eval_gpu.into_pair();
        let gathered = test_support::gather(y_gpu).expect("gather");

        let eval_cpu = evaluate(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x), &[])
            .expect("cpu filter");
        let (y_cpu, _) = eval_cpu.into_pair();
        let (cpu_data, _) = real_tensor_parts(y_cpu, "output");

        approx_eq_slice(&gathered.materialize_f64(), &cpu_data);
    });
}

#[test]
fn resident_filter_does_not_download_the_signal_before_provider_execution() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &[4, 1],
        })
        .expect("resident signal");
    provider.reset_telemetry();
    let result = evaluate(
        Value::Num(1.0),
        Value::Num(1.0),
        Value::GpuTensor(signal.clone()),
        &[],
    )
    .expect("identity filter");
    let (output, _) = result.into_pair();
    let Value::GpuTensor(output) = output else {
        panic!("supported filter route must retain the resident output")
    };
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert!(!output.shares_allocation_with(&signal));
    assert_eq!(
        block_on(provider.download(&output)).unwrap().data,
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert!(signal.allocation_lease().validate_live().is_ok());
}

#[test]
fn resident_filter_nonleading_axis_preserves_order_first_state_geometry() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            shape: &[3, 4],
        })
        .expect("resident matrix");
    provider.reset_telemetry();
    let (output, final_state) = evaluate(
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        Value::GpuTensor(signal.clone()),
        &[
            Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap()),
            Value::Num(2.0),
        ],
    )
    .expect("resident nonleading filter")
    .into_pair();
    let (Value::GpuTensor(output), Value::GpuTensor(final_state)) = (output, final_state) else {
        panic!("supported matrix filter must retain output and state on device")
    };
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(final_state.shape(), &[1, 3]);
    assert_eq!(
        block_on(provider.download(&output)).unwrap().data,
        vec![11.0, 22.0, 33.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
    assert_eq!(
        block_on(provider.download(&final_state)).unwrap().data,
        vec![0.0; 3]
    );
    assert!(signal.allocation_lease().validate_live().is_ok());
}

#[test]
fn resident_filter_third_axis_uses_order_first_state_geometry() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
            shape: &[2, 3, 4],
        })
        .expect("resident three-dimensional signal");
    provider.reset_telemetry();
    let (output, final_state) = evaluate(
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        Value::GpuTensor(signal.clone()),
        &[
            Value::Tensor(
                Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![1, 2, 3]).unwrap(),
            ),
            Value::Num(3.0),
        ],
    )
    .expect("resident third-axis filter")
    .into_pair();
    let (Value::GpuTensor(output), Value::GpuTensor(final_state)) = (output, final_state) else {
        panic!("supported third-axis filter must retain output and state on device")
    };
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(final_state.shape(), &[1, 2, 3]);
    assert_eq!(
        block_on(provider.download(&output)).unwrap().data,
        vec![
            11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ]
    );
    assert_eq!(
        block_on(provider.download(&final_state)).unwrap().data,
        vec![0.0; 6]
    );
    assert!(signal.allocation_lease().validate_live().is_ok());
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires a live WGPU adapter; run explicitly for hardware qualification"]
fn filter_wgpu_matches_cpu() {
    let _serial = test_support::accel_test_lock();
    let _restore = runmat_accelerate_api::ThreadProviderGuard::set(None);
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("live WGPU adapter required");

    let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::new(vec![1.0, 5.0, 2.0, 0.0, 3.0], vec![1, 5]).unwrap();

    let cpu_eval = evaluate(
        Value::Tensor(b.clone()),
        Value::Tensor(a.clone()),
        Value::Tensor(x.clone()),
        &[],
    )
    .expect("cpu filter");
    let (cpu_value, _) = cpu_eval.into_pair();
    let cpu_tensor = match cpu_value {
        Value::Tensor(t) => t,
        other => panic!("expected tensor output, got {other:?}"),
    };

    let view = runmat_accelerate_api::HostTensorView {
        data: &x.materialize_f64(),
        shape: x.shape(),
    };
    let x_gpu = provider.upload(&view).expect("upload signal");
    let borrowed_signal = x_gpu.clone();
    provider.reset_telemetry();

    let gpu_eval = evaluate(
        Value::Tensor(b),
        Value::Tensor(a),
        Value::GpuTensor(x_gpu),
        &[],
    )
    .expect("wgpu filter");
    let (gpu_value, _) = gpu_eval.into_pair();
    assert!(matches!(&gpu_value, Value::GpuTensor(_)));
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    let gathered = test_support::gather(gpu_value).expect("gather");

    approx_eq_slice(&gathered.materialize_f64(), &cpu_tensor.materialize_f64());
    assert!(borrowed_signal.allocation_lease().validate_live().is_ok());
}
