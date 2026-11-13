use super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_gpu_roundtrip_mtimes() {
    test_support::with_test_provider(|provider| {
        let tensor =
            Tensor::new(vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0], vec![2, 2, 2]).unwrap();
        let identity = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let view_lhs = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let view_rhs = HostTensorView {
            data: &identity.materialize_f64(),
            shape: identity.shape(),
        };
        let lhs = provider.upload(&view_lhs).expect("upload lhs");
        let rhs = provider.upload(&view_rhs).expect("upload rhs");

        let result = pagefun_impl(
            Value::FunctionHandle("mtimes".into()),
            Value::GpuTensor(lhs),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(rhs)]),
        );
        let result = block_on(result).expect("pagefun");

        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape(), &[2, 2, 2]);
        assert_eq!(
            gathered.materialize_f64(),
            vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
            "GPU fallback should match identity broadcast"
        );
    });
}

#[test]
fn pagefun_mixed_gpu_host_inputs_return_gpu_results() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("lhs");
        let rhs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("rhs");
        let lhs = gpu_helpers::upload_tensor(provider, &lhs).expect("upload lhs");

        let result = block_on(pagefun_impl(
            Value::FunctionHandle("mtimes".into()),
            Value::GpuTensor(lhs),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
        ))
        .expect("mixed pagefun");
        assert!(
            matches!(result, Value::GpuTensor(_)),
            "mixed host/GPU pagefun must return a gpuArray"
        );
        let gathered = test_support::gather(result).expect("gather result");
        assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
    });
}

#[test]
fn pagefun_mixed_complex_gpu_host_inputs_return_complex_gpu_results() {
    test_support::with_test_provider(|provider| {
        let lhs = ComplexTensor::new(
            vec![(1.0, 1.0), (3.0, -1.0), (2.0, 0.5), (4.0, 2.0)],
            vec![2, 2],
        )
        .expect("lhs");
        let rhs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("rhs");
        let lhs = gpu_helpers::upload_complex_tensor(provider, &lhs).expect("upload lhs");

        let result = block_on(pagefun_impl(
            Value::FunctionHandle("mtimes".into()),
            Value::GpuTensor(lhs),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
        ))
        .expect("mixed complex pagefun");
        let Value::GpuTensor(handle) = result else {
            panic!("mixed complex host/GPU pagefun must return a gpuArray");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&handle),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn pagefun_wgpu_mtimes_batches() {
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};

    let _ = register_wgpu_provider(WgpuProviderOptions::default()).expect("register wgpu provider");
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");

    let lhs = Tensor::new(
        vec![
            1.0, 4.0, 2.0, 5.0, //
            3.0, 6.0, 4.0, 7.0,
        ],
        vec![2, 2, 2],
    )
    .unwrap();
    let rhs = Tensor::new(
        vec![
            1.0, 0.0, 0.0, 1.0, //
            2.0, 1.0, 3.0, 2.0,
        ],
        vec![2, 2, 2],
    )
    .unwrap();

    let view_lhs = HostTensorView {
        data: &lhs.materialize_f64(),
        shape: lhs.shape(),
    };
    let view_rhs = HostTensorView {
        data: &rhs.materialize_f64(),
        shape: rhs.shape(),
    };

    let lhs_handle = provider.upload(&view_lhs).expect("upload lhs");
    let rhs_handle = provider.upload(&view_rhs).expect("upload rhs");

    let provider_handles = vec![lhs_handle.clone(), rhs_handle.clone()];
    let request = build_pagefun_request(&PageOperation::Mtimes, &provider_handles)
        .expect("build request")
        .expect("request available");

    let provider_result = provider.pagefun(&request).expect("wgpu pagefun execution");
    let provider_tensor =
        test_support::gather(Value::GpuTensor(provider_result)).expect("gather provider");

    let builtin_value = pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        Value::GpuTensor(lhs_handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            rhs_handle.clone(),
        )]),
    );
    let builtin_value = block_on(builtin_value).expect("pagefun builtin on GPU");
    let builtin_tensor = test_support::gather(builtin_value).expect("gather builtin");

    let expected_value = pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        Value::Tensor(lhs.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs.clone())]),
    );
    let expected_value = block_on(expected_value).expect("pagefun host baseline");
    let expected_tensor = match expected_value {
        Value::Tensor(t) => t,
        other => panic!("expected tensor result, got {other:?}"),
    };

    assert_eq!(provider_tensor.shape(), expected_tensor.shape());
    assert_eq!(
        provider_tensor.materialize_f64(),
        expected_tensor.materialize_f64()
    );
    assert_eq!(builtin_tensor.shape(), expected_tensor.shape());
    assert_eq!(
        builtin_tensor.materialize_f64(),
        expected_tensor.materialize_f64()
    );
}
