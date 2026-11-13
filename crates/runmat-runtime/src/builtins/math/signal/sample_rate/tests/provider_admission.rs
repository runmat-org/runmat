use super::*;

#[test]
fn physical_alias_policy_rejects_distinct_intermediate_view() {
    test_support::with_test_provider(|provider| {
        let intermediate = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0],
                shape: &[1, 4],
            })
            .unwrap();
        let other_view = intermediate
            .with_view(runmat_accelerate_api::GpuTensorView::contiguous(vec![1, 4]))
            .unwrap();
        assert_ne!(intermediate.view_identity(), other_view.view_identity());
        assert!(other_view.shares_allocation_with(&intermediate));
        assert!(!valid_class_preserving_gpu_output(
            provider,
            &intermediate,
            &other_view,
            &[1, 4],
            true
        ));
        assert!(intermediate.allocation_lease().validate_live().is_ok());
        let _ = other_view.release();
        let _ = intermediate.release();
    });
}

#[test]
fn upsample_gpu_uses_provider_scatter_and_preserves_residency() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            })
            .expect("upload input");
        provider.reset_telemetry();

        let out = call_upsample(vec![
            Value::GpuTensor(handle.clone()),
            Value::Num(2.0),
            Value::Num(1.0),
        ]);
        let Value::GpuTensor(out_handle) = out else {
            panic!("expected resident gpu tensor");
        };
        assert_eq!(out_handle.shape(), vec![4, 2]);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let gathered = test_support::gather(Value::GpuTensor(out_handle)).expect("gather output");
        assert_eq!(gathered.shape(), [4, 2]);
        assert_eq!(
            gathered.materialize_f64(),
            vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0]
        );
        let _ = handle.release();
    });
}

#[test]
fn upsample_gpu_preserves_poisoned_uint64_storage_without_host_download() {
    test_support::with_test_provider(|provider| {
        let input =
            Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2])
                .expect("native uint64 input");
        let handle = gpu_helpers::upload_tensor(provider, &input)
            .expect("upload native uint64 input without mirror materialization");
        provider.reset_telemetry();

        let out = call_upsample(vec![Value::GpuTensor(handle.clone()), Value::Num(2.0)]);
        let Value::GpuTensor(out_handle) = out else {
            panic!("expected resident native integer gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&out_handle),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let gathered = test_support::gather(Value::GpuTensor(out_handle.clone()))
            .expect("gather native integer output");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![1_u64 << 63, 0, u64::MAX, 0]))
        );
        handle.release().expect("free input");
        out_handle.release().expect("free output");
    });
}

#[test]
fn downsample_resident_integer_preserves_owner_class_and_values() {
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView, IntegerElementType};

    test_support::with_test_provider(|provider| {
        let values = [1u64, 2, 3, u64::MAX, 5];
        let input = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&values),
                shape: &[1, 5],
            })
            .expect("integer upload");
        let output = call_downsample(vec![
            Value::GpuTensor(input.clone()),
            Value::Num(2.0),
            Value::Num(1.0),
        ]);
        let Value::GpuTensor(output) = output else {
            panic!("expected resident integer output")
        };
        assert_eq!(output.device_id(), input.device_id());
        assert_eq!(output.shape(), vec![1, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(IntegerElementType::U64)
        );
        assert!(
            runmat_accelerate_api::provider_for_handle(&output).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        let gathered = block_on(provider.download_integer(&output)).expect("integer download");
        assert_eq!(
            gathered.data,
            runmat_accelerate_api::HostIntegerDataOwned::U64(vec![2, u64::MAX])
        );
        let input_lease = input.allocation_lease().downgrade();
        let output_lease = output.allocation_lease().downgrade();
        assert_eq!(
            block_on(provider.download_integer(&input))
                .expect("caller input survives")
                .data,
            runmat_accelerate_api::HostIntegerDataOwned::U64(values.to_vec())
        );
        input.release().expect("release input");
        output.release().expect("release output");
        assert!(input_lease.acquire().is_none());
        assert!(output_lease.acquire().is_none());
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn downsample_wgpu_integer_data_and_typed_controls_stay_exact_and_resident() {
    use runmat_accelerate_api::{
        HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView, IntegerElementType,
    };

    let _accel_guard = test_support::accel_test_lock();
    match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) {
        Ok(_) => {}
        Err(error) if error.to_string() == "wgpu: no compatible adapter found" => return,
        Err(error) => panic!("register wgpu provider failed: {error:?}"),
    }
    let provider = runmat_accelerate_api::provider().expect("WGPU provider");
    let values = [
        1_u64,
        9_007_199_254_740_993,
        3,
        u64::MAX,
        5,
        9_007_199_254_740_995,
    ];
    let input = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&values),
            shape: &[1, 6],
        })
        .expect("upload WGPU uint64 input");

    let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let output = call_downsample(vec![
        Value::GpuTensor(input.clone()),
        Value::Int(runmat_value::IntValue::U8(2)),
        Value::Int(runmat_value::IntValue::U8(1)),
    ]);
    drop(extensions);
    let Value::GpuTensor(output) = output else {
        panic!("expected resident WGPU integer output")
    };

    assert_eq!(output.device_id(), input.device_id());
    assert_eq!(output.shape(), [1, 3]);
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&output),
        Some(IntegerElementType::U64)
    );
    assert!(runmat_accelerate_api::provider_for_handle(&output)
        .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider)));
    let downloaded =
        block_on(provider.download_integer(&output)).expect("download WGPU uint64 output");
    assert_eq!(downloaded.shape, vec![1, 3]);
    assert_eq!(
        downloaded.data,
        HostIntegerDataOwned::U64(vec![9_007_199_254_740_993, u64::MAX, 9_007_199_254_740_995,])
    );

    input.release().expect("free WGPU input");
    output.release().expect("free WGPU output");
}

#[test]
fn downsample_gpu_uses_provider_linear_gather_and_preserves_residency() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            })
            .expect("upload input");
        provider.reset_telemetry();

        let out = call_downsample(vec![
            Value::GpuTensor(handle.clone()),
            Value::Num(2.0),
            Value::Num(1.0),
        ]);
        let Value::GpuTensor(out_handle) = out else {
            panic!("expected resident gpu tensor");
        };
        assert_eq!(out_handle.shape(), vec![1, 2]);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let gathered = test_support::gather(Value::GpuTensor(out_handle)).expect("gather output");
        assert_eq!(gathered.shape(), [1, 2]);
        assert_eq!(gathered.materialize_f64(), vec![2.0, 4.0]);
        let _ = handle.release();
    });
}

#[test]
fn upsample_complex_gpu_stays_resident_and_preserves_interleaved_storage() {
    test_support::with_test_provider(|provider| {
        let input = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
        provider.reset_telemetry();

        let out = call_upsample(vec![
            Value::GpuTensor(handle.clone()),
            Value::Num(2.0),
            Value::Num(1.0),
        ]);
        let Value::GpuTensor(out_handle) = out else {
            panic!("expected resident gpu tensor");
        };
        assert_eq!(out_handle.shape(), vec![1, 4]);
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let host = block_on(provider.download(&out_handle)).expect("download output");
        assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(host.shape, vec![1, 4]);
        assert_eq!(host.data, vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
        let _ = handle.release();
        let _ = out_handle.release();
    });
}

#[test]
fn downsample_complex_gpu_stays_resident_and_gathers_logical_elements() {
    test_support::with_test_provider(|provider| {
        let input = ComplexTensor::new(
            vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)],
            vec![1, 4],
        )
        .unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
        provider.reset_telemetry();

        let out = call_downsample(vec![
            Value::GpuTensor(handle.clone()),
            Value::Num(2.0),
            Value::Num(1.0),
        ]);
        let Value::GpuTensor(out_handle) = out else {
            panic!("expected resident gpu tensor");
        };
        assert_eq!(out_handle.shape(), vec![1, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let host = block_on(provider.download(&out_handle)).expect("download output");
        assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(host.shape, vec![1, 2]);
        assert_eq!(host.data, vec![2.0, -2.0, 4.0, -4.0]);
        let _ = handle.release();
        let _ = out_handle.release();
    });
}

#[test]
fn resample_gpu_composes_scatter_filter_and_gather_resident_output() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            })
            .expect("upload input");
        provider.reset_telemetry();

        let out = call_resample(vec![
            Value::GpuTensor(handle.clone()),
            Value::Num(2.0),
            Value::Num(1.0),
            tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
        ]);
        let Value::GpuTensor(out_handle) = out else {
            panic!("expected resident gpu tensor");
        };
        assert_eq!(out_handle.shape(), vec![1, 6]);

        let gathered = test_support::gather(Value::GpuTensor(out_handle)).expect("gather output");
        assert_eq!(gathered.shape(), [1, 6]);
        assert_eq!(
            gathered.materialize_f64(),
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
        );
        let _ = handle.release();
    });
}

#[test]
fn resample_gpu_accepts_resident_custom_filter_without_download() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            })
            .expect("upload input");
        let filter = [0.0, 1.0, 0.0];
        let filter_shape = [1usize, 3usize];
        let filter_handle = provider
            .upload(&HostTensorView {
                data: &filter,
                shape: &filter_shape,
            })
            .expect("upload filter");
        provider.reset_telemetry();

        let out = call_resample(vec![
            Value::GpuTensor(handle.clone()),
            Value::Num(2.0),
            Value::Num(1.0),
            Value::GpuTensor(filter_handle.clone()),
        ]);
        let Value::GpuTensor(out_handle) = out else {
            panic!("expected resident gpu tensor");
        };
        assert_eq!(out_handle.shape(), vec![1, 6]);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let gathered =
            test_support::gather(Value::GpuTensor(out_handle.clone())).expect("gather output");
        assert_eq!(gathered.shape(), [1, 6]);
        assert_eq!(
            gathered.materialize_f64(),
            vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
        );
        let _ = handle.release();
        let _ = filter_handle.release();
        let _ = out_handle.release();
    });
}

#[test]
fn resample_complex_gpu_stays_resident_through_filter_pipeline() {
    test_support::with_test_provider(|provider| {
        let input = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
        provider.reset_telemetry();

        let out = call_resample(vec![
            Value::GpuTensor(handle.clone()),
            Value::Num(2.0),
            Value::Num(1.0),
            tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
        ]);
        let Value::GpuTensor(out_handle) = out else {
            panic!("expected resident gpu tensor");
        };
        assert_eq!(out_handle.shape(), vec![1, 4]);
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let host = block_on(provider.download(&out_handle)).expect("download output");
        assert_eq!(host.storage, GpuTensorStorage::ComplexInterleaved);
        assert_eq!(host.shape, vec![1, 4]);
        assert_eq!(host.data, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
        let _ = handle.release();
        let _ = out_handle.release();
    });
}

#[test]
fn resample_gpu_two_outputs_return_resident_signal_and_host_filter() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            })
            .expect("upload input");

        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let out = block_on(resample_builtin(
            Value::GpuTensor(handle.clone()),
            Value::Num(1.0),
            Value::Num(2.0),
            vec![tensor(vec![0.0, 1.0, 0.0], vec![1, 3])],
        ))
        .expect("gpu resample");
        let Value::OutputList(outputs) = out else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 2);
        let Value::GpuTensor(signal_handle) = &outputs[0] else {
            panic!("expected resident signal output");
        };
        assert_eq!(signal_handle.shape(), vec![1, 3]);
        let gathered = test_support::gather(outputs[0].clone()).expect("gather signal output");
        assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 5.0]);
        let Value::Tensor(filter) = &outputs[1] else {
            panic!("expected host filter tensor");
        };
        assert_eq!(filter.materialize_f64(), vec![0.0, 1.0, 0.0]);
        let _ = handle.release();
    });
}

#[test]
fn resample_gpu_short_filter_high_decimation_stays_resident() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new((1..=10).map(f64::from).collect(), vec![1, 10]).unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            })
            .expect("upload input");

        let out = call_resample(vec![
            Value::GpuTensor(handle.clone()),
            Value::Num(1.0),
            Value::Num(10.0),
            tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
        ]);
        let Value::GpuTensor(out_handle) = out else {
            panic!("expected resident gpu tensor");
        };
        assert_eq!(out_handle.shape(), vec![1, 1]);
        let gathered = test_support::gather(Value::GpuTensor(out_handle)).expect("gather output");
        assert_eq!(gathered.materialize_f64(), vec![1.0]);
        let _ = handle.release();
    });
}
