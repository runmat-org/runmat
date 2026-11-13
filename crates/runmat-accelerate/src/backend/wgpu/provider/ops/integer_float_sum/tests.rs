use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelProvider, HostIntegerDataView, HostIntegerTensorView,
    HostLogicalTensorView, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
};
use std::error::Error;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn temporary_completion_retains_primary_and_typed_cleanup_owners() {
    let owner = TestAllocationOwner::new();
    let failed = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    failed.probe.fail_retirement("temporary retirement failed");
    let error = finish_temporary(
        Err(ProviderExecutionError::from(anyhow!(
            "primary execution failed"
        ))),
        failed.handle,
    )
    .unwrap_err();
    let carrier = error
        .source()
        .unwrap()
        .downcast_ref::<ProviderExecutionCompletionError>()
        .unwrap();
    assert_eq!(
        carrier.primary().unwrap().to_string(),
        "primary execution failed"
    );
    assert_eq!(failed.probe.retirements(), 1);
    assert_eq!(carrier.cleanup_errors().len(), 1);
    assert!(carrier.cleanup_errors()[0].physical.is_err());

    let late_unsupported = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = finish_temporary(
        Err(ProviderExecutionError::from(
            runmat_accelerate_api::unsupported_provider_operation("primary unsupported route"),
        )),
        late_unsupported.handle,
    )
    .unwrap_err();
    assert!(error.is_after_effect());
    assert!(
        !error.is_unsupported(),
        "an allocated temporary forbids replay"
    );
    assert_eq!(late_unsupported.probe.retirements(), 1);

    let unsupported_temporary = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    unsupported_temporary
        .probe
        .fail_retirement("unsupported route retirement failed");
    let error = finish_temporary(
        Err(ProviderExecutionError::from(
            runmat_accelerate_api::unsupported_provider_operation("primary unsupported route"),
        )),
        unsupported_temporary.handle,
    )
    .unwrap_err();
    assert!(
        !error.is_unsupported(),
        "failed cleanup cannot authorize fallback"
    );
    let carrier = error
        .source()
        .unwrap()
        .downcast_ref::<ProviderExecutionCompletionError>()
        .unwrap();
    assert!(carrier.primary().is_some());
    assert_eq!(carrier.cleanup_errors().len(), 1);
    assert_eq!(unsupported_temporary.probe.retirements(), 1);

    let temporary = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let output = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    temporary
        .probe
        .fail_retirement("temporary retirement failed");
    output.probe.fail_retirement("output retirement failed");
    let error = finish_temporary(Ok(output.handle), temporary.handle).unwrap_err();
    let carrier = error
        .source()
        .unwrap()
        .downcast_ref::<ProviderExecutionCompletionError>()
        .unwrap();
    assert!(carrier.primary().is_none());
    assert_eq!(carrier.cleanup_errors().len(), 2);
    assert!(carrier.cleanup_errors()[0].physical.is_err());
    assert!(carrier.cleanup_errors()[1].physical.is_err());
    assert_eq!(temporary.probe.retirements(), 1);
    assert_eq!(output.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn successful_conversion_forbids_later_unsupported_replay() {
    let owner = TestAllocationOwner::new();
    let converted = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = finish_temporary(
        Err(ProviderExecutionError::from(
            runmat_accelerate_api::unsupported_provider_operation("late reduction unsupported"),
        )),
        converted.handle,
    )
    .unwrap_err();
    assert!(error.is_after_effect());
    assert!(!error.is_unsupported());
    assert_eq!(converted.probe.retirements(), 1);
}

fn actual_adapter() -> &'static WgpuProvider {
    crate::backend::wgpu::provider::register_wgpu_provider(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU adapter required; an unavailable adapter is not a passing test")
}

#[test]
fn wide_word_conversion_has_one_rounding_boundary() {
    fn unsigned(low: u32, high: u32) -> f64 {
        f64::from(high) * 4_294_967_296.0 + f64::from(low)
    }
    fn signed(value: i64) -> f64 {
        let bits = value as u64;
        let low = bits as u32;
        let high = (bits >> 32) as u32;
        if (high & 0x8000_0000) == 0 {
            return unsigned(low, high);
        }
        let magnitude_low = (!low).wrapping_add(1);
        let carry = u32::from(magnitude_low == 0);
        -unsigned(magnitude_low, (!high).wrapping_add(carry))
    }
    for value in [0, 1, (1_u64 << 53) + 1, 1_u64 << 63, u64::MAX] {
        assert_eq!(unsigned(value as u32, (value >> 32) as u32), value as f64);
    }
    for value in [i64::MIN, -((1_i64 << 53) + 1), -1, 0, 1, i64::MAX] {
        assert_eq!(signed(value), value as f64);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; unpolled drop and retry"]
fn unpolled_f64_sum_drops_without_publishing_then_retries_numerically() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64,
    );
    let input = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I16(&[1, 2]),
            shape: &[2, 1],
        })
        .unwrap();
    let records_before = provider.allocations.record_count().unwrap();
    let cache_before = provider.validated_fusion_cache.lock().unwrap().len();
    let future = provider.integer_floating_sum_global_exec(&input);
    drop(future);
    assert_eq!(provider.allocations.record_count().unwrap(), records_before);
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        cache_before
    );
    input.allocation_lease().validate_live().unwrap();

    let output = block_on(provider.integer_floating_sum_global_exec(&input)).unwrap();
    assert_eq!(
        block_on(provider.download_numeric(&output)).unwrap().data,
        HostNumericDataOwned::F64(vec![3.0]),
    );
    input.allocation_lease().validate_live().unwrap();
    output.release().unwrap();
    input.release().unwrap();
}

#[test]
#[ignore = "explicit actual F64 WGPU hardware gate; run with --ignored on a shader-f64 adapter"]
fn ordinary_integer_sum_stays_resident_and_differs_from_native_saturation() {
    let provider = actual_adapter();
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let input = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I8(&[100, 100]),
            shape: &[2, 1],
        })
        .unwrap();
    let ordinary = block_on(provider.reduce_sum(&input)).unwrap();
    let native = block_on(provider.reduce_integer_sum_native(&input)).unwrap();
    assert_eq!(
        ordinary.descriptor().element_type,
        Some(NumericElementType::F64)
    );
    assert_eq!(
        block_on(provider.download_numeric(&ordinary)).unwrap().data,
        HostNumericDataOwned::F64(vec![200.0])
    );
    assert_eq!(
        block_on(provider.download_integer(&native)).unwrap().data,
        runmat_accelerate_api::HostIntegerDataOwned::I8(vec![127])
    );
    input.release().unwrap();
    ordinary.release().unwrap();
    native.release().unwrap();
}

#[test]
#[ignore = "explicit actual F64 WGPU hardware gate; run with --ignored on a shader-f64 adapter"]
fn ordinary_integer_sum_handles_wide_rounding_nd_and_lazy_transpose() {
    let provider = actual_adapter();
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let wide = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&[u64::MAX]),
            shape: &[1, 1],
        })
        .unwrap();
    let wide_sum = block_on(provider.reduce_sum(&wide)).unwrap();
    assert_eq!(
        block_on(provider.download_numeric(&wide_sum)).unwrap().data,
        HostNumericDataOwned::F64(vec![u64::MAX as f64])
    );
    wide_sum.release().unwrap();
    wide.release().unwrap();

    let input = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I16(&[1, 2, 3, 4, 5, 6, 7, 8]),
            shape: &[2, 2, 2],
        })
        .unwrap();
    let reduced = block_on(provider.reduce_sum_dim(&input, 1)).unwrap();
    assert_eq!(reduced.shape(), &[2, 1, 2]);
    assert_eq!(
        block_on(provider.download_numeric(&reduced)).unwrap().data,
        HostNumericDataOwned::F64(vec![4.0, 6.0, 12.0, 14.0])
    );
    let invalid_axis = block_on(provider.reduce_sum_dim(&input, 3)).unwrap_err();
    assert!(!invalid_axis.is_after_effect());
    assert!(!invalid_axis.is_unsupported());
    input.allocation_lease().validate_live().unwrap();
    reduced.release().unwrap();
    input.release().unwrap();

    let matrix = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I32(&[1, 2, 3, 4, 5, 6]),
            shape: &[2, 3],
        })
        .unwrap();
    let view = provider.transpose_exec(&matrix).unwrap();
    let columns = block_on(provider.reduce_sum_dim(&view, 0)).unwrap();
    assert_eq!(
        block_on(provider.download_numeric(&columns)).unwrap().data,
        HostNumericDataOwned::F64(vec![9.0, 12.0])
    );
    columns.release().unwrap();
    view.release().unwrap();
    matrix.release().unwrap();

    let logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[0, 1, 1, 0, 1, 1, 0, 0],
            shape: &[2, 2, 2],
        })
        .unwrap();
    let logical_sum = block_on(provider.reduce_sum_dim(&logical, 1)).unwrap();
    assert_eq!(logical_sum.shape(), &[2, 1, 2]);
    assert_eq!(
        logical_sum.physical().element_type(),
        NumericElementType::F64
    );
    assert_eq!(
        block_on(provider.download_numeric(&logical_sum))
            .unwrap()
            .data,
        HostNumericDataOwned::F64(vec![1.0, 1.0, 1.0, 1.0]),
    );
    logical.allocation_lease().validate_live().unwrap();
    logical_sum.release().unwrap();
    logical.release().unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn browser_sum_requires_real_double_capability_or_executes_actual_output() {
    let provider = WgpuProvider::new_async(Default::default())
        .await
        .expect("actual browser WebGPU adapter required");
    let input = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I16(&[1, 2, 3, 4, 5, 6, 7, 8]),
            shape: &[2, 2, 2],
        })
        .unwrap();
    if provider.precision() == runmat_accelerate_api::ProviderPrecision::F64 {
        let output = provider.reduce_sum_dim(&input, 1).await.unwrap();
        assert_eq!(output.shape(), &[2, 1, 2]);
        assert_eq!(
            provider.download_numeric(&output).await.unwrap().data,
            HostNumericDataOwned::F64(vec![4.0, 6.0, 12.0, 14.0]),
        );
        output.release().unwrap();
    } else {
        let error = provider.reduce_sum_dim(&input, 1).await.unwrap_err();
        assert!(error.is_unsupported());
        assert!(!error.is_after_effect());
    }
    input.allocation_lease().validate_live().unwrap();
    input.release().unwrap();
}

#[test]
#[ignore = "explicit actual F64 WGPU hardware gate; run with --ignored on a shader-f64 adapter"]
fn ordinary_single_sum_casts_before_reduction() {
    let provider = actual_adapter();
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[16_777_216.0, 1.0, -16_777_216.0]),
            shape: &[3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let output = block_on(provider.reduce_sum(&input)).unwrap();
    assert_eq!(output.physical().element_type(), NumericElementType::F64);
    assert_eq!(
        block_on(provider.download_numeric(&output)).unwrap().data,
        HostNumericDataOwned::F64(vec![1.0])
    );
    output.release().unwrap();
    input.release().unwrap();
}

#[test]
#[ignore = "explicit actual F32 WGPU hardware gate; run with --ignored on an F32-only adapter"]
fn f32_only_device_rejects_double_integer_and_logical_sums_before_dispatch() {
    let provider = actual_adapter();
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F32
    );
    let integer = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I8(&[100, 100]),
            shape: &[2, 1],
        })
        .unwrap();
    assert!(block_on(provider.reduce_sum(&integer))
        .unwrap_err()
        .is_unsupported());
    assert!(block_on(provider.reduce_sum_dim(&integer, 0))
        .unwrap_err()
        .is_unsupported());
    integer.release().unwrap();

    let logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 1],
            shape: &[2, 1],
        })
        .unwrap();
    assert!(block_on(provider.reduce_sum(&logical))
        .unwrap_err()
        .is_unsupported());
    assert!(block_on(provider.reduce_sum_dim(&logical, 0))
        .unwrap_err()
        .is_unsupported());
    logical.release().unwrap();

    let single = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[16_777_216.0, 1.0, -16_777_216.0]),
            shape: &[3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(block_on(provider.reduce_sum(&single))
        .unwrap_err()
        .is_unsupported());
    single.release().unwrap();
}
