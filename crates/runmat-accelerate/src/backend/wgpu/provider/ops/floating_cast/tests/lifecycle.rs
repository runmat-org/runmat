use super::*;
use runmat_accelerate_api::observation::{observe_provider_future, ProviderOperationReceipt};

fn receipt(provider: &WgpuProvider) -> ProviderOperationReceipt {
    #[cfg(not(target_arch = "wasm32"))]
    {
        ProviderOperationReceipt::with_wgpu_progress(
            provider.observation_budget.clone(),
            provider
                .observation_progress
                .as_ref()
                .expect("native progress owner")
                .clone(),
        )
    }
    #[cfg(target_arch = "wasm32")]
    {
        ProviderOperationReceipt::with_budget(provider.observation_budget.clone())
    }
}

async fn logical_copy_maxbuffer_denial_is_pre_effect_unsupported() {
    let mut provider = owner().await;
    let logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[0, 1],
            shape: &[1, 2],
        })
        .unwrap();
    provider.adapter_limits.max_buffer_size = 3;
    let before = provider.allocations.record_count().unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let observation = receipt(&provider);
    let submitted = observation.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(observation.recorder()),
        provider.floating_cast_exec_typed(&logical, NumericPrecision::F32),
    )
    .await
    .expect_err("fresh numeric copy exceeds selected output extent");
    assert!(error.is_unsupported());
    assert!(!error.is_after_effect());
    assert_eq!(provider.allocations.record_count().unwrap(), before);
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(observation.recorder().submission_checkpoint(), submitted);
    logical.allocation_lease().validate_live().unwrap();
    logical.release().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual native WGPU adapter; run explicitly"]
async fn native_floating_copy_maxbuffer_denial_has_typed_pre_effect_phase() {
    logical_copy_maxbuffer_denial_is_pre_effect_unsupported().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_floating_copy_maxbuffer_denial_has_typed_pre_effect_phase() {
    logical_copy_maxbuffer_denial_is_pre_effect_unsupported().await;
}

async fn owner() -> WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    {
        WgpuProvider::new(Default::default()).expect("actual native WGPU adapter required")
    }
    #[cfg(target_arch = "wasm32")]
    {
        WgpuProvider::new_async(Default::default())
            .await
            .expect("actual browser WebGPU adapter required")
    }
}

async fn alias_and_logical_copy_without_compute_geometry() {
    let mut provider = owner().await;
    let ordinary = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[-0.0, f32::from_bits(1)]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[0, 1],
            shape: &[1, 2],
        })
        .unwrap();
    provider.workgroup_config.scalar = 0;
    provider.adapter_limits.max_compute_workgroups_per_dimension = 0;
    let before = provider.allocations.record_count().unwrap();
    let alias = provider
        .floating_cast_exec(&ordinary, NumericPrecision::F32)
        .await
        .unwrap();
    assert!(alias.shares_allocation_with(&ordinary));
    assert_eq!(provider.allocations.record_count().unwrap(), before);
    let copy = provider
        .floating_cast_exec(&logical, NumericPrecision::F32)
        .await
        .unwrap();
    assert!(!copy.shares_allocation_with(&logical));
    assert_eq!(
        copy.descriptor().element_type,
        Some(NumericElementType::F32)
    );
    assert_eq!(
        provider.download_numeric(&copy).await.unwrap().data,
        HostNumericDataOwned::F32(vec![0.0, 1.0])
    );
    assert_eq!(provider.allocations.record_count().unwrap(), before + 1);
    ordinary.allocation_lease().validate_live().unwrap();
    logical.allocation_lease().validate_live().unwrap();
    alias.release().unwrap();
    copy.release().unwrap();
    ordinary.release().unwrap();
    logical.release().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual native F32 WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f32"]
async fn native_non_dispatch_casts_ignore_disabled_geometry() {
    alias_and_logical_copy_without_compute_geometry().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_non_dispatch_casts_ignore_disabled_geometry() {
    alias_and_logical_copy_without_compute_geometry().await;
}

#[cfg(not(target_arch = "wasm32"))]
async fn f64_precision_cast_and_typed_phase() {
    let mut provider = owner().await;
    assert_eq!(
        provider.precision(),
        ProviderPrecision::F64,
        "this qualification requires an actual shader-f64 adapter"
    );
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[16_777_217.0, f64::INFINITY, f64::NAN, -0.0]),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let storage_bindings = provider.adapter_limits.max_storage_buffers_per_shader_stage;
    provider.adapter_limits.max_storage_buffers_per_shader_stage = 1;
    let before_denial = provider.allocations.record_count().unwrap();
    let cache_before_denial = provider.validated_fusion_cache.lock().unwrap().len();
    let observation = receipt(&provider);
    let submitted = observation.recorder().submission_checkpoint();
    let denied = observe_provider_future(
        Some(observation.recorder()),
        provider.floating_cast_exec_typed(&input, NumericPrecision::F32),
    )
    .await
    .expect_err("valid class-changing cast exceeds selected binding count");
    assert!(denied.is_unsupported());
    assert!(!denied.is_after_effect());
    assert_eq!(provider.allocations.record_count().unwrap(), before_denial);
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        cache_before_denial
    );
    assert_eq!(observation.recorder().submission_checkpoint(), submitted);
    provider.adapter_limits.max_storage_buffers_per_shader_stage = storage_bindings;
    let result = provider
        .floating_cast_exec_typed(&input, NumericPrecision::F32)
        .await
        .unwrap();
    let HostNumericDataOwned::F32(values) = provider.download_numeric(&result).await.unwrap().data
    else {
        panic!("physical F32 output required")
    };
    assert_eq!(values[0].to_bits(), 16_777_216.0_f32.to_bits());
    assert_eq!(values[1], f32::INFINITY);
    assert!(values[2].is_nan());
    assert_eq!(values[3].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(provider.get_entry_raw(&result).unwrap().allocated_bytes, 16);
    result.release().unwrap();
    let before = provider.allocations.record_count().unwrap();
    let mut physical = None;
    let error = provider
        .floating_cast_exec_typed_after_allocation(&input, NumericPrecision::F32, |buffer| {
            physical = Some(weak_buffer(buffer));
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "injected after F64 cast allocation",
            ))
        })
        .await
        .expect_err("postallocation unsupported is terminal");
    assert!(error.is_after_effect());
    assert!(!error.is_unsupported());
    assert!(error
        .source()
        .is_some_and(|source| source.is::<runmat_accelerate_api::UnsupportedProviderOperation>()));
    assert!(physical
        .expect("actual physical output boundary reached")
        .upgrade()
        .is_none());
    assert_eq!(provider.allocations.record_count().unwrap(), before);
    input.allocation_lease().validate_live().unwrap();
    input.release().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual shader-f64 native WGPU adapter; run explicitly"]
async fn native_f64_precision_and_postallocation_cause() {
    f64_precision_cast_and_typed_phase().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_f32_source_rejects_f64_target_before_effects() {
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    let source = [f32::from_bits(1), -0.0];
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&source),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert_eq!(
        provider.get_entry_raw(&input).unwrap().element_type,
        NumericElementType::F32
    );
    browser_f64_capability_refusal(&provider, &input, NumericPrecision::F64).await;
    let HostNumericDataOwned::F32(actual) = provider.download_numeric(&input).await.unwrap().data
    else {
        panic!("borrowed source must remain physical F32")
    };
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        source.map(f32::to_bits).to_vec()
    );
    input.release().unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_physical_f64_source_rejects_f32_target_before_effects() {
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    let source = [
        f64::from_bits(1),
        f64::from_bits(0x7ff8_0000_0000_0042),
        -0.0,
    ];
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&source),
            shape: &[1, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert_eq!(
        provider.get_entry_raw(&input).unwrap().element_type,
        NumericElementType::F64
    );
    browser_f64_capability_refusal(&provider, &input, NumericPrecision::F32).await;
    let HostNumericDataOwned::F64(actual) = provider.download_numeric(&input).await.unwrap().data
    else {
        panic!("borrowed source must remain physical F64")
    };
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        source.map(f64::to_bits).to_vec()
    );
    input.release().unwrap();
}

#[cfg(target_arch = "wasm32")]
async fn browser_f64_capability_refusal(
    provider: &WgpuProvider,
    input: &GpuTensorHandle,
    target: NumericPrecision,
) {
    let before = provider.allocations.record_count().unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let observation = receipt(provider);
    let submitted = observation.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(observation.recorder()),
        provider.floating_cast_exec_typed(input, target),
    )
    .await
    .expect_err("browser WebGPU has no shader-f64 cast capability");
    assert!(error.is_unsupported());
    assert!(!error.is_after_effect());
    assert_eq!(provider.allocations.record_count().unwrap(), before);
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(observation.recorder().submission_checkpoint(), submitted);
    input.allocation_lease().validate_live().unwrap();
}
