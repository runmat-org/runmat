use super::*;
use runmat_accelerate_api::observation::observe_provider_future;

async fn valid_integer_resource_denial_is_pre_effect_unsupported() {
    let mut provider = owner().await;
    let input = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[-1, 9_007_199_254_740_993]),
            shape: &[1, 2],
        })
        .unwrap();
    provider.adapter_limits.max_storage_buffers_per_shader_stage = 1;
    let before = provider.allocations.record_count().unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let observation = receipt(&provider);
    let submitted = observation.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(observation.recorder()),
        provider.cast_to_integer_exec(&input, IntegerElementType::U64),
    )
    .await
    .expect_err("valid cast exceeds selected adapter binding count");
    assert!(error.is_unsupported());
    assert!(!error.is_after_effect());
    assert_eq!(provider.allocations.record_count().unwrap(), before);
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(observation.recorder().submission_checkpoint(), submitted);
    input.allocation_lease().validate_live().unwrap();
    input.release().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual native WGPU adapter; run explicitly"]
async fn native_integer_resource_denial_has_typed_pre_effect_phase() {
    valid_integer_resource_denial_is_pre_effect_unsupported().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_integer_resource_denial_has_typed_pre_effect_phase() {
    valid_integer_resource_denial_is_pre_effect_unsupported().await;
}
async fn f32_adapter_rejects_physical_f64_before_any_integer_output() {
    let provider = owner().await;
    assert_eq!(
        provider.precision(),
        ProviderPrecision::F32,
        "this qualification requires an actual F32 WebGPU adapter"
    );
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[-2.5, f64::INFINITY]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let before = provider.allocations.record_count().unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let error = provider
        .cast_to_integer_exec(&input, IntegerElementType::I16)
        .await
        .expect_err("an F32-only adapter cannot read physical F64 source lanes");
    assert!(error.is_unsupported());
    assert!(!error.is_after_effect());
    assert_eq!(provider.allocations.record_count().unwrap(), before);
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    input.allocation_lease().validate_live().unwrap();
    input.release().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual F32 native WGPU adapter; run explicitly"]
async fn native_f32_pre_effect_integer_cast_refusal() {
    f32_adapter_rejects_physical_f64_before_any_integer_output().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_f32_pre_effect_integer_cast_refusal() {
    f32_adapter_rejects_physical_f64_before_any_integer_output().await;
}
