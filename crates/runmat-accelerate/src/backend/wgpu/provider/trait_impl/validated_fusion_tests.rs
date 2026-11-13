use super::*;
use crate::wgsl::f64_math::PreparationError;
use runmat_accelerate_api::observation::{observe_provider_future, ProviderOperationReceipt};
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataView, HostIntegerTensorView, HostTensorView, ProviderElementType,
    ProviderShaderRequirements,
};

async fn owner() -> &'static WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let mut provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(Default::default())
        .await
        .expect("actual WebGPU adapter");
    #[cfg(not(target_arch = "wasm32"))]
    {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        provider.pipeline_cache_dir = Some(std::env::temp_dir().join(format!(
            "runmat-validated-fusion-{}-{nonce}",
            std::process::id()
        )));
    }
    let provider = Box::leak(Box::new(provider));
    // SAFETY: this independently created actual owner lives for the process.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

fn native_word(provider: &WgpuProvider) -> (&'static str, ProviderElementType) {
    match provider.precision {
        NumericPrecision::F32 => ("f32", ProviderElementType::F32),
        NumericPrecision::F64 => ("f64", ProviderElementType::F64),
    }
}

fn shader(word: &str, multi: bool) -> String {
    let second = if multi {
        format!("@group(0) @binding(2) var<storage, read_write> output1: array<{word}>;")
    } else {
        String::new()
    };
    let uniform_binding = if multi { 3 } else { 2 };
    let second_store = if multi {
        format!("output1[j] = input0[j] + {word}(2.0);")
    } else {
        String::new()
    };
    format!(
        "@group(0) @binding(0) var<storage, read> input0: array<{word}>;
         @group(0) @binding(1) var<storage, read_write> output0: array<{word}>;
         {second}
         struct Params {{ len: u32, offset: u32, rank: u32, pad: u32 }};
         @group(0) @binding({uniform_binding}) var<uniform> params: Params;
         @compute @workgroup_size(@WG@)
         fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
             if (id.x >= params.len) {{ return; }}
             let j = params.offset + id.x;
             output0[j] = input0[j] + {word}(1.0);
             {second_store}
         }}"
    )
}

async fn exercise_cold_and_warm(provider: &'static WgpuProvider) {
    let input = provider
        .upload(&HostTensorView {
            data: &[2.0, 4.0],
            shape: &[2],
        })
        .unwrap();
    let (word, class) = native_word(provider);
    let bad = "@compute @workgroup_size(@WG@) fn main() { let invalid = ; }";
    let before = provider.validated_fusion_cache.lock().unwrap().len();
    let records_before = provider.allocations.record_count().unwrap();
    let telemetry_before = provider.telemetry_snapshot();
    #[cfg(not(target_arch = "wasm32"))]
    let bad_recipe = super::super::pipeline_recipe::PipelineRecipe::fusion(
        bad,
        1,
        super::super::pipeline_recipe::FusionPipelineLayout::Single,
        provider.linear_launch.workgroup(),
    );
    let receipt = ProviderOperationReceipt::new();
    let checkpoint = receipt.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(receipt.recorder()),
        provider.fused_elementwise(
            bad,
            ProviderShaderRequirements::Basic,
            std::slice::from_ref(&input),
            &[2],
            2,
            class,
        ),
    )
    .await
    .unwrap_err();
    assert_eq!(receipt.recorder().submission_checkpoint(), checkpoint);
    assert!(!error.is_unsupported());
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    while !cause.is::<PreparationError>() {
        cause = cause
            .source()
            .expect("original typed shader compiler cause");
    }
    assert!(matches!(
        cause.downcast_ref::<PreparationError>().unwrap(),
        PreparationError::Parse(_)
    ));
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before
    );
    assert_eq!(provider.allocations.record_count().unwrap(), records_before);
    let telemetry_after = provider.telemetry_snapshot();
    assert_eq!(
        telemetry_after.fusion_cache_misses,
        telemetry_before.fusion_cache_misses
    );
    assert_eq!(
        telemetry_after.fused_elementwise.count,
        telemetry_before.fused_elementwise.count
    );
    assert_eq!(
        telemetry_after.kernel_launches.len(),
        telemetry_before.kernel_launches.len()
    );
    // A malformed request must be rejected before the deliberately malformed
    // shader is sent to the driver or published in either cache.
    let wrong_shape = provider
        .fused_elementwise(
            bad,
            ProviderShaderRequirements::Basic,
            std::slice::from_ref(&input),
            &[3],
            2,
            class,
        )
        .await
        .unwrap_err();
    assert!(wrong_shape
        .to_string()
        .contains("shape and physical output words disagree"));
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before
    );
    assert_eq!(provider.allocations.record_count().unwrap(), records_before);
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        telemetry_before.fusion_cache_misses
    );
    let empty_before_shape = provider
        .fused_elementwise(bad, ProviderShaderRequirements::Basic, &[], &[3], 2, class)
        .await
        .unwrap_err();
    assert!(empty_before_shape
        .to_string()
        .contains("fused_elementwise: no inputs"));
    #[cfg(not(target_arch = "wasm32"))]
    let oversized_before_shape = provider
        .fused_elementwise(
            bad,
            ProviderShaderRequirements::Basic,
            std::slice::from_ref(&input),
            &[3],
            usize::MAX,
            class,
        )
        .await
        .unwrap_err();
    #[cfg(not(target_arch = "wasm32"))]
    assert!(oversized_before_shape
        .to_string()
        .contains("fused_elementwise: tensor too large"));
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before
    );
    assert_eq!(provider.allocations.record_count().unwrap(), records_before);
    let integer = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I32(&[2, 4]),
            shape: &[2],
        })
        .unwrap();
    let integer_records = provider.allocations.record_count().unwrap();
    let integer_telemetry = provider.telemetry_snapshot();
    let wrong_input = provider
        .fused_elementwise(
            bad,
            ProviderShaderRequirements::Basic,
            std::slice::from_ref(&integer),
            &[2],
            2,
            class,
        )
        .await
        .unwrap_err();
    assert!(wrong_input.is_unsupported());
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before
    );
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        integer_records
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        integer_telemetry.fusion_cache_misses
    );
    integer.allocation_lease().validate_live().unwrap();
    let invalid_multi = provider
        .fused_elementwise_multi(
            bad,
            ProviderShaderRequirements::Basic,
            std::slice::from_ref(&input),
            &[3],
            2,
            &[class, class],
        )
        .await
        .unwrap_err();
    assert!(invalid_multi
        .to_string()
        .contains("shape and physical output words disagree"));
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before
    );
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        integer_records
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        integer_telemetry.fusion_cache_misses
    );
    input.allocation_lease().validate_live().unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = provider.pipeline_cache_dir.as_ref().unwrap();
        assert!(!dir.join(format!("{:016x}.wgsl", bad_recipe.key())).exists());
        assert!(!dir.join(format!("{:016x}.json", bad_recipe.key())).exists());
    }
    let source = shader(word, false);
    let one = provider
        .fused_elementwise(
            &source,
            ProviderShaderRequirements::Basic,
            std::slice::from_ref(&input),
            &[2],
            2,
            class,
        )
        .await
        .unwrap();
    assert_eq!(provider.download(&one).await.unwrap().data, [3.0, 5.0]);
    let count = provider.validated_fusion_cache.lock().unwrap().len();
    let two = provider
        .fused_elementwise(
            &source,
            ProviderShaderRequirements::Basic,
            std::slice::from_ref(&input),
            &[2],
            2,
            class,
        )
        .await
        .unwrap();
    assert_eq!(provider.download(&two).await.unwrap().data, [3.0, 5.0]);
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), count);
    let multi = shader(word, true);
    let classes = [class, class];
    for _ in 0..2 {
        let outputs = provider
            .fused_elementwise_multi(
                &multi,
                ProviderShaderRequirements::Basic,
                std::slice::from_ref(&input),
                &[2],
                2,
                &classes,
            )
            .await
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(
            provider.download(&outputs[0]).await.unwrap().data,
            [3.0, 5.0]
        );
        assert_eq!(
            provider.download(&outputs[1]).await.unwrap().data,
            [4.0, 6.0]
        );
    }
    input.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(not(target_arch = "wasm32"), ignore = "requires actual WGPU adapter")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_fused_cold_error_then_single_multi_cold_warm_numerical() {
    exercise_cold_and_warm(owner().await).await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn pending_cold_preparation_cancellation_keeps_borrowed_input() {
    use std::future::Future;
    let provider = owner().await;
    let input = provider
        .upload(&HostTensorView {
            data: &[2.0, 4.0],
            shape: &[2],
        })
        .unwrap();
    let (word, class) = native_word(provider);
    let source = shader(word, false);
    let inputs = [input.clone()];
    let shape = [2];
    let mut future = Box::pin(provider.fused_elementwise(
        &source,
        ProviderShaderRequirements::Basic,
        &inputs,
        &shape,
        2,
        class,
    ));
    let pending = std::future::poll_fn(|context| {
        std::task::Poll::Ready(future.as_mut().poll(context).is_pending())
    })
    .await;
    assert!(pending);
    assert!(provider.validated_fusion_cache.lock().unwrap().is_empty());
    drop(future);
    assert!(provider.validated_fusion_cache.lock().unwrap().is_empty());
    input.allocation_lease().validate_live().unwrap();
    // The following real scoped preparation also verifies that cancellation
    // left the device gate usable and did not publish a partial candidate.
    let output = provider
        .fused_elementwise(
            &source,
            ProviderShaderRequirements::Basic,
            &inputs,
            &shape,
            2,
            class,
        )
        .await
        .unwrap();
    assert_eq!(provider.download(&output).await.unwrap().data, [3.0, 5.0]);
    input.allocation_lease().validate_live().unwrap();
}
