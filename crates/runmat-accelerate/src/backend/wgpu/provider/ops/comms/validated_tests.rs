use super::*;
use crate::wgsl::f64_math::PreparationError;
use runmat_accelerate_api::observation::{observe_provider_future, ProviderOperationReceipt};
use runmat_accelerate_api::{
    AccelProvider, ProviderBitModulationRequest, ProviderModulationRequest,
};

fn operation_receipt(provider: &WgpuProvider) -> ProviderOperationReceipt {
    #[cfg(not(target_arch = "wasm32"))]
    {
        ProviderOperationReceipt::with_wgpu_progress(
            provider.observation_budget.clone(),
            provider
                .observation_progress
                .clone()
                .expect("actual native observation progress"),
        )
    }
    #[cfg(target_arch = "wasm32")]
    {
        ProviderOperationReceipt::with_budget(provider.observation_budget.clone())
    }
}

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
            "runmat-validated-modulation-{}-{nonce}",
            std::process::id()
        )));
    }
    let provider = Box::leak(Box::new(provider));
    // SAFETY: this independent actual owner lives for the process.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

fn assert_complex_points(
    host: &runmat_accelerate_api::HostTensorOwned,
    indices: &[usize],
    constellation: &[f64],
) {
    assert_eq!(host.data.len(), indices.len() * 2);
    for (point, &index) in host.data.chunks_exact(2).zip(indices) {
        assert_eq!(point, &constellation[index * 2..index * 2 + 2]);
    }
}

async fn cold_warm_and_failed_scope(provider: &'static WgpuProvider) {
    let indices: Vec<usize> = (0..4097).map(|i| (i + 1) % 4).collect();
    let symbols: Vec<f64> = indices.iter().map(|&index| index as f64).collect();
    let shape = [1usize, symbols.len()];
    let input = provider
        .upload(&HostTensorView {
            data: &symbols,
            shape: &shape,
        })
        .unwrap();
    let bit_values: Vec<f64> = indices
        .iter()
        .flat_map(|&index| [((index >> 1) & 1) as f64, (index & 1) as f64])
        .collect();
    let bit_shape = [2usize, indices.len()];
    let bit_input = provider
        .upload(&HostTensorView {
            data: &bit_values,
            shape: &bit_shape,
        })
        .unwrap();
    let before_records = provider.allocations.record_count().unwrap();
    let before_cache = provider.validated_fusion_cache.lock().unwrap().len();
    let before_metrics = provider.telemetry_snapshot();
    assert!(provider
        .modulate_constellation(ProviderModulationRequest {
            input: &input,
            constellation: &[],
        })
        .await
        .unwrap_err()
        .to_string()
        .contains("constellation pairs"));
    assert!(provider
        .modulate_bits_constellation(ProviderBitModulationRequest {
            input: &input,
            input_rows: 0,
            bits_per_symbol: 1,
            constellation: &[2.0, 3.0],
        })
        .await
        .unwrap_err()
        .to_string()
        .contains("invalid bit grouping"));
    assert_eq!(provider.allocations.record_count().unwrap(), before_records);
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before_cache
    );
    let bad = super::super::pipeline_recipe::PipelineRecipe {
        shader: "@compute @workgroup_size(1) fn main() { let bad = ; }".into(),
        layout: validated_route::LAYOUT_TAG.into(),
        workgroup: provider.workgroup_config.scalar,
    };
    let receipt = operation_receipt(provider);
    let checkpoint = receipt.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(receipt.recorder()),
        provider.prepare_validated_modulation_pipeline(&bad, "runmat-modulate-invalid-cold"),
    )
    .await
    .err()
    .expect("invalid WGSL must fail typed compiler preparation before driver effects");
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
    assert_eq!(receipt.recorder().submission_checkpoint(), checkpoint);
    assert_eq!(provider.allocations.record_count().unwrap(), before_records);
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before_cache
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        before_metrics.fusion_cache_misses
    );
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = provider.pipeline_cache_dir.as_ref().unwrap();
        assert!(!dir.join(format!("{:016x}.wgsl", bad.key())).exists());
        assert!(!dir.join(format!("{:016x}.json", bad.key())).exists());
    }

    let constellation = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0];
    for route in 0..2 {
        let cold_before = provider.telemetry_snapshot().fusion_cache_misses;
        let hits_before = provider.telemetry_snapshot().fusion_cache_hits;
        for attempt in 0..2 {
            let output = if route == 0 {
                provider
                    .modulate_constellation(ProviderModulationRequest {
                        input: &input,
                        constellation: &constellation,
                    })
                    .await
                    .unwrap()
            } else {
                provider
                    .modulate_bits_constellation(ProviderBitModulationRequest {
                        input: &bit_input,
                        input_rows: 2,
                        bits_per_symbol: 2,
                        constellation: &constellation,
                    })
                    .await
                    .unwrap()
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(output.device_id(), provider.device_id());
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(provider.precision())
            );
            assert!(!output.shares_allocation_with(if route == 0 { &input } else { &bit_input }));
            output.allocation_lease().validate_live().unwrap();
            let gathered = provider.download(&output).await.unwrap();
            assert_eq!(gathered.shape, shape);
            assert_complex_points(&gathered, &indices, &constellation);
            assert_eq!(
                provider.telemetry_snapshot().fusion_cache_misses,
                cold_before + 1,
            );
            assert_eq!(
                provider.telemetry_snapshot().fusion_cache_hits,
                hits_before + attempt,
            );
        }
    }
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before_cache + 2
    );
    input.allocation_lease().validate_live().unwrap();
    bit_input.allocation_lease().validate_live().unwrap();
    // The full u32 width is valid. Each column places its one-bit index at
    // the end of a 32-row group to distinguish the low-bit outputs.
    let wide_indices: Vec<usize> = (0..4097).map(|i| (i + 1) % 2).collect();
    let wide_bits: Vec<f64> = wide_indices
        .iter()
        .flat_map(|&index| (0..32).map(move |row| f64::from((row == 31 && index == 1) as u8)))
        .collect();
    let wide_shape = [32usize, wide_indices.len()];
    let wide_input = provider
        .upload(&HostTensorView {
            data: &wide_bits,
            shape: &wide_shape,
        })
        .unwrap();
    let wide_constellation = [2.0, 3.0, 5.0, 7.0];
    let wide_output = provider
        .modulate_bits_constellation(ProviderBitModulationRequest {
            input: &wide_input,
            input_rows: 32,
            bits_per_symbol: 32,
            constellation: &wide_constellation,
        })
        .await
        .unwrap();
    let wide_host = provider.download(&wide_output).await.unwrap();
    assert_eq!(wide_host.shape, shape);
    assert_complex_points(&wide_host, &wide_indices, &wide_constellation);
    wide_input.allocation_lease().validate_live().unwrap();

    let overflow_bits: Vec<f64> = std::iter::once(1.0)
        .chain(std::iter::repeat_n(0.0, 32))
        .collect();
    let overflow_shape = [33usize, 1];
    let overflow_input = provider
        .upload(&HostTensorView {
            data: &overflow_bits,
            shape: &overflow_shape,
        })
        .unwrap();
    let before_records = provider.allocations.record_count().unwrap();
    let before_cache = provider.validated_fusion_cache.lock().unwrap().len();
    let before_misses = provider.telemetry_snapshot().fusion_cache_misses;
    let receipt = operation_receipt(provider);
    let checkpoint = receipt.recorder().submission_checkpoint();
    let refused = observe_provider_future(
        Some(receipt.recorder()),
        provider.modulate_bits_constellation(ProviderBitModulationRequest {
            input: &overflow_input,
            input_rows: 33,
            bits_per_symbol: 33,
            constellation: &wide_constellation,
        }),
    )
    .await
    .unwrap_err();
    assert!(refused.is_unsupported());
    assert_eq!(receipt.recorder().submission_checkpoint(), checkpoint);
    assert_eq!(provider.allocations.record_count().unwrap(), before_records);
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before_cache
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        before_misses
    );
    overflow_input.allocation_lease().validate_live().unwrap();
    let mut invalid_symbols = symbols.clone();
    invalid_symbols[4096] = 0.5;
    let invalid_input = provider
        .upload(&HostTensorView {
            data: &invalid_symbols,
            shape: &shape,
        })
        .unwrap();
    let receipt = operation_receipt(provider);
    let failed = observe_provider_future(
        Some(receipt.recorder()),
        provider.modulate_constellation(ProviderModulationRequest {
            input: &invalid_input,
            constellation: &constellation,
        }),
    )
    .await
    .unwrap_err();
    assert!(failed.to_string().contains("nonnegative integers"));
    assert!(!failed.is_unsupported());
    assert!(receipt.recorder().submission_checkpoint() > 0);
    invalid_input.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(not(target_arch = "wasm32"), ignore = "requires actual WGPU adapter")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_tagged_modulation_cold_validation_and_both_public_routes() {
    cold_warm_and_failed_scope(owner().await).await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn pending_tagged_modulation_cancel_retains_borrowed_input() {
    use std::future::Future;
    let provider = owner().await;
    let symbols = vec![0.0; 4097];
    let shape = [1usize, symbols.len()];
    let input = provider
        .upload(&HostTensorView {
            data: &symbols,
            shape: &shape,
        })
        .unwrap();
    let constellation = [2.0, 3.0, 5.0, 7.0];
    let mut future = Box::pin(provider.modulate_constellation(ProviderModulationRequest {
        input: &input,
        constellation: &constellation,
    }));
    let pending = std::future::poll_fn(|context| {
        std::task::Poll::Ready(future.as_mut().poll(context).is_pending())
    })
    .await;
    assert!(pending);
    drop(future);
    assert!(provider.validated_fusion_cache.lock().unwrap().is_empty());
    input.allocation_lease().validate_live().unwrap();
    let output = provider
        .modulate_constellation(ProviderModulationRequest {
            input: &input,
            constellation: &constellation,
        })
        .await
        .unwrap();
    assert_complex_points(
        &provider.download(&output).await.unwrap(),
        &vec![0; symbols.len()],
        &constellation,
    );
    input.allocation_lease().validate_live().unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn postsubmit_modulation_readback_cancel_keeps_input_and_device_usable() {
    use std::future::Future;
    let provider = owner().await;
    let symbols = vec![0.0; 4097];
    let shape = [1usize, symbols.len()];
    let input = provider
        .upload(&HostTensorView {
            data: &symbols,
            shape: &shape,
        })
        .unwrap();
    let constellation = [2.0, 3.0, 5.0, 7.0];
    let receipt = operation_receipt(provider);
    let recorder = receipt.recorder();
    let mut future = Box::pin(observe_provider_future(
        Some(recorder.clone()),
        provider.modulate_constellation(ProviderModulationRequest {
            input: &input,
            constellation: &constellation,
        }),
    ));
    let reached_readback = std::future::poll_fn(|context| match future.as_mut().poll(context) {
        std::task::Poll::Pending if recorder.submission_checkpoint() > 0 => {
            std::task::Poll::Ready(true)
        }
        std::task::Poll::Pending => std::task::Poll::Pending,
        std::task::Poll::Ready(_) => std::task::Poll::Ready(false),
    })
    .await;
    assert!(
        reached_readback,
        "must cancel a submitted, pending readback"
    );
    drop(future);
    input.allocation_lease().validate_live().unwrap();
    // No absent receipt is interpreted as GPU completion. A fresh actual
    // operation/readback proves the cleanup did not strand the device gate.
    let output = provider
        .modulate_constellation(ProviderModulationRequest {
            input: &input,
            constellation: &constellation,
        })
        .await
        .unwrap();
    assert_complex_points(
        &provider.download(&output).await.unwrap(),
        &vec![0; symbols.len()],
        &constellation,
    );
    input.allocation_lease().validate_live().unwrap();
}
