use super::*;
use crate::wgsl::f64_math::PreparationError;
use runmat_accelerate_api::observation::{observe_provider_future, ProviderOperationReceipt};
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataView, HostIntegerTensorView, HostLogicalTensorView,
    HostTensorView, MatmulOffset, NumericElementType, ScaleOp,
};

fn receipt(provider: &WgpuProvider) -> ProviderOperationReceipt {
    #[cfg(not(target_arch = "wasm32"))]
    {
        ProviderOperationReceipt::with_wgpu_progress(
            provider.observation_budget.clone(),
            provider
                .observation_progress
                .as_ref()
                .expect("actual WGPU progress port")
                .clone(),
        )
    }
    #[cfg(target_arch = "wasm32")]
    {
        ProviderOperationReceipt::with_budget(provider.observation_budget.clone())
    }
}

async fn provider() -> &'static WgpuProvider {
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
            "runmat-validated-epilogue-{}-{nonce}",
            std::process::id()
        )));
    }
    let provider = Box::leak(Box::new(provider));
    // SAFETY: the actual fixture owner remains live for the process.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

async fn qualification(provider: &'static WgpuProvider) {
    let columns = provider.workgroup_config.matmul_tile as usize + 1;
    let a_data = [1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f64> = (0..columns)
        .flat_map(|col| [col as f64 + 1.0, (col % 3) as f64 + 2.0])
        .collect();
    let a = provider
        .upload(&HostTensorView {
            data: &a_data,
            shape: &[2, 2],
        })
        .unwrap();
    let b = provider
        .upload(&HostTensorView {
            data: &b_data,
            shape: &[2, columns],
        })
        .unwrap();
    let row = provider
        .upload(&HostTensorView {
            data: &[2.0, 4.0],
            shape: &[2, 1],
        })
        .unwrap();
    let col_data: Vec<f64> = (0..columns).map(|col| (col % 4 + 1) as f64).collect();
    let col = provider
        .upload(&HostTensorView {
            data: &col_data,
            shape: &[1, columns],
        })
        .unwrap();
    let mut ep = MatmulEpilogue::noop();
    ep.affine.scale = Some(2.0);
    ep.affine.offset = Some(MatmulOffset::SubtractFrom(100.0));
    ep.row_scale = Some(row.clone());
    ep.row_op = ScaleOp::Divide;
    ep.col_scale = Some(col.clone());
    ep.clamp_min = Some(0.0);
    ep.pow_exponent = Some(2.0);
    let expected: Vec<f64> = (0..columns)
        .flat_map(|column| {
            let x = b_data[2 * column];
            let y = b_data[2 * column + 1];
            let row0 = ((100.0 - 2.0 * (x + 3.0 * y)) / 2.0 * col_data[column]).max(0.0);
            let row1 = ((100.0 - 2.0 * (2.0 * x + 4.0 * y)) / 4.0 * col_data[column]).max(0.0);
            [row0.powi(2), row1.powi(2)]
        })
        .collect();
    let warm_expected: Vec<f64> = (0..columns)
        .flat_map(|column| {
            let x = b_data[2 * column];
            let y = b_data[2 * column + 1];
            [x + 3.0 * y, 2.0 * x + 4.0 * y]
        })
        .collect();
    let invalid = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 0, 0, 1],
            shape: &[2, 2],
        })
        .unwrap();
    let allocations = provider.allocations.record_count().unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let misses = provider.telemetry_snapshot().fusion_cache_misses;
    let hits = provider.telemetry_snapshot().fusion_cache_hits;
    let refused = receipt(provider);
    let before_submit = refused.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(refused.recorder()),
        provider.matmul_epilogue(&invalid, &b, &ep),
    )
    .await
    .expect_err("logical owner refused before pipeline preparation");
    assert!(error.is_unsupported());
    assert_eq!(refused.recorder().submission_checkpoint(), before_submit);
    assert_eq!(provider.allocations.record_count().unwrap(), allocations);
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(provider.telemetry_snapshot().fusion_cache_misses, misses);
    invalid.allocation_lease().validate_live().unwrap();
    let integer = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I32(&[1, 2, 3, 4]),
            shape: &[2, 2],
        })
        .unwrap();
    assert_eq!(integer.physical().element_type(), NumericElementType::I32);
    let integer_allocations = provider.allocations.record_count().unwrap();
    let integer_receipt = receipt(provider);
    let integer_checkpoint = integer_receipt.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(integer_receipt.recorder()),
        provider.matmul_epilogue(&integer, &b, &ep),
    )
    .await
    .expect_err("integer owner must be refused before cold preparation");
    assert!(error.is_unsupported());
    assert_eq!(
        integer_receipt.recorder().submission_checkpoint(),
        integer_checkpoint
    );
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        integer_allocations
    );
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(provider.telemetry_snapshot().fusion_cache_misses, misses);
    integer.allocation_lease().validate_live().unwrap();
    let short_row = provider
        .upload(&HostTensorView {
            data: &[2.0],
            shape: &[1, 1],
        })
        .unwrap();
    let mut invalid_scale = ep.clone();
    invalid_scale.row_scale = Some(short_row.clone());
    let scale_allocations = provider.allocations.record_count().unwrap();
    let scale_receipt = receipt(provider);
    let scale_checkpoint = scale_receipt.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(scale_receipt.recorder()),
        provider.matmul_epilogue(&a, &b, &invalid_scale),
    )
    .await
    .expect_err("invalid row extent must precede pipeline creation");
    assert!(!error.is_after_effect());
    assert_eq!(
        scale_receipt.recorder().submission_checkpoint(),
        scale_checkpoint
    );
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        scale_allocations
    );
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(provider.telemetry_snapshot().fusion_cache_misses, misses);
    short_row.allocation_lease().validate_live().unwrap();
    let mut unsupported_diag = ep.clone();
    unsupported_diag.diag_output = Some(short_row.clone());
    let diag_receipt = receipt(provider);
    let diag_checkpoint = diag_receipt.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(diag_receipt.recorder()),
        provider.matmul_epilogue(&a, &b, &unsupported_diag),
    )
    .await
    .expect_err("unsupported diagonal destination must precede any allocation");
    assert!(error.is_unsupported());
    assert_eq!(
        diag_receipt.recorder().submission_checkpoint(),
        diag_checkpoint
    );
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        scale_allocations
    );
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(provider.telemetry_snapshot().fusion_cache_misses, misses);
    let bad_shader = PipelineRecipe {
        shader: "@compute @workgroup_size(1) fn main() { let bad = ; }".into(),
        layout: crate::backend::wgpu::bindings::MATMUL_EPILOGUE_LAYOUT_TAG.into(),
        workgroup: provider.workgroup_config.matmul_tile,
    };
    let bad = receipt(provider);
    let before_submit = bad.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(bad.recorder()),
        provider.prepare_validated_matmul_epilogue_pipeline(&bad_shader),
    )
    .await
    .err()
    .expect("invalid WGSL must fail typed compiler preparation before driver effects");
    let mut source: &(dyn std::error::Error + 'static) = &error;
    while !source.is::<PreparationError>() {
        source = source
            .source()
            .expect("original typed shader compiler cause");
    }
    assert!(matches!(
        source.downcast_ref::<PreparationError>().unwrap(),
        PreparationError::Parse(_)
    ));
    assert_eq!(bad.recorder().submission_checkpoint(), before_submit);
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        scale_allocations
    );
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(provider.telemetry_snapshot().fusion_cache_misses, misses);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = provider.pipeline_cache_dir.as_ref().unwrap();
        assert!(!dir.join(format!("{:016x}.wgsl", bad_shader.key())).exists());
        assert!(!dir.join(format!("{:016x}.json", bad_shader.key())).exists());
    }
    let empty_a = provider
        .upload(&HostTensorView {
            data: &[],
            shape: &[0, 2],
        })
        .unwrap();
    let empty_b = provider
        .upload(&HostTensorView {
            data: &[1.0, 0.0, 0.0, 1.0],
            shape: &[2, 2],
        })
        .unwrap();
    let empty = provider
        .matmul_epilogue(&empty_a, &empty_b, &MatmulEpilogue::noop())
        .await
        .unwrap();
    assert_eq!(empty.shape(), &[0, 2]);
    assert!(!empty.shares_allocation_with(&empty_a));
    assert!(!empty.shares_allocation_with(&empty_b));
    let empty_host = provider.download(&empty).await.unwrap();
    assert_eq!(empty_host.shape, [0, 2]);
    assert!(empty_host.data.is_empty());
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(provider.telemetry_snapshot().fusion_cache_misses, misses);
    empty_a.allocation_lease().validate_live().unwrap();
    empty_b.allocation_lease().validate_live().unwrap();
    let tolerance = if provider.precision == crate::backend::wgpu::types::NumericPrecision::F64 {
        1.0e-8
    } else {
        1.0e-3
    };
    let warm_ep = MatmulEpilogue::noop();
    for attempt in 0..2 {
        let selected_ep = if attempt == 0 { &ep } else { &warm_ep };
        let selected_expected = if attempt == 0 {
            &expected
        } else {
            &warm_expected
        };
        let observed = receipt(provider);
        let before_submit = observed.recorder().submission_checkpoint();
        let output = observe_provider_future(
            Some(observed.recorder()),
            provider.matmul_epilogue(&a, &b, selected_ep),
        )
        .await
        .unwrap();
        assert!(observed.recorder().submission_checkpoint() > before_submit);
        assert_eq!(output.shape(), &[2, columns]);
        assert_eq!(output.device_id(), provider.device_id());
        assert_eq!(
            output.physical().element_type(),
            if provider.precision == crate::backend::wgpu::types::NumericPrecision::F64 {
                NumericElementType::F64
            } else {
                NumericElementType::F32
            }
        );
        for input in [&a, &b, &row, &col] {
            assert!(!output.shares_allocation_with(input));
            input.allocation_lease().validate_live().unwrap();
        }
        let host = provider.download(&output).await.unwrap();
        assert_eq!(host.shape, [2, columns]);
        assert_eq!(host.data.len(), selected_expected.len());
        for (actual, wanted) in host.data.iter().zip(selected_expected) {
            assert!(
                (actual - wanted).abs() <= tolerance * wanted.abs().max(1.0),
                "{actual} != {wanted}"
            );
        }
        assert_eq!(
            provider.telemetry_snapshot().fusion_cache_misses,
            misses + 1
        );
        assert_eq!(
            provider.telemetry_snapshot().fusion_cache_hits,
            hits + attempt as u64
        );
        assert_eq!(
            provider.validated_fusion_cache.lock().unwrap().len(),
            cache + 1
        );
    }
    let transposed_a = provider.transpose_exec(&a).unwrap();
    let observed = receipt(provider);
    let checkpoint = observed.recorder().submission_checkpoint();
    let transposed = observe_provider_future(
        Some(observed.recorder()),
        provider.matmul_epilogue(&transposed_a, &b, &warm_ep),
    )
    .await
    .unwrap();
    assert!(observed.recorder().submission_checkpoint() > checkpoint);
    let transposed_host = provider.download(&transposed).await.unwrap();
    assert_eq!(transposed_host.shape, [2, columns]);
    for column in 0..columns {
        let x = b_data[2 * column];
        let y = b_data[2 * column + 1];
        for (row, wanted) in [x + 2.0 * y, 3.0 * x + 4.0 * y].iter().enumerate() {
            let actual = transposed_host.data[2 * column + row];
            assert!((actual - wanted).abs() <= tolerance * wanted.abs().max(1.0));
        }
    }
    assert_eq!(provider.telemetry_snapshot().fusion_cache_hits, hits + 2);
    assert!(!transposed.shares_allocation_with(&transposed_a));
    transposed_a.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(not(target_arch = "wasm32"), ignore = "requires actual WGPU adapter")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_matmul_epilogue_validated_cold_warm_and_refusals() {
    qualification(provider().await).await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn pending_epilogue_preparation_cancel_retains_borrowed_inputs() {
    use std::future::Future;
    let provider = provider().await;
    let a = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &[2, 2],
        })
        .unwrap();
    let b = provider
        .upload(&HostTensorView {
            data: &[1.0, 0.0, 0.0, 1.0],
            shape: &[2, 2],
        })
        .unwrap();
    let ep = MatmulEpilogue::noop();
    let allocations = provider.allocations.record_count().unwrap();
    let mut future = Box::pin(provider.matmul_epilogue(&a, &b, &ep));
    let pending = std::future::poll_fn(|context| {
        std::task::Poll::Ready(future.as_mut().poll(context).is_pending())
    })
    .await;
    assert!(pending);
    drop(future);
    assert_eq!(provider.allocations.record_count().unwrap(), allocations);
    assert!(provider.validated_fusion_cache.lock().unwrap().is_empty());
    a.allocation_lease().validate_live().unwrap();
    b.allocation_lease().validate_live().unwrap();
    let output = provider.matmul_epilogue(&a, &b, &ep).await.unwrap();
    let host = provider.download(&output).await.unwrap();
    assert_eq!(host.shape, [2, 2]);
    assert_eq!(host.data, [1.0, 2.0, 3.0, 4.0]);
}
