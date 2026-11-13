use super::*;
use crate::wgsl::f64_math::PreparationError;
use runmat_accelerate_api::observation::{observe_provider_future, ProviderOperationReceipt};
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataView, HostIntegerTensorView, HostLogicalTensorView,
    HostNumericDataOwned, HostNumericDataView, HostNumericTensorView, NumericElementType,
};

#[cfg(not(target_arch = "wasm32"))]
#[path = "tests/metal_observation_diagnostic.rs"]
mod metal_observation_diagnostic;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_elementwise_limits_refuse_before_driver_creation() {
    let limits = wgpu::Limits::default();
    let launch = FrozenLinearLaunch::new(256, limits.max_compute_workgroups_per_dimension)
        .expect("valid selected launch");
    assert!(check_kernel_limits("round_digits", &limits, 2, 3, 16, &[1024, 1024], launch).is_ok());
    let too_many_bindings = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 1,
        ..limits.clone()
    };
    assert!(check_kernel_limits(
        "round_digits",
        &too_many_bindings,
        2,
        3,
        16,
        &[8, 8],
        launch
    )
    .is_err());
    let too_small = wgpu::Limits {
        max_storage_buffer_binding_size: 8,
        ..limits.clone()
    };
    assert!(check_kernel_limits("round_digits", &too_small, 2, 3, 16, &[12, 12], launch).is_err());
    let workgroup_limited = wgpu::Limits {
        max_compute_workgroup_size_x: 128,
        max_compute_invocations_per_workgroup: 128,
        ..limits
    };
    assert!(check_kernel_limits(
        "round_digits",
        &workgroup_limited,
        2,
        3,
        16,
        &[8, 8],
        launch,
    )
    .is_err());
}

fn receipt(provider: &WgpuProvider) -> ProviderOperationReceipt {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let progress = provider
            .observation_progress
            .as_ref()
            .expect("actual WGPU provider has a native progress port")
            .clone();
        ProviderOperationReceipt::with_wgpu_progress(provider.observation_budget.clone(), progress)
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
    let mut provider = WgpuProvider::new_async(Default::default())
        .await
        .expect("actual browser WebGPU adapter");
    assert_eq!(provider.precision, NumericPrecision::F32);
    assert!(provider.adapter_limits.max_compute_workgroup_size_x >= 256);
    assert!(
        provider
            .adapter_limits
            .max_compute_invocations_per_workgroup
            >= 256
    );
    provider.linear_launch = FrozenLinearLaunch::new(
        256,
        provider.adapter_limits.max_compute_workgroups_per_dimension,
    )
    .expect("selected 256-lane launch");
    provider.workgroup_config.scalar = 256;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        provider.pipeline_cache_dir = Some(std::env::temp_dir().join(format!(
            "runmat-scoped-elementwise-{}-{nonce}",
            std::process::id()
        )));
    }
    let provider = Box::leak(Box::new(provider));
    // SAFETY: this actual provider remains live for the process or browser worker.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

fn expected_decimal(index: usize) -> f64 {
    match index % 6 {
        0 => 12.3,
        1 => -1.3,
        2 => 2.3,
        3 => -3.5,
        4 => -0.0,
        _ => f64::INFINITY,
    }
}

fn assert_words(context: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected == 0.0 {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{context}, lane {index}"
            );
        } else if expected.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
    }
}

async fn scoped_cold_warm_numeric_and_invalid_scope() {
    let provider = owner().await;
    let pattern = [12.34_f32, -1.26, 2.34, -3.46, -0.0, f32::INFINITY];
    let real_data: Vec<_> = (0..257)
        .map(|index| pattern[index % pattern.len()])
        .collect();
    let real = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&real_data),
            shape: &[1, 257],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let expected: Vec<_> = (0..real_data.len()).map(expected_decimal).collect();
    let uploaded = provider.download_numeric(&real).await.unwrap();
    let HostNumericDataOwned::F32(uploaded) = uploaded.data else {
        panic!("raw rounding input must retain Single storage");
    };
    assert_eq!(
        uploaded
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        real_data
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "raw uploaded rounding input must retain every bit"
    );
    let frame = RoundDigitsFrame::plan(provider, &real, 1, false).unwrap();
    assert_eq!(frame.recipe.workgroup, 256);
    assert!(frame.recipe.shader.contains("@workgroup_size(256)"));
    let bad = PipelineRecipe {
        shader: "@compute @workgroup_size(1) fn main() { let invalid = ; }".into(),
        layout: frame.recipe.layout.clone(),
        workgroup: frame.recipe.workgroup,
    };
    let before_cache = provider.validated_fusion_cache.lock().unwrap().len();
    let before_allocations = provider.allocations.record_count().unwrap();
    let before_metrics = provider.telemetry_snapshot();
    let failure_receipt = receipt(provider);
    let checkpoint = failure_receipt.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(failure_receipt.recorder()),
        provider.prepare_validated_round_digits_pipeline(&bad, false),
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
    assert_eq!(
        failure_receipt.recorder().submission_checkpoint(),
        checkpoint
    );
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before_cache
    );
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        before_allocations
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        before_metrics.fusion_cache_misses
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_hits,
        before_metrics.fusion_cache_hits
    );
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = provider.pipeline_cache_dir.as_ref().unwrap();
        assert!(!dir.join(format!("{:016x}.wgsl", bad.key())).exists());
        assert!(!dir.join(format!("{:016x}.json", bad.key())).exists());
    }

    for attempt in 0..2 {
        let before = provider.telemetry_snapshot();
        let operation = receipt(provider);
        let checkpoint = operation.recorder().submission_checkpoint();
        let output = observe_provider_future(
            Some(operation.recorder()),
            provider.round_digits(&real, 1, false),
        )
        .await
        .unwrap();
        assert!(operation.recorder().submission_checkpoint() > checkpoint);
        assert_eq!(output.shape(), [1, 257]);
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        assert_eq!(output.physical().storage(), GpuTensorStorage::Real);
        assert_eq!(output.physical().requested_bytes(), 257 * 4);
        assert!(!output.shares_allocation_with(&real));
        let host = provider.download(&output).await.unwrap();
        assert_eq!(host.shape, [1, 257]);
        assert_eq!(host.storage, GpuTensorStorage::Real);
        assert_words("decimal cold/warm", &host.data, &expected);
        assert_eq!(
            provider.telemetry_snapshot().fusion_cache_misses,
            before.fusion_cache_misses + u64::from(attempt == 0),
        );
        assert_eq!(
            provider.telemetry_snapshot().fusion_cache_hits,
            before.fusion_cache_hits + u64::from(attempt == 1),
        );
        real.allocation_lease().validate_live().unwrap();
    }

    let significant = provider.round_digits(&real, 2, true).await.unwrap();
    let significant_host = provider.download(&significant).await.unwrap();
    let significant_expected: Vec<f64> = (0..real_data.len())
        .map(|index| {
            if index % 6 == 0 {
                12.0
            } else if index % 6 == 4 {
                0.0
            } else {
                expected_decimal(index)
            }
        })
        .collect();
    assert_words(
        "significant digits",
        &significant_host.data,
        &significant_expected,
    );
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let allocations = provider.allocations.record_count().unwrap();
    let invalid_digits = provider
        .round_digits(&real, 0, true)
        .await
        .expect_err("significant digits require positive count");
    assert!(invalid_digits.is_unsupported());
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(provider.allocations.record_count().unwrap(), allocations);

    let complex_data: Vec<_> = (0..258)
        .map(|index| pattern[index % pattern.len()])
        .collect();
    let complex = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&complex_data),
            shape: &[1, 129],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let complex_frame = RoundDigitsFrame::plan(provider, &complex, 1, false).unwrap();
    assert_eq!(complex_frame.input.len, 258);
    assert_eq!(complex_frame.scalar_len, 258);
    assert_eq!(complex_frame.allocated_bytes, 258 * 4);
    assert_eq!(complex_frame.input.shape.as_slice(), &[1, 129]);
    assert_eq!(complex_frame.storage, GpuTensorStorage::ComplexInterleaved);
    let complex_output = provider.round_digits(&complex, 1, false).await.unwrap();
    assert_eq!(complex_output.shape(), [1, 129]);
    assert_eq!(
        complex_output.physical().storage(),
        GpuTensorStorage::ComplexInterleaved
    );
    assert_eq!(
        complex_output.physical().element_type(),
        NumericElementType::F32
    );
    assert_eq!(complex_output.physical().requested_bytes(), 258 * 4);
    assert!(!complex_output.shares_allocation_with(&complex));
    let complex_host = provider.download(&complex_output).await.unwrap();
    assert_eq!(complex_host.storage, GpuTensorStorage::ComplexInterleaved);
    assert_words(
        "complex decimal lanes",
        &complex_host.data,
        &(0..258).map(expected_decimal).collect::<Vec<_>>(),
    );
    complex.allocation_lease().validate_live().unwrap();

    let lhs_data: Vec<i64> = (0..257)
        .map(|index| [i64::MIN, 9_007_199_254_740_993, -1, 0, i64::MAX][index % 5])
        .collect();
    let rhs_data: Vec<i64> = (0..257)
        .map(|index| [0, 9_007_199_254_740_992, 0, 1, i64::MAX][index % 5])
        .collect();
    let lhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&lhs_data),
            shape: &[1, 257],
        })
        .unwrap();
    let rhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&rhs_data),
            shape: &[1, 257],
        })
        .unwrap();
    let comparison_frame = IntegerComparisonFrame::plan(provider, "elem_lt", &lhs, &rhs).unwrap();
    let invalid_comparison = PipelineRecipe {
        shader: "@compute @workgroup_size(1) fn main() { let invalid = ; }".into(),
        layout: comparison_frame.recipe.layout.clone(),
        workgroup: comparison_frame.recipe.workgroup,
    };
    let before_cache = provider.validated_fusion_cache.lock().unwrap().len();
    let before_allocations = provider.allocations.record_count().unwrap();
    let before_misses = provider.telemetry_snapshot().fusion_cache_misses;
    let operation = receipt(provider);
    let checkpoint = operation.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(operation.recorder()),
        provider.prepare_validated_integer_comparison_pipeline(&invalid_comparison),
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
    assert_eq!(operation.recorder().submission_checkpoint(), checkpoint);
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before_cache
    );
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        before_allocations
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        before_misses
    );
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = provider.pipeline_cache_dir.as_ref().unwrap();
        assert!(!dir
            .join(format!("{:016x}.wgsl", invalid_comparison.key()))
            .exists());
        assert!(!dir
            .join(format!("{:016x}.json", invalid_comparison.key()))
            .exists());
    }
    for attempt in 0..2 {
        let before = provider.telemetry_snapshot();
        let operation = receipt(provider);
        let checkpoint = operation.recorder().submission_checkpoint();
        let output =
            observe_provider_future(Some(operation.recorder()), provider.elem_lt(&lhs, &rhs))
                .await
                .unwrap();
        assert!(operation.recorder().submission_checkpoint() > checkpoint);
        assert_eq!(output.shape(), [1, 257]);
        assert!(runmat_accelerate_api::handle_is_logical(&output));
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        assert!(!output.shares_allocation_with(&lhs));
        assert!(!output.shares_allocation_with(&rhs));
        let host = provider.download(&output).await.unwrap();
        let expected: Vec<f64> = lhs_data
            .iter()
            .zip(&rhs_data)
            .map(|(&lhs, &rhs)| if lhs < rhs { 1.0 } else { 0.0 })
            .collect();
        assert_eq!(host.data, expected);
        assert_eq!(
            provider.telemetry_snapshot().fusion_cache_misses,
            before.fusion_cache_misses + u64::from(attempt == 0),
        );
        assert_eq!(
            provider.telemetry_snapshot().fusion_cache_hits,
            before.fusion_cache_hits + u64::from(attempt == 1),
        );
        lhs.allocation_lease().validate_live().unwrap();
        rhs.allocation_lease().validate_live().unwrap();
    }
    let logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 0],
            shape: &[1, 2],
        })
        .unwrap();
    let numeric_view = logical
        .with_view(logical.view().clone().with_logical(false))
        .unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let allocations = provider.allocations.record_count().unwrap();
    let rejected = provider
        .round_digits(&numeric_view, 1, false)
        .await
        .expect_err("logical producer must not masquerade as numeric");
    assert!(rejected.is_unsupported());
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(provider.allocations.record_count().unwrap(), allocations);
    logical.allocation_lease().validate_live().unwrap();
    let other_precision = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[12.34, -1.26]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let allocations = provider.allocations.record_count().unwrap();
    let error = provider
        .round_digits(&other_precision, 1, false)
        .await
        .expect_err("cross-precision owner must refuse native F32 shader");
    assert!(error.is_unsupported());
    assert_eq!(
        other_precision.physical().element_type(),
        NumericElementType::F64
    );
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(provider.allocations.record_count().unwrap(), allocations);
    other_precision.allocation_lease().validate_live().unwrap();
    let wrong_class = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&[0, 1]),
            shape: &[1, 2],
        })
        .unwrap();
    let small_lhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[0, 1]),
            shape: &[1, 2],
        })
        .unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let allocations = provider.allocations.record_count().unwrap();
    provider
        .elem_lt(&small_lhs, &wrong_class)
        .await
        .expect_err("mixed integer classes must be rejected before effects");
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(provider.allocations.record_count().unwrap(), allocations);
    small_lhs.allocation_lease().validate_live().unwrap();
    wrong_class.allocation_lease().validate_live().unwrap();
    let empty_real = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[]),
            shape: &[0, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let before_cache = provider.validated_fusion_cache.lock().unwrap().len();
    let empty_round = provider.round_digits(&empty_real, 1, false).await.unwrap();
    assert_eq!(empty_round.shape(), [0, 1]);
    assert_eq!(
        empty_round.physical().element_type(),
        NumericElementType::F32
    );
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before_cache
    );
    assert!(provider
        .download(&empty_round)
        .await
        .unwrap()
        .data
        .is_empty());
    let empty_integer = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[]),
            shape: &[0, 1],
        })
        .unwrap();
    let before_cache = provider.validated_fusion_cache.lock().unwrap().len();
    let empty_comparison = provider
        .elem_lt(&empty_integer, &empty_integer)
        .await
        .unwrap();
    assert_eq!(empty_comparison.shape(), [0, 1]);
    assert!(runmat_accelerate_api::handle_is_logical(&empty_comparison));
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before_cache
    );
    assert!(provider
        .download(&empty_comparison)
        .await
        .unwrap()
        .data
        .is_empty());
    empty_real.allocation_lease().validate_live().unwrap();
    empty_integer.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual F32 WGPU adapter"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_scoped_elementwise_comparison_and_rounding() {
    scoped_cold_warm_numeric_and_invalid_scope().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn pending_round_digits_cancellation_retains_input() {
    use std::future::Future;
    let provider = owner().await;
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[12.34, -1.26]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let mut future = Box::pin(provider.round_digits(&input, 1, false));
    let pending = std::future::poll_fn(|context| {
        std::task::Poll::Ready(future.as_mut().poll(context).is_pending())
    })
    .await;
    assert!(pending);
    drop(future);
    assert!(provider.validated_fusion_cache.lock().unwrap().is_empty());
    input.allocation_lease().validate_live().unwrap();
    let output = provider.round_digits(&input, 1, false).await.unwrap();
    assert_words(
        "cancelled decimal retry",
        &provider.download(&output).await.unwrap().data,
        &[12.3, -1.3],
    );
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn pending_integer_comparison_cancellation_retains_inputs() {
    use std::future::Future;
    let provider = owner().await;
    let lhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[i64::MIN, 9_007_199_254_740_993]),
            shape: &[1, 2],
        })
        .unwrap();
    let rhs = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[0, 9_007_199_254_740_992]),
            shape: &[1, 2],
        })
        .unwrap();
    let mut future = Box::pin(provider.elem_lt(&lhs, &rhs));
    let pending = std::future::poll_fn(|context| {
        std::task::Poll::Ready(future.as_mut().poll(context).is_pending())
    })
    .await;
    assert!(pending);
    drop(future);
    assert!(provider.validated_fusion_cache.lock().unwrap().is_empty());
    lhs.allocation_lease().validate_live().unwrap();
    rhs.allocation_lease().validate_live().unwrap();
    let output = provider.elem_lt(&lhs, &rhs).await.unwrap();
    assert_eq!(provider.download(&output).await.unwrap().data, [1.0, 0.0]);
}
