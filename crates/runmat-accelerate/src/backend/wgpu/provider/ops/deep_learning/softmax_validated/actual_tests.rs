use super::*;
use crate::wgsl::f64_math::PreparationError;
use runmat_accelerate_api::observation::{observe_provider_future, ProviderOperationReceipt};
use runmat_accelerate_api::{
    AccelProvider, DeepLearningFeedForwardCapability, DeepLearningFeedForwardStage,
    DeepLearningForwardOperation, DeepLearningForwardParameters, DeepLearningForwardRoute,
    DeepLearningForwardSource, DeepLearningForwardStep, GpuTensorView, HostIntegerDataView,
    HostIntegerTensorView, HostLogicalTensorView, HostNumericDataView, HostNumericTensorView,
    NumericElementType,
};
use runmat_execution::CandidatePreparationState;

fn operation_receipt(provider: &WgpuProvider) -> ProviderOperationReceipt {
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

async fn owner_with_scalar(scalar: Option<u32>) -> &'static WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let mut provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let mut provider = WgpuProvider::new_async(Default::default())
        .await
        .expect("actual WebGPU adapter");
    if let Some(scalar) = scalar {
        assert!(scalar <= provider.adapter_limits.max_compute_workgroup_size_x);
        assert!(
            scalar
                <= provider
                    .adapter_limits
                    .max_compute_invocations_per_workgroup
        );
        provider.workgroup_config.scalar = scalar;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        provider.pipeline_cache_dir = Some(std::env::temp_dir().join(format!(
            "runmat-validated-softmax-{}-{nonce}",
            std::process::id()
        )));
    }
    let provider = Box::leak(Box::new(provider));
    // SAFETY: this actual provider lives for the process.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

fn fixture(workgroup: u32) -> (Vec<f64>, [usize; 2], Vec<f64>) {
    let columns = workgroup as usize + 1;
    let shape = [2usize, columns];
    // Different periodicities prevent row mixing from passing. The last
    // column is distinct and forces the shader's second column-loop pass.
    let logits: Vec<[f64; 2]> = (0..columns)
        .map(|column| {
            [
                (column % 7) as f64 * 0.5 - 1.5,
                (column % 5) as f64 * 0.75 - 1.5 + if column + 1 == columns { 2.0 } else { 0.0 },
            ]
        })
        .collect();
    let data = logits.iter().flat_map(|pair| *pair).collect();
    let maximum = [0, 1].map(|row| {
        logits
            .iter()
            .map(|pair| pair[row])
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let denominator = [0, 1].map(|row| {
        logits
            .iter()
            .map(|pair| (pair[row] - maximum[row]).exp())
            .sum::<f64>()
    });
    let expected = logits
        .iter()
        .flat_map(|pair| {
            [
                (pair[0] - maximum[0]).exp() / denominator[0],
                (pair[1] - maximum[1]).exp() / denominator[1],
            ]
        })
        .collect();
    (data, shape, expected)
}

async fn assert_typed_input_refused(
    provider: &'static WgpuProvider,
    input: &GpuTensorHandle,
    expected_class: NumericElementType,
    expected_logical: bool,
    expected_logical_producer: bool,
) {
    assert_eq!(input.physical().element_type(), expected_class);
    assert_eq!(
        runmat_accelerate_api::handle_is_logical(input),
        expected_logical
    );
    assert_eq!(
        provider
            .allocations
            .requires_logical_encoding(input)
            .unwrap(),
        expected_logical_producer
    );
    assert_eq!(
        provider
            .deep_learning_feed_forward_stage_capability(
                input,
                DeepLearningFeedForwardStage::SoftmaxRows {
                    rows: input.shape()[0],
                    columns: input.shape()[1],
                },
            )
            .unwrap(),
        DeepLearningFeedForwardCapability::UnsupportedOperation
    );
    let allocations = provider.allocations.record_count().unwrap();
    let cache_entries = provider.validated_fusion_cache.lock().unwrap().len();
    let metrics = provider.telemetry_snapshot();
    let receipt = operation_receipt(provider);
    let checkpoint = receipt.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(receipt.recorder()),
        provider.activation_softmax_rows(input),
    )
    .await
    .expect_err("nonmatching producer or physical class must be refused");
    assert!(error.is_unsupported());
    assert_eq!(receipt.recorder().submission_checkpoint(), checkpoint);
    assert_eq!(provider.allocations.record_count().unwrap(), allocations);
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        cache_entries
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        metrics.fusion_cache_misses
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_hits,
        metrics.fusion_cache_hits
    );
    assert_eq!(input.physical().element_type(), expected_class);
    assert_eq!(
        runmat_accelerate_api::handle_is_logical(input),
        expected_logical
    );
    assert_eq!(
        provider
            .allocations
            .requires_logical_encoding(input)
            .unwrap(),
        expected_logical_producer
    );
    input.allocation_lease().validate_live().unwrap();
}

async fn cold_warm_and_invalid_scope(provider: &'static WgpuProvider) {
    let integer = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I32(&[1, 2, 3, 4]),
            shape: &[2, 2],
        })
        .unwrap();
    assert_typed_input_refused(provider, &integer, NumericElementType::I32, false, false).await;
    let empty_integer = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I32(&[]),
            shape: &[0, 1],
        })
        .unwrap();
    assert_typed_input_refused(
        provider,
        &empty_integer,
        NumericElementType::I32,
        false,
        false,
    )
    .await;
    let opposite_class = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => NumericElementType::F64,
        runmat_accelerate_api::ProviderPrecision::F64 => NumericElementType::F32,
    };
    let opposite_data = match opposite_class {
        NumericElementType::F32 => HostNumericDataView::F32(&[1.0_f32, 2.0, 3.0, 4.0]),
        NumericElementType::F64 => HostNumericDataView::F64(&[1.0_f64, 2.0, 3.0, 4.0]),
        _ => unreachable!("opposite class is floating"),
    };
    let mismatched = provider
        .upload_numeric(&HostNumericTensorView {
            data: opposite_data,
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert_typed_input_refused(provider, &mismatched, opposite_class, false, false).await;
    let empty_opposite_data = match opposite_class {
        NumericElementType::F32 => HostNumericDataView::F32(&[]),
        NumericElementType::F64 => HostNumericDataView::F64(&[]),
        _ => unreachable!("opposite class is floating"),
    };
    let empty_mismatched = provider
        .upload_numeric(&HostNumericTensorView {
            data: empty_opposite_data,
            shape: &[0, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert_typed_input_refused(provider, &empty_mismatched, opposite_class, false, false).await;
    let native_class = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => NumericElementType::F32,
        runmat_accelerate_api::ProviderPrecision::F64 => NumericElementType::F64,
    };
    let logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 0, 0, 1],
            shape: &[2, 2],
        })
        .unwrap();
    assert_typed_input_refused(provider, &logical, native_class, true, true).await;
    let logical_numeric_view = logical
        .with_view(logical.view().clone().with_logical(false))
        .unwrap();
    assert!(logical_numeric_view.shares_allocation_with(&logical));
    assert_typed_input_refused(provider, &logical_numeric_view, native_class, false, true).await;
    let empty_logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[],
            shape: &[0, 1],
        })
        .unwrap();
    assert_typed_input_refused(provider, &empty_logical, native_class, true, true).await;
    let empty_logical_numeric_view = empty_logical
        .with_view(empty_logical.view().clone().with_logical(false))
        .unwrap();
    assert!(empty_logical_numeric_view.shares_allocation_with(&empty_logical));
    assert_typed_input_refused(
        provider,
        &empty_logical_numeric_view,
        native_class,
        false,
        true,
    )
    .await;
    let explicit_numeric = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => {
            provider.unary_single(&logical).await.unwrap()
        }
        runmat_accelerate_api::ProviderPrecision::F64 => {
            provider.unary_double(&logical).await.unwrap()
        }
    };
    assert!(!explicit_numeric.shares_allocation_with(&logical));
    assert!(!runmat_accelerate_api::handle_is_logical(&explicit_numeric));
    assert!(!provider
        .allocations
        .requires_logical_encoding(&explicit_numeric)
        .unwrap());
    assert_eq!(explicit_numeric.physical().element_type(), native_class);
    assert_eq!(
        provider
            .deep_learning_feed_forward_stage_capability(
                &explicit_numeric,
                DeepLearningFeedForwardStage::SoftmaxRows {
                    rows: 2,
                    columns: 2
                },
            )
            .unwrap(),
        DeepLearningFeedForwardCapability::Ready
    );
    let (data, shape, expected) = fixture(provider.workgroup_config.scalar);
    let input = provider
        .upload(&HostTensorView {
            data: &data,
            shape: &shape,
        })
        .unwrap();
    let expected_input_class = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => NumericElementType::F32,
        runmat_accelerate_api::ProviderPrecision::F64 => NumericElementType::F64,
    };
    assert_eq!(input.physical().element_type(), expected_input_class);
    let invalid_shape = [1usize, 1, 1];
    let invalid_input = provider
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &invalid_shape,
        })
        .unwrap();
    let before_allocations = provider.allocations.record_count().unwrap();
    let before_cache = provider.validated_fusion_cache.lock().unwrap().len();
    let before_misses = provider.telemetry_snapshot().fusion_cache_misses;
    let before_hits = provider.telemetry_snapshot().fusion_cache_hits;
    let invalid_receipt = operation_receipt(provider);
    let invalid_checkpoint = invalid_receipt.recorder().submission_checkpoint();
    let invalid_error = observe_provider_future(
        Some(invalid_receipt.recorder()),
        provider.activation_softmax_rows(&invalid_input),
    )
    .await
    .expect_err("invalid public input shape must fail before preparation");
    assert!(invalid_error.to_string().contains("2-D tensor"));
    assert_eq!(
        invalid_receipt.recorder().submission_checkpoint(),
        invalid_checkpoint
    );
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        before_allocations
    );
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before_cache
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        before_misses
    );
    invalid_input.allocation_lease().validate_live().unwrap();
    let invalid = super::super::super::pipeline_recipe::PipelineRecipe {
        shader: "@compute @workgroup_size(1) fn main() { let bad = ; }".into(),
        layout: crate::backend::wgpu::bindings::SOFTMAX_ROWS_LAYOUT_TAG.into(),
        workgroup: provider.workgroup_config.scalar,
    };
    let receipt = operation_receipt(provider);
    let checkpoint = receipt.recorder().submission_checkpoint();
    let error = observe_provider_future(
        Some(receipt.recorder()),
        provider.prepare_validated_softmax_rows_pipeline(&invalid),
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
    assert_eq!(
        provider.allocations.record_count().unwrap(),
        before_allocations
    );
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before_cache
    );
    assert_eq!(
        provider.telemetry_snapshot().fusion_cache_misses,
        before_misses
    );
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = provider.pipeline_cache_dir.as_ref().unwrap();
        assert!(!dir.join(format!("{:016x}.wgsl", invalid.key())).exists());
        assert!(!dir.join(format!("{:016x}.json", invalid.key())).exists());
    }
    let tolerance = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-10,
        runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
    };
    assert_eq!(
        provider
            .deep_learning_feed_forward_stage_capability(
                &input,
                DeepLearningFeedForwardStage::SoftmaxRows {
                    rows: shape[0],
                    columns: shape[1],
                },
            )
            .unwrap(),
        DeepLearningFeedForwardCapability::Ready
    );
    let route = DeepLearningForwardRoute {
        input: DeepLearningForwardSource::Host {
            element_type: expected_input_class,
            shape: shape.to_vec(),
        },
        parameters: vec![],
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::FeedForward(
                DeepLearningFeedForwardStage::SoftmaxRows {
                    rows: shape[0],
                    columns: shape[1],
                },
            ),
            parameters: DeepLearningForwardParameters::None,
        }],
        output_view: GpuTensorView::contiguous(shape.to_vec()),
    };
    let cold = provider
        .deep_learning_forward_preparation_snapshot(&route)
        .expect("softmax preparation route");
    assert_eq!(cold.state, Some(CandidatePreparationState::Cold));
    assert_eq!((cold.cold_pipelines, cold.warm_pipelines), (1, 0));
    for attempt in 0..2 {
        let receipt = operation_receipt(provider);
        let checkpoint = receipt.recorder().submission_checkpoint();
        let output = observe_provider_future(
            Some(receipt.recorder()),
            provider.activation_softmax_rows(&input),
        )
        .await
        .unwrap();
        assert!(
            receipt.recorder().submission_checkpoint() > checkpoint,
            "public softmax must register its own GPU submission"
        );
        assert_eq!(output.shape(), shape);
        assert_eq!(output.device_id(), provider.device_id());
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            GpuTensorStorage::Real
        );
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            Some(provider.precision())
        );
        assert!(!output.shares_allocation_with(&input));
        output.allocation_lease().validate_live().unwrap();
        let host = provider.download(&output).await.unwrap();
        assert_eq!(host.shape, shape);
        assert_eq!(host.data.len(), expected.len());
        for (actual, expected) in host.data.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() < tolerance,
                "got {actual}, expected {expected}"
            );
        }
        assert_eq!(
            provider.telemetry_snapshot().fusion_cache_misses,
            before_misses + 1
        );
        assert_eq!(
            provider.telemetry_snapshot().fusion_cache_hits,
            before_hits + attempt as u64
        );
    }
    let warm = provider
        .deep_learning_forward_preparation_snapshot(&route)
        .expect("same validated softmax route");
    assert_eq!(warm.identity, cold.identity);
    assert_eq!(warm.state, Some(CandidatePreparationState::Warm));
    assert_eq!((warm.cold_pipelines, warm.warm_pipelines), (0, 1));
    assert_eq!(
        provider.validated_fusion_cache.lock().unwrap().len(),
        before_cache + 1
    );
    input.allocation_lease().validate_live().unwrap();
    let cast_softmax = provider
        .activation_softmax_rows(&explicit_numeric)
        .await
        .unwrap();
    assert_eq!(cast_softmax.shape(), &[2, 2]);
    assert_eq!(cast_softmax.device_id(), provider.device_id());
    assert_eq!(cast_softmax.physical().element_type(), native_class);
    assert!(!runmat_accelerate_api::handle_is_logical(&cast_softmax));
    assert!(!provider
        .allocations
        .requires_logical_encoding(&cast_softmax)
        .unwrap());
    assert!(!cast_softmax.shares_allocation_with(&explicit_numeric));
    let cast_host = provider.download(&cast_softmax).await.unwrap();
    let p = std::f64::consts::E / (1.0 + std::f64::consts::E);
    let q = 1.0 - p;
    let cast_tolerance = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
        runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-10,
    };
    assert_eq!(cast_host.shape, [2, 2]);
    assert_eq!(cast_host.data.len(), 4);
    for (actual, expected) in cast_host.data.iter().zip([p, q, q, p]) {
        assert!((actual - expected).abs() < cast_tolerance);
    }
    logical.allocation_lease().validate_live().unwrap();
    explicit_numeric.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(not(target_arch = "wasm32"), ignore = "requires actual WGPU adapter")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_softmax_rows_validated_cold_warm_and_invalid_scope() {
    cold_warm_and_invalid_scope(owner_with_scalar(None).await).await;
    cold_warm_and_invalid_scope(owner_with_scalar(Some(256)).await).await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn pending_softmax_preparation_cancel_retains_borrowed_input() {
    use std::future::Future;
    let provider = owner_with_scalar(None).await;
    let (data, shape, expected) = fixture(provider.workgroup_config.scalar);
    let input = provider
        .upload(&HostTensorView {
            data: &data,
            shape: &shape,
        })
        .unwrap();
    let mut future = Box::pin(provider.activation_softmax_rows(&input));
    let pending = std::future::poll_fn(|context| {
        std::task::Poll::Ready(future.as_mut().poll(context).is_pending())
    })
    .await;
    assert!(pending);
    drop(future);
    assert!(provider.validated_fusion_cache.lock().unwrap().is_empty());
    input.allocation_lease().validate_live().unwrap();
    let output = provider.activation_softmax_rows(&input).await.unwrap();
    let host = provider.download(&output).await.unwrap();
    assert_eq!(host.shape, shape);
    assert_eq!(host.data.len(), expected.len());
    for (actual, expected) in host.data.iter().zip(&expected) {
        assert!((actual - expected).abs() < 1.0e-5);
    }
    input.allocation_lease().validate_live().unwrap();
}
