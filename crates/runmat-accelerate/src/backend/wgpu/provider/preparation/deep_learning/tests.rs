use super::*;
use runmat_accelerate_api::{
    AccelProvider, DeepLearningForwardParameter, DeepLearningForwardParameters,
    DeepLearningForwardSource, DeepLearningForwardStep, GpuTensorHandle, GpuTensorStorage,
    GpuTensorView, HostNumericDataView, HostNumericTensorView, NumericElementType,
    ProviderViewStep,
};

fn upload(provider: &WgpuProvider, values: &[f64], shape: &[usize]) -> GpuTensorHandle {
    let single: Vec<f32> = values.iter().map(|value| *value as f32).collect();
    let data = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => HostNumericDataView::F32(&single),
        runmat_accelerate_api::ProviderPrecision::F64 => HostNumericDataView::F64(values),
    };
    provider
        .upload_numeric(&HostNumericTensorView {
            data,
            shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
}

fn host_route(
    element_type: NumericElementType,
    stage: DeepLearningFeedForwardStage,
) -> DeepLearningForwardRoute<'static> {
    DeepLearningForwardRoute {
        input: DeepLearningForwardSource::Host {
            element_type,
            shape: vec![2, 2],
        },
        parameters: vec![],
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::FeedForward(stage),
            parameters: DeepLearningForwardParameters::None,
        }],
        output_view: GpuTensorView::contiguous(vec![2, 2]),
    }
}

#[test]
#[ignore = "requires an actual WGPU device; execute explicitly for qualification"]
fn exact_relu_route_changes_from_cold_to_warm_after_execution() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let (element_type, data) = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => (
            NumericElementType::F32,
            HostNumericDataView::F32(&[1.0_f32, -2.0, 3.0, 4.0]),
        ),
        runmat_accelerate_api::ProviderPrecision::F64 => (
            NumericElementType::F64,
            HostNumericDataView::F64(&[1.0_f64, -2.0, 3.0, 4.0]),
        ),
    };
    let route = host_route(
        element_type,
        DeepLearningFeedForwardStage::Relu { elements: 4 },
    );
    let cold = provider
        .deep_learning_forward_preparation_snapshot(&route)
        .expect("fully audited ReLU route");
    assert_eq!(cold.state, Some(CandidatePreparationState::Cold));
    assert_eq!((cold.cold_pipelines, cold.warm_pipelines), (1, 0));
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data,
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let _output = provider.scalar_max(&input, 0.0).unwrap();
    let warm = provider
        .deep_learning_forward_preparation_snapshot(&route)
        .expect("same admitted route");
    assert_eq!(warm.identity, cold.identity);
    assert_eq!(warm.state, Some(CandidatePreparationState::Warm));
    assert_eq!((warm.cold_pipelines, warm.warm_pipelines), (0, 1));
}

#[test]
#[ignore = "requires an actual WGPU device; execute explicitly for qualification"]
fn contended_dynamic_cache_is_unavailable_not_warm() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let element_type = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => NumericElementType::F32,
        runmat_accelerate_api::ProviderPrecision::F64 => NumericElementType::F64,
    };
    let route = host_route(
        element_type,
        DeepLearningFeedForwardStage::Elu { elements: 4 },
    );
    let _guard = provider.fused_pipeline_cache.lock().unwrap();
    let snapshot = provider
        .deep_learning_forward_preparation_snapshot(&route)
        .expect("route identity remains available during contention");
    assert_eq!(snapshot.state, None);
    assert_eq!((snapshot.cold_pipelines, snapshot.warm_pipelines), (0, 0));
}

#[test]
#[ignore = "requires an actual WGPU device; execute explicitly for qualification"]
fn dense_route_inspects_actual_matmul_and_bias_broadcast_bundles() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let element_type = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => NumericElementType::F32,
        runmat_accelerate_api::ProviderPrecision::F64 => NumericElementType::F64,
    };
    let route = DeepLearningForwardRoute {
        input: DeepLearningForwardSource::Host {
            element_type,
            shape: vec![2, 2],
        },
        parameters: vec![
            DeepLearningForwardParameter {
                source: DeepLearningForwardSource::Host {
                    element_type,
                    shape: vec![2, 2],
                },
                view_steps: vec![],
            },
            DeepLearningForwardParameter {
                source: DeepLearningForwardSource::Host {
                    element_type,
                    shape: vec![1, 2],
                },
                view_steps: vec![],
            },
        ],
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::FeedForward(
                DeepLearningFeedForwardStage::Dense {
                    batch: 2,
                    input_channels: 2,
                    output_channels: 2,
                },
            ),
            parameters: DeepLearningForwardParameters::Dense {
                weights: 0,
                bias: 1,
            },
        }],
        output_view: GpuTensorView::contiguous(vec![2, 2]),
    };
    let cold = provider
        .deep_learning_forward_preparation_snapshot(&route)
        .expect("host-origin dense route is fully audited");
    assert_eq!(cold.state, Some(CandidatePreparationState::Cold));
    assert_eq!((cold.cold_pipelines, cold.warm_pipelines), (2, 0));
    let input = upload(&provider, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let weights = upload(&provider, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let bias = upload(&provider, &[1.0, 2.0], &[1, 2]);
    let product = futures::executor::block_on(provider.matmul(&input, &weights)).unwrap();
    let _output = futures::executor::block_on(provider.elem_add(&product, &bias)).unwrap();
    let warm = provider
        .deep_learning_forward_preparation_snapshot(&route)
        .expect("same host-origin dense route");
    assert_eq!(warm.identity, cold.identity);
    assert_eq!(warm.state, Some(CandidatePreparationState::Warm));
    assert_eq!((warm.cold_pipelines, warm.warm_pipelines), (0, 2));
}

#[test]
#[ignore = "requires an actual WGPU device; execute explicitly for qualification"]
fn resident_parameter_view_materialization_is_counted_without_dispatch() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let element_type = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => NumericElementType::F32,
        runmat_accelerate_api::ProviderPrecision::F64 => NumericElementType::F64,
    };
    let bias_base = upload(&provider, &[1.0, 2.0], &[2, 1]);
    let lazy_bias = provider.transpose(&bias_base).unwrap();
    let route = DeepLearningForwardRoute {
        input: DeepLearningForwardSource::Host {
            element_type,
            shape: vec![2, 2],
        },
        parameters: vec![
            DeepLearningForwardParameter {
                source: DeepLearningForwardSource::Host {
                    element_type,
                    shape: vec![2, 2],
                },
                view_steps: vec![],
            },
            DeepLearningForwardParameter {
                source: DeepLearningForwardSource::Resident(&lazy_bias),
                view_steps: vec![ProviderViewStep::Reshape(vec![1, 2])],
            },
        ],
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::FeedForward(
                DeepLearningFeedForwardStage::Dense {
                    batch: 2,
                    input_channels: 2,
                    output_channels: 2,
                },
            ),
            parameters: DeepLearningForwardParameters::Dense {
                weights: 0,
                bias: 1,
            },
        }],
        output_view: GpuTensorView::contiguous(vec![2, 2]),
    };
    let snapshot = provider
        .deep_learning_forward_preparation_snapshot(&route)
        .expect("view-chain materialization is admitted");
    assert_eq!(snapshot.state, Some(CandidatePreparationState::Cold));
    assert_eq!((snapshot.cold_pipelines, snapshot.warm_pipelines), (3, 0));
}
