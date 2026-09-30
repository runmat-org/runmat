use super::*;
#[test]
#[ignore = "requires actual WGPU device; run explicitly for effect-free forward-route admission"]
fn admitted_resource_bounds_are_effect_free() {
    let fixture = Fixture::new();
    let provider = &fixture.provider;
    let class = fixture.class;
    let resident = fixture.resident();
    let before = provider.telemetry_snapshot();
    let pooling = DeepLearningForwardRoute {
        input: DeepLearningForwardSource::Host {
            element_type: class,
            shape: vec![2, 3],
        },
        parameters: Vec::new(),
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::Pool1D {
                input_shape: vec![2, 3],
                route: runmat_accelerate_api::DeepLearningPool1DRoute {
                    reduction: runmat_accelerate_api::DeepLearningPool1DRequest {
                        axis_zero_based: 1,
                    },
                    result_axis: runmat_accelerate_api::DeepLearningPool1DResultAxis::KeepSingleton,
                    source_shape: vec![2, 1],
                    target_shape: vec![2, 1],
                    target_to_source: vec![0, 1],
                },
            },
            parameters: DeepLearningForwardParameters::None,
        }],
        output_view: GpuTensorView::contiguous(vec![2, 1]),
    };
    let Some(DeepLearningForwardCapability::Ready(pool_resources)) = provider
        .deep_learning_forward_route_capability(&pooling)
        .unwrap()
    else {
        panic!("admitted pooling should remain ready")
    };
    assert_eq!(
        pool_resources.new_buffer_bytes_upper_bound,
        DeepLearningForwardBufferBound::Known(8 * class.element_size() as u64 + 16),
    );
    let lazy_pool_input = provider.transpose(resident).unwrap();
    let lazy_pool_route = DeepLearningForwardRoute {
        input: DeepLearningForwardSource::Resident(&lazy_pool_input),
        parameters: Vec::new(),
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::Pool1D {
                input_shape: vec![3, 2],
                route: runmat_accelerate_api::DeepLearningPool1DRoute {
                    reduction: runmat_accelerate_api::DeepLearningPool1DRequest {
                        axis_zero_based: 1,
                    },
                    result_axis: runmat_accelerate_api::DeepLearningPool1DResultAxis::KeepSingleton,
                    source_shape: vec![3, 1],
                    target_shape: vec![3, 1],
                    target_to_source: vec![0, 1],
                },
            },
            parameters: DeepLearningForwardParameters::None,
        }],
        output_view: GpuTensorView::contiguous(vec![3, 1]),
    };
    assert!(
        provider
            .deep_learning_forward_route_capability(&lazy_pool_route)
            .is_err(),
        "a lazy transpose is not an admitted pooling stage input"
    );
    let mut materialized_pool = lazy_pool_route.clone();
    materialized_pool.steps.insert(
        0,
        DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::View(ProviderViewStep::Reshape(vec![3, 2])),
            parameters: DeepLearningForwardParameters::None,
        },
    );
    let Some(DeepLearningForwardCapability::Ready(materialized_pool_resources)) = provider
        .deep_learning_forward_route_capability(&materialized_pool)
        .unwrap()
    else {
        panic!("explicit view materialization makes pooling admissible")
    };
    assert_eq!(
        materialized_pool_resources.new_buffer_bytes_upper_bound,
        DeepLearningForwardBufferBound::Known(
            9 * class.element_size() as u64
                + std::mem::size_of::<crate::backend::wgpu::params::PermuteParams>() as u64
                + std::mem::size_of::<[u32; 4]>() as u64
        ),
        "the view materialization and pool output/uniform are distinct owned buffers"
    );
    let empty_pool = DeepLearningForwardRoute {
        input: DeepLearningForwardSource::Host {
            element_type: class,
            shape: vec![0, 3],
        },
        parameters: Vec::new(),
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::Pool1D {
                input_shape: vec![0, 3],
                route: runmat_accelerate_api::DeepLearningPool1DRoute {
                    reduction: runmat_accelerate_api::DeepLearningPool1DRequest {
                        axis_zero_based: 1,
                    },
                    result_axis: runmat_accelerate_api::DeepLearningPool1DResultAxis::KeepSingleton,
                    source_shape: vec![0, 1],
                    target_shape: vec![0, 1],
                    target_to_source: vec![0, 1],
                },
            },
            parameters: DeepLearningForwardParameters::None,
        }],
        output_view: GpuTensorView::contiguous(vec![0, 1]),
    };
    let Some(DeepLearningForwardCapability::Ready(empty_pool_resources)) = provider
        .deep_learning_forward_route_capability(&empty_pool)
        .unwrap()
    else {
        panic!("empty non-reduced axis must remain admissible")
    };
    assert_eq!(
        empty_pool_resources.new_buffer_bytes_upper_bound,
        DeepLearningForwardBufferBound::Known(2 * class.element_size() as u64),
        "an empty upload and an empty pool output each allocate one physical lane"
    );
    let physical_permute = DeepLearningForwardRoute {
        input: DeepLearningForwardSource::Host {
            element_type: class,
            shape: vec![2, 3],
        },
        parameters: Vec::new(),
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::View(ProviderViewStep::Permute2D),
            parameters: DeepLearningForwardParameters::None,
        }],
        output_view: GpuTensorView::contiguous(vec![3, 2]),
    };
    let Some(DeepLearningForwardCapability::Ready(permutation_resources)) = provider
        .deep_learning_forward_route_capability(&physical_permute)
        .unwrap()
    else {
        panic!("physical permutation should remain ready")
    };
    assert_eq!(
        permutation_resources.new_buffer_bytes_upper_bound,
        DeepLearningForwardBufferBound::Known(
            12 * class.element_size() as u64
                + std::mem::size_of::<crate::backend::wgpu::params::PermuteParams>() as u64
        ),
    );
    assert_eq!(provider.telemetry_snapshot(), before);
}
