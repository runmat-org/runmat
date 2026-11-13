use super::*;
#[test]
#[ignore = "requires actual WGPU device; run explicitly for effect-free forward-route admission"]
fn admitted_resource_bounds_are_effect_free() {
    let fixture = Fixture::new();
    let provider = &fixture.provider;
    let class = fixture.class;
    let small_input = fixture.small_input();
    let shared_parameter = fixture.shared_parameter();
    let before = provider.telemetry_snapshot();
    let normalization = DeepLearningForwardRoute {
        input: DeepLearningForwardSource::Host {
            element_type: class,
            shape: vec![2, 3],
        },
        parameters: (0..2)
            .map(|_| DeepLearningForwardParameter {
                source: DeepLearningForwardSource::Host {
                    element_type: class,
                    shape: vec![2, 1],
                },
                view_steps: Vec::new(),
            })
            .collect(),
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::LayerNorm(
                runmat_accelerate_api::DeepLearningLayerNormRequest {
                    shape: vec![2, 3],
                    reduce_mask: 1,
                    channel_axis: 0,
                    scale_shape: vec![2, 1],
                    offset_shape: vec![2, 1],
                    epsilon: 1.0e-5,
                },
            ),
            parameters: DeepLearningForwardParameters::LayerNorm {
                scale: 0,
                offset: 1,
            },
        }],
        output_view: GpuTensorView::contiguous(vec![2, 3]),
    };
    let Some(DeepLearningForwardCapability::Ready(norm_resources)) = provider
        .deep_learning_forward_route_capability(&normalization)
        .unwrap()
    else {
        panic!("host-origin normalization should remain admitted")
    };
    assert_eq!(
        norm_resources.new_buffer_bytes_upper_bound,
        DeepLearningForwardBufferBound::Known(25 * class.element_size() as u64 + 112),
        "uploads, output, three statistics per group, and two 56-byte uniforms"
    );
    let shared = DeepLearningForwardRoute {
        input: DeepLearningForwardSource::Resident(small_input),
        parameters: vec![
            DeepLearningForwardParameter {
                source: DeepLearningForwardSource::Resident(shared_parameter),
                view_steps: vec![ProviderViewStep::Transpose2D],
            },
            DeepLearningForwardParameter {
                source: DeepLearningForwardSource::Resident(shared_parameter),
                view_steps: vec![ProviderViewStep::Reshape(vec![1, 1])],
            },
        ],
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::FeedForward(
                DeepLearningFeedForwardStage::Dense {
                    batch: 2,
                    input_channels: 1,
                    output_channels: 1,
                },
            ),
            parameters: DeepLearningForwardParameters::Dense {
                weights: 0,
                bias: 1,
            },
        }],
        output_view: GpuTensorView::contiguous(vec![2, 1]),
    };
    let Some(DeepLearningForwardCapability::Ready(shared_resources)) = provider
        .deep_learning_forward_route_capability(&shared)
        .unwrap()
    else {
        panic!("same allocation may supply two admitted parameter views")
    };
    assert_eq!(
        shared_resources.resident_bytes,
        small_input.physical().requested_bytes() + shared_parameter.physical().requested_bytes()
    );
    assert_eq!(
        shared_resources.new_buffer_bytes_upper_bound,
        DeepLearningForwardBufferBound::Known(
            provider.matmul_buffer_demand(2, 1, 1).unwrap()
                + provider.binary_broadcast_buffer_demand(2).unwrap()
        ),
        "dense route accounts for product and bias output buffers"
    );
    let host = route(DeepLearningForwardSource::Host {
        element_type: class,
        shape: vec![2, 3],
    });
    let mut unsupported = host.clone();
    unsupported.input = DeepLearningForwardSource::Host {
        element_type: NumericElementType::I32,
        shape: vec![2, 3],
    };
    assert_eq!(
        provider
            .deep_learning_forward_route_capability(&unsupported)
            .unwrap(),
        Some(DeepLearningForwardCapability::UnsupportedOperation)
    );
    let mut malformed = host;
    malformed.steps[0].parameters = DeepLearningForwardParameters::Dense {
        weights: 0,
        bias: 1,
    };
    assert!(provider
        .deep_learning_forward_route_capability(&malformed)
        .is_err());
    assert_eq!(provider.telemetry_snapshot(), before);
}
