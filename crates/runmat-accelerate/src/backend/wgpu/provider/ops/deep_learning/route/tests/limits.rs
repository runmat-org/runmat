use super::*;

#[test]
#[ignore = "requires actual WGPU device; route-capability queries allocate no large buffers"]
fn chunked_dense_route_sums_every_native_buffer_without_execution() {
    let fixture = Fixture::new();
    let provider = &fixture.provider;
    let class = fixture.class;
    let before = provider.telemetry_snapshot();
    let channels = 65_536_usize;
    let route = DeepLearningForwardRoute {
        input: DeepLearningForwardSource::Host {
            element_type: class,
            shape: vec![2, channels],
        },
        parameters: vec![
            DeepLearningForwardParameter {
                source: DeepLearningForwardSource::Host {
                    element_type: class,
                    shape: vec![channels, 3],
                },
                view_steps: Vec::new(),
            },
            DeepLearningForwardParameter {
                source: DeepLearningForwardSource::Host {
                    element_type: class,
                    shape: vec![1, 3],
                },
                view_steps: Vec::new(),
            },
        ],
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::FeedForward(
                DeepLearningFeedForwardStage::Dense {
                    batch: 2,
                    input_channels: channels,
                    output_channels: 3,
                },
            ),
            parameters: DeepLearningForwardParameters::Dense {
                weights: 0,
                bias: 1,
            },
        }],
        output_view: GpuTensorView::contiguous(vec![2, 3]),
    };
    let Some(DeepLearningForwardCapability::Ready(resources)) = provider
        .deep_learning_forward_route_capability(&route)
        .unwrap()
    else {
        panic!("admitted large-K dense geometry should not allocate or execute")
    };
    let element = class.element_size() as u64;
    let product = 6 * element;
    let upload = (5 * channels as u64 + 3) * element;
    // Eight 8192-wide K chunks use two reusable product buffers, seven
    // accumulation outputs, and immutable parameter buffers per dispatch.
    // Broadcast parameters include all supported axes and exceed one page.
    let chunked_matmul = 2 * product + 8 * 256 + 7 * (product + 256);
    let broadcast_uniform =
        (std::mem::size_of::<crate::backend::wgpu::params::BinaryBroadcastParams>() as u64)
            .div_ceil(256)
            * 256;
    let bias_add = product + broadcast_uniform;
    assert_eq!(resources.upload_bytes, upload);
    assert_eq!(
        resources.new_buffer_bytes_upper_bound,
        DeepLearningForwardBufferBound::Known(upload + chunked_matmul + bias_add)
    );
    assert_eq!(provider.telemetry_snapshot(), before);
}

#[test]
#[ignore = "requires actual WGPU device; buffer-demand queries allocate no large buffers"]
fn pool_chunk_boundary_empty_and_overflow_demands_are_checked() {
    let fixture = Fixture::new();
    let provider = &fixture.provider;
    let element = fixture.class.element_size() as u64;
    let before = provider.telemetry_snapshot();
    let capacity = u64::from(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS)
        * u64::from(crate::backend::wgpu::shaders::deep_learning::POOL_WORKGROUP_SIZE);
    let outputs = usize::try_from(capacity + 1).unwrap();
    let route = runmat_accelerate_api::DeepLearningPool1DRoute {
        reduction: runmat_accelerate_api::DeepLearningPool1DRequest { axis_zero_based: 0 },
        result_axis: runmat_accelerate_api::DeepLearningPool1DResultAxis::KeepSingleton,
        source_shape: vec![1, outputs],
        target_shape: vec![1, outputs],
        target_to_source: vec![0, 1],
    };
    assert_eq!(
        provider.pool_1d_buffer_demand(&GpuTensorView::contiguous(vec![2, outputs]), &route),
        Some((capacity + 1) * element + 2 * std::mem::size_of::<[u32; 4]>() as u64),
        "the second output chunk has its own immutable uniform"
    );
    assert_eq!(
        provider.pool_1d_buffer_demand(&GpuTensorView::contiguous(vec![2, 0]), &route),
        Some(element),
        "an empty output still owns one physical storage lane and no uniforms"
    );
    assert_eq!(
        provider.pool_1d_buffer_demand(&GpuTensorView::contiguous(vec![2, usize::MAX]), &route,),
        None,
        "checked output bytes must not wrap"
    );
    assert_eq!(provider.matmul_buffer_demand(u64::MAX, 2, 65_536), None);
    let overflowing_route = DeepLearningForwardRoute {
        input: DeepLearningForwardSource::Host {
            element_type: fixture.class,
            shape: vec![usize::MAX, 2],
        },
        parameters: Vec::new(),
        steps: vec![DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::FeedForward(
                DeepLearningFeedForwardStage::Relu { elements: 1 },
            ),
            parameters: DeepLearningForwardParameters::None,
        }],
        output_view: GpuTensorView::contiguous(vec![usize::MAX, 2]),
    };
    assert_eq!(
        provider
            .deep_learning_forward_route_capability(&overflowing_route)
            .unwrap(),
        Some(DeepLearningForwardCapability::ResourceLimit),
        "physical-route overflow must be rejected before effects"
    );
    assert_eq!(provider.telemetry_snapshot(), before);
}
