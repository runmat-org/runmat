use super::*;
#[test]
#[ignore = "requires actual WGPU device; run explicitly for effect-free forward-route admission"]
fn admitted_resource_bounds_are_effect_free() {
    let fixture = Fixture::new();
    let provider = &fixture.provider;
    let class = fixture.class;
    let resident = fixture.resident();
    let before = provider.telemetry_snapshot();
    let host = route(DeepLearningForwardSource::Host {
        element_type: class,
        shape: vec![2, 3],
    });
    let resident_route = route(DeepLearningForwardSource::Resident(resident));
    for (candidate, expected_upload) in [
        (&host, 6 * class.element_size() as u64),
        (&resident_route, 0),
    ] {
        let Some(DeepLearningForwardCapability::Ready(resources)) = provider
            .deep_learning_forward_route_capability(candidate)
            .unwrap()
        else {
            panic!("legal physical route should be ready");
        };
        assert_eq!(resources.output_bytes, 6 * class.element_size() as u64);
        assert_eq!(
            resources.new_buffer_bytes_upper_bound,
            DeepLearningForwardBufferBound::Known(
                expected_upload + 6 * class.element_size() as u64 + 256
            )
        );
    }
    let mut two_relu = host.clone();
    two_relu.steps.push(two_relu.steps[0].clone());
    let Some(DeepLearningForwardCapability::Ready(two_relu_resources)) = provider
        .deep_learning_forward_route_capability(&two_relu)
        .unwrap()
    else {
        panic!("two admitted ReLU stages should remain ready")
    };
    assert_eq!(
        two_relu_resources.new_buffer_bytes_upper_bound,
        DeepLearningForwardBufferBound::Known(3 * 6 * class.element_size() as u64 + 512),
        "pending earlier stages cannot be omitted from the conservative sum"
    );
    let mut softmax = host.clone();
    softmax.steps[0].operation =
        DeepLearningForwardOperation::FeedForward(DeepLearningFeedForwardStage::SoftmaxRows {
            rows: 2,
            columns: 3,
        });
    let Some(DeepLearningForwardCapability::Ready(softmax_resources)) = provider
        .deep_learning_forward_route_capability(&softmax)
        .unwrap()
    else {
        panic!("admitted softmax should remain ready")
    };
    assert_eq!(
        softmax_resources.new_buffer_bytes_upper_bound,
        DeepLearningForwardBufferBound::Known(2 * 6 * class.element_size() as u64 + 16),
    );
    let mut elu = host.clone();
    elu.steps[0].operation =
        DeepLearningForwardOperation::FeedForward(DeepLearningFeedForwardStage::Elu {
            elements: 6,
        });
    let Some(DeepLearningForwardCapability::Ready(elu_resources)) = provider
        .deep_learning_forward_route_capability(&elu)
        .unwrap()
    else {
        panic!("admitted ELU should remain ready")
    };
    let fill_uniform = match class {
        NumericElementType::F32 => {
            std::mem::size_of::<crate::backend::wgpu::params::FillParamsF32>() as u64
        }
        NumericElementType::F64 => {
            std::mem::size_of::<crate::backend::wgpu::params::FillParamsF64>() as u64
        }
        _ => unreachable!("provider precision is floating"),
    };
    assert_eq!(
        elu_resources.new_buffer_bytes_upper_bound,
        DeepLearningForwardBufferBound::Known(
            3 * 6 * class.element_size() as u64 + fill_uniform + 16
        ),
    );
    assert_eq!(provider.telemetry_snapshot(), before);
}
