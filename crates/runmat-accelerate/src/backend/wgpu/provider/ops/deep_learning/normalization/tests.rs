use super::*;

#[test]
fn chunks_cover_each_output_once() {
    for total in [0, 1, 63, 64, 65, 4097] {
        let mut seen = vec![0; total];
        let mut offset = 0;
        while let Some(chunk) = next_chunk(total as u32, offset) {
            for local in 0..chunk.workgroups * 64 {
                if local < chunk.count {
                    seen[(chunk.offset + local) as usize] += 1;
                }
            }
            offset += chunk.count;
        }
        assert!(seen.iter().all(|&visits| visits == 1));
    }
}

#[test]
fn typed_geometry_rejects_malformed_reduction_and_parameter_shapes() {
    let mut request = DeepLearningLayerNormRequest {
        shape: vec![2, 3, 2],
        reduce_mask: 0b010,
        channel_axis: 1,
        scale_shape: vec![3, 1],
        offset_shape: vec![3, 1],
        epsilon: 1e-5,
    };
    assert_eq!(geometry(&request).unwrap(), (4, 3));
    request.reduce_mask = 0b001;
    assert!(geometry(&request).is_err());
    request.reduce_mask = 0b010;
    request.offset_shape = vec![1, 2];
    assert!(geometry(&request).is_err());
    request.offset_shape = vec![3, 1];
    request.epsilon = f64::NAN;
    assert!(geometry(&request).is_err());
}

#[test]
#[ignore = "requires actual WGPU device; run explicitly for pure future-stage admission"]
fn planned_normalization_admits_future_geometry_without_provider_effects() {
    use runmat_accelerate_api::{AccelProvider, HostTensorView};

    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let anchor = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[2, 1],
        })
        .unwrap();
    let request = DeepLearningLayerNormRequest {
        shape: vec![3, 4],
        reduce_mask: 0b01,
        channel_axis: 0,
        scale_shape: vec![3, 1],
        offset_shape: vec![3, 1],
        epsilon: 1e-5,
    };
    provider.reset_telemetry();
    let before = provider.telemetry_snapshot();
    assert_eq!(
        provider
            .deep_learning_layer_norm_planned_capability_exec(&anchor, None, None, &request)
            .unwrap(),
        DeepLearningLayerNormCapability::Ready
    );
    assert!(provider
        .deep_learning_layer_norm_capability_exec(&anchor, None, None, &request)
        .is_err());
    let telemetry = provider.telemetry_snapshot();
    assert_eq!(
        telemetry, before,
        "capability queries must have no provider effects"
    );
    anchor.release().unwrap();
}

#[test]
#[ignore = "requires actual WGPU device; run explicitly for native precision qualification"]
fn resident_channel_normalization_matches_independent_cb_oracle() {
    use runmat_accelerate_api::{AccelProvider, HostTensorView};

    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: &[3, 2],
        })
        .unwrap();
    let scale = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0],
            shape: &[3, 1],
        })
        .unwrap();
    let offset = provider
        .upload(&HostTensorView {
            data: &[0.5, -0.5, 1.0],
            shape: &[3, 1],
        })
        .unwrap();
    let request = DeepLearningLayerNormRequest {
        shape: vec![3, 2],
        reduce_mask: 0b01,
        channel_axis: 0,
        scale_shape: vec![3, 1],
        offset_shape: vec![3, 1],
        epsilon: 1e-5,
    };
    assert_eq!(
        provider
            .deep_learning_layer_norm_capability_exec(&input, Some(&scale), Some(&offset), &request)
            .unwrap(),
        DeepLearningLayerNormCapability::Ready
    );
    provider.reset_telemetry();
    let output = provider
        .deep_learning_layer_norm_exec(&input, &scale, &offset, &request)
        .unwrap();
    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(output.device_id(), input.device_id());
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        runmat_accelerate_api::handle_precision(&input)
    );
    assert!(!output.shares_allocation_with(&input));
    assert!(!output.shares_allocation_with(&scale));
    assert!(!output.shares_allocation_with(&offset));
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    let values = pollster::block_on(provider.download(&output)).unwrap().data;
    assert_eq!(values.len(), 6);
    let std = (2.0_f64 / 3.0 + 1e-5).sqrt();
    let expected = [
        -1.0 / std + 0.5,
        -0.5,
        3.0 / std + 1.0,
        -1.0 / std + 0.5,
        -0.5,
        3.0 / std + 1.0,
    ];
    let tolerance = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => 2e-5,
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-11,
    };
    for (actual, expected) in values.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < tolerance,
            "actual {actual}, expected {expected}"
        );
    }
    assert_eq!(
        pollster::block_on(provider.download(&input)).unwrap().data,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        pollster::block_on(provider.download(&scale)).unwrap().data,
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        pollster::block_on(provider.download(&offset)).unwrap().data,
        vec![0.5, -0.5, 1.0]
    );
    for handle in [input, scale, offset, output] {
        handle.release().unwrap();
    }
}

#[test]
#[ignore = "requires actual WGPU device; run explicitly for native precision qualification"]
fn resident_nonsquare_scbt_operation_dimensions_match_independent_groups() {
    use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};

    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let values = (0..24)
        .map(|index| ((index * 7) % 17) as f64 - 8.0)
        .collect::<Vec<_>>();
    let input = provider
        .upload(&HostTensorView {
            data: &values,
            shape: &[2, 3, 2, 2],
        })
        .unwrap();
    let scale = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0],
            shape: &[1, 3],
        })
        .unwrap();
    let offset = provider
        .upload(&HostTensorView {
            data: &[0.5, -0.5, 1.0],
            shape: &[1, 3],
        })
        .unwrap();
    let tolerance = match provider.precision() {
        ProviderPrecision::F32 => 2e-4,
        ProviderPrecision::F64 => 1e-11,
    };
    for (mask, groups) in [(0b0010, 8), (0b0011, 4), (0b1011, 2)] {
        let request = DeepLearningLayerNormRequest {
            shape: vec![2, 3, 2, 2],
            reduce_mask: mask,
            channel_axis: 1,
            scale_shape: vec![1, 3],
            offset_shape: vec![1, 3],
            epsilon: 1e-5,
        };
        assert_eq!(geometry(&request).unwrap().0, groups);
        assert_eq!(
            provider
                .deep_learning_layer_norm_capability_exec(
                    &input,
                    Some(&scale),
                    Some(&offset),
                    &request
                )
                .unwrap(),
            DeepLearningLayerNormCapability::Ready
        );
        provider.reset_telemetry();
        let output = provider
            .deep_learning_layer_norm_exec(&input, &scale, &offset, &request)
            .unwrap();
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(output.shape(), &[2, 3, 2, 2]);
        assert!(!output.shares_allocation_with(&input));
        assert!(!output.shares_allocation_with(&scale));
        assert!(!output.shares_allocation_with(&offset));
        let actual = pollster::block_on(provider.download(&output)).unwrap().data;
        assert_eq!(actual.len(), 24);
        let mut buckets = vec![Vec::new(); groups];
        for index in 0..24 {
            let spatial = index % 2;
            let batch = (index / 6) % 2;
            let time = index / 12;
            let group = match mask {
                0b0010 => spatial + 2 * (batch + 2 * time),
                0b0011 => batch + 2 * time,
                0b1011 => batch,
                _ => unreachable!(),
            };
            buckets[group].push(index);
        }
        let mut expected = vec![0.0; 24];
        for bucket in buckets {
            let count = bucket.len() as f64;
            let mean = bucket.iter().map(|&index| values[index]).sum::<f64>() / count;
            let variance = bucket
                .iter()
                .map(|&index| (values[index] - mean).powi(2))
                .sum::<f64>()
                / count;
            let denominator = (variance + 1e-5).sqrt();
            for index in bucket {
                let channel = (index / 2) % 3;
                expected[index] = (values[index] - mean) / denominator * [1.0, 2.0, 3.0][channel]
                    + [0.5, -0.5, 1.0][channel];
            }
        }
        for (index, (&left, &right)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (left - right).abs() < tolerance,
                "mask {mask:#b}, index {index}: actual {left}, expected {right}"
            );
        }
        output.release().unwrap();
    }
    for handle in [input, scale, offset] {
        handle.release().unwrap();
    }
}

#[test]
#[ignore = "requires actual WGPU device; run explicitly for native precision qualification"]
fn resident_large_cancellation_and_nonfinite_groups_follow_stable_contract() {
    use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};

    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
    let large = match provider.precision() {
        ProviderPrecision::F32 => 1e38,
        ProviderPrecision::F64 => 1e308,
    };
    let values = [large, -large, 1e-10, f64::INFINITY, 1.0, 2.0];
    let input = provider
        .upload(&HostTensorView {
            data: &values,
            shape: &[3, 2],
        })
        .unwrap();
    let scale = provider
        .upload(&HostTensorView {
            data: &[1.0, 1.0, 1.0],
            shape: &[3, 1],
        })
        .unwrap();
    let offset = provider
        .upload(&HostTensorView {
            data: &[0.0, 0.0, 0.0],
            shape: &[3, 1],
        })
        .unwrap();
    let request = DeepLearningLayerNormRequest {
        shape: vec![3, 2],
        reduce_mask: 0b01,
        channel_axis: 0,
        scale_shape: vec![3, 1],
        offset_shape: vec![3, 1],
        epsilon: 1e-5,
    };
    let output = provider
        .deep_learning_layer_norm_exec(&input, &scale, &offset, &request)
        .unwrap();
    let actual = pollster::block_on(provider.download(&output)).unwrap().data;
    assert_eq!(actual.len(), 6);
    let tolerance = if provider.precision() == ProviderPrecision::F32 {
        3e-5
    } else {
        1e-12
    };
    let expected = (1.5_f64).sqrt();
    assert!(
        (actual[0] - expected).abs() < tolerance,
        "positive large output {}",
        actual[0]
    );
    assert!(
        (actual[1] + expected).abs() < tolerance,
        "negative large output {}",
        actual[1]
    );
    assert!(
        actual[2].abs() < tolerance,
        "cancellation residual output {}",
        actual[2]
    );
    assert!(actual[3..].iter().all(|value| value.is_nan()));
    for handle in [input, scale, offset, output] {
        handle.release().unwrap();
    }
}
