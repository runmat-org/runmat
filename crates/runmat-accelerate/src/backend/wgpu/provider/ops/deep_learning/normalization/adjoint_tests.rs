use super::*;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    ProviderPrecision,
};

fn upload(provider: &WgpuProvider, values: &[f64], shape: &[usize]) -> GpuTensorHandle {
    match provider.precision() {
        ProviderPrecision::F32 => {
            let native = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&native),
                    shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
        }
        ProviderPrecision::F64 => provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(values),
                shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap(),
    }
}

fn download(provider: &WgpuProvider, value: &GpuTensorHandle) -> Vec<f64> {
    match pollster::block_on(provider.download_numeric(value))
        .unwrap()
        .data
    {
        HostNumericDataOwned::F32(values) => values.into_iter().map(f64::from).collect(),
        HostNumericDataOwned::F64(values) => values,
        other => panic!("expected a native real floating result, got {other:?}"),
    }
}

fn independent_loss(x: &[f64; 12], gamma: &[f64; 3], beta: &[f64; 3], dy: &[f64; 12]) -> f64 {
    let mut loss = 0.0;
    for group in 0..2 {
        let start = 6 * group;
        let mean = x[start..start + 6].iter().sum::<f64>() / 6.0;
        let variance = x[start..start + 6]
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / 6.0;
        let inv_std = (variance + 1e-5).sqrt().recip();
        for local in 0..6 {
            let index = start + local;
            let channel = local / 2;
            loss += dy[index] * ((x[index] - mean) * inv_std * gamma[channel] + beta[channel]);
        }
    }
    loss
}

fn gpu_forward_loss(
    provider: &WgpuProvider,
    x: &[f64; 12],
    gamma: &[f64; 3],
    beta: &[f64; 3],
    dy: &[f64; 12],
    request: &DeepLearningLayerNormRequest,
) -> f64 {
    let input = upload(provider, x, &[2, 3, 2]);
    let scale = upload(provider, gamma, &[1, 3]);
    let offset = upload(provider, beta, &[1, 3]);
    let output = provider
        .deep_learning_layer_norm_exec(&input, &scale, &offset, request)
        .unwrap();
    let values = download(provider, &output);
    assert_eq!(values.len(), dy.len());
    values
        .iter()
        .zip(dy)
        .map(|(value, weight)| value * weight)
        .sum()
}

fn recorded_nonsquare_normalization_adjoint_matches_independent_oracle(
    precision: ProviderPrecision,
) {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    assert_eq!(
        provider.precision(),
        precision,
        "set RUNMAT_WGPU_FORCE_PRECISION to the test precision; F64 requires SHADER_F64"
    );
    let shape = [2, 3, 2];
    let x = [1., 2., 4., 5., 7., 9., -2., 3., -1., 6., 8., 10.];
    let gamma = [0.7, -1.2, 2.0];
    let beta = [0.25, -0.5, 1.0];
    let dy = [
        0.2, -0.3, 0.4, 0.1, -0.5, 0.7, -0.2, 0.5, 0.3, -0.4, 0.6, 0.8,
    ];
    let input = upload(&provider, &x, &shape);
    let scale = upload(&provider, &gamma, &[1, 3]);
    let offset = upload(&provider, &beta, &[1, 3]);
    let upstream = upload(&provider, &dy, &shape);
    let request = DeepLearningLayerNormRequest {
        shape: shape.to_vec(),
        reduce_mask: 0b011,
        channel_axis: 1,
        scale_shape: vec![1, 3],
        offset_shape: vec![1, 3],
        epsilon: 1e-5,
    };
    assert_eq!(
        provider
            .deep_learning_layer_norm_adjoint_capability_exec(
                &input,
                Some(&scale),
                Some(&offset),
                None,
                None,
                &request,
            )
            .unwrap(),
        DeepLearningLayerNormCapability::Ready
    );
    let wrong_statistics = upload(&provider, &[1.0; 6], &[2, 3]);
    let before_invalid = provider.telemetry_snapshot();
    assert!(provider
        .deep_learning_layer_norm_adjoint_capability_exec(
            &input,
            Some(&scale),
            Some(&offset),
            Some(&wrong_statistics),
            Some(&upstream),
            &request,
        )
        .is_err());
    assert_eq!(provider.telemetry_snapshot(), before_invalid);
    provider.reset_telemetry();
    let recorded = provider
        .deep_learning_layer_norm_recorded_exec(&input, &scale, &offset, &request)
        .unwrap();
    assert_eq!(recorded.statistics.shape(), &[3, 2]);
    assert!(!recorded.statistics.shares_allocation_with(&recorded.output));
    let gradients = provider
        .deep_learning_layer_norm_adjoint_exec(
            &input,
            &scale,
            &offset,
            &recorded.statistics,
            &upstream,
            &request,
        )
        .unwrap();
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(gradients.input.shape(), shape);
    assert_eq!(gradients.scale.shape(), &[1, 3]);
    assert_eq!(gradients.offset.shape(), &[1, 3]);
    for value in [&gradients.input, &gradients.scale, &gradients.offset] {
        assert_eq!(value.device_id(), input.device_id());
        assert_eq!(
            value.physical().element_type(),
            input.physical().element_type()
        );
        assert!(!value.shares_allocation_with(&input));
        assert!(!value.shares_allocation_with(&upstream));
    }

    // Independent two-group column-major reference; no production geometry
    // helper or GPU readback participates in this expected-value calculation.
    let mut expected_dx = [0.0; 12];
    let mut expected_gamma = [0.0; 3];
    let mut expected_beta = [0.0; 3];
    for group in 0..2 {
        let start = 6 * group;
        let mean = x[start..start + 6].iter().sum::<f64>() / 6.0;
        let variance = x[start..start + 6]
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / 6.0;
        let inv_std = (variance + request.epsilon).sqrt().recip();
        let mut sum_dxhat = 0.0;
        let mut sum_dxhat_xhat = 0.0;
        for local in 0..6 {
            let index = start + local;
            let channel = local / 2;
            let xhat = (x[index] - mean) * inv_std;
            let dxhat = dy[index] * gamma[channel];
            expected_gamma[channel] += dy[index] * xhat;
            expected_beta[channel] += dy[index];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * xhat;
        }
        for local in 0..6 {
            let index = start + local;
            let channel = local / 2;
            let xhat = (x[index] - mean) * inv_std;
            let dxhat = dy[index] * gamma[channel];
            expected_dx[index] = inv_std / 6.0 * (6.0 * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
        }
    }
    let tolerance = match provider.precision() {
        ProviderPrecision::F32 => 2e-4,
        ProviderPrecision::F64 => 2e-11,
    };
    let actual_dx = download(&provider, &gradients.input);
    let actual_gamma = download(&provider, &gradients.scale);
    let actual_beta = download(&provider, &gradients.offset);
    assert_eq!(actual_dx.len(), expected_dx.len());
    assert_eq!(actual_gamma.len(), expected_gamma.len());
    assert_eq!(actual_beta.len(), expected_beta.len());
    for (actual, expected) in actual_dx.iter().zip(expected_dx) {
        assert!(
            (actual - expected).abs() < tolerance,
            "dX {actual} != {expected}"
        );
    }
    for (actual, expected) in actual_gamma.iter().zip(expected_gamma) {
        assert!(
            (actual - expected).abs() < tolerance,
            "dScale {actual} != {expected}"
        );
    }
    for (actual, expected) in actual_beta.iter().zip(expected_beta) {
        assert!(
            (actual - expected).abs() < tolerance,
            "dOffset {actual} != {expected}"
        );
    }
    // This finite difference independently checks the analytical CPU
    // expected values. Device adjoints were compared with those values above;
    // this is not a second GPU finite-difference execution.
    let step = 1e-5;
    let mut plus_x = x;
    plus_x[0] += step;
    let mut minus_x = x;
    minus_x[0] -= step;
    let fd_input = (independent_loss(&plus_x, &gamma, &beta, &dy)
        - independent_loss(&minus_x, &gamma, &beta, &dy))
        / (2.0 * step);
    assert!((expected_dx[0] - fd_input).abs() < 1e-8);
    let mut plus_gamma = gamma;
    plus_gamma[1] += step;
    let mut minus_gamma = gamma;
    minus_gamma[1] -= step;
    let fd_scale = (independent_loss(&x, &plus_gamma, &beta, &dy)
        - independent_loss(&x, &minus_gamma, &beta, &dy))
        / (2.0 * step);
    assert!((expected_gamma[1] - fd_scale).abs() < 1e-8);
    let mut plus_beta = beta;
    plus_beta[2] += step;
    let mut minus_beta = beta;
    minus_beta[2] -= step;
    let fd_offset = (independent_loss(&x, &gamma, &plus_beta, &dy)
        - independent_loss(&x, &gamma, &minus_beta, &dy))
        / (2.0 * step);
    assert!((expected_beta[2] - fd_offset).abs() < 1e-8);

    // Also perturb the *executed provider forward*, so the test catches a
    // derivative inconsistent with its actual native-precision output.
    let gpu_step = match provider.precision() {
        ProviderPrecision::F32 => 1e-2,
        ProviderPrecision::F64 => 1e-5,
    };
    let mut gpu_plus_x = x;
    gpu_plus_x[0] += gpu_step;
    let mut gpu_minus_x = x;
    gpu_minus_x[0] -= gpu_step;
    let gpu_fd_x = (gpu_forward_loss(&provider, &gpu_plus_x, &gamma, &beta, &dy, &request)
        - gpu_forward_loss(&provider, &gpu_minus_x, &gamma, &beta, &dy, &request))
        / (2.0 * gpu_step);
    let gpu_fd_tolerance = match provider.precision() {
        ProviderPrecision::F32 => 3e-3,
        ProviderPrecision::F64 => 1e-8,
    };
    assert!((actual_dx[0] - gpu_fd_x).abs() < gpu_fd_tolerance);

    // The saved execution leases, not the caller's original handle values,
    // keep the exact forward input/parameters/statistics available to reverse.
    let saved_input = input.clone();
    let saved_scale = scale.clone();
    let saved_offset = offset.clone();
    let saved_statistics = recorded.statistics.clone();
    let saved_upstream = upstream.clone();
    input.release().unwrap();
    scale.release().unwrap();
    offset.release().unwrap();
    upstream.release().unwrap();
    recorded.output.release().unwrap();
    recorded.statistics.release().unwrap();
    let retained = provider
        .deep_learning_layer_norm_adjoint_exec(
            &saved_input,
            &saved_scale,
            &saved_offset,
            &saved_statistics,
            &saved_upstream,
            &request,
        )
        .unwrap();
    let retained_dx = download(&provider, &retained.input);
    assert_eq!(retained_dx.len(), actual_dx.len());
    for (actual, previous) in retained_dx.iter().zip(actual_dx) {
        assert!((actual - previous).abs() < tolerance);
    }
}

#[test]
#[ignore = "requires actual WGPU; RUNMAT_WGPU_FORCE_PRECISION=f32"]
fn recorded_nonsquare_normalization_adjoint_f32() {
    recorded_nonsquare_normalization_adjoint_matches_independent_oracle(ProviderPrecision::F32);
}

#[test]
#[ignore = "requires SHADER_F64 WGPU; RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn recorded_nonsquare_normalization_adjoint_f64() {
    recorded_nonsquare_normalization_adjoint_matches_independent_oracle(ProviderPrecision::F64);
}

#[test]
#[ignore = "requires an actual WGPU adapter; run explicitly as a required-device gate"]
fn empty_batch_retains_parameter_gradient_shape_and_zero_values() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    let input = upload(&provider, &[], &[2, 3, 0]);
    let scale = upload(&provider, &[1.0, 2.0, 3.0], &[1, 3]);
    let offset = upload(&provider, &[0.0, 0.0, 0.0], &[1, 3]);
    let upstream = upload(&provider, &[], &[2, 3, 0]);
    let request = DeepLearningLayerNormRequest {
        shape: vec![2, 3, 0],
        reduce_mask: 0b011,
        channel_axis: 1,
        scale_shape: vec![1, 3],
        offset_shape: vec![1, 3],
        epsilon: 1e-5,
    };
    let recorded = provider
        .deep_learning_layer_norm_recorded_exec(&input, &scale, &offset, &request)
        .unwrap();
    assert_eq!(recorded.output.shape(), &[2, 3, 0]);
    assert_eq!(recorded.statistics.shape(), &[3, 0]);
    let gradients = provider
        .deep_learning_layer_norm_adjoint_exec(
            &input,
            &scale,
            &offset,
            &recorded.statistics,
            &upstream,
            &request,
        )
        .unwrap();
    assert_eq!(gradients.input.shape(), &[2, 3, 0]);
    assert_eq!(download(&provider, &gradients.input), Vec::<f64>::new());
    assert_eq!(download(&provider, &gradients.scale), vec![0.0; 3]);
    assert_eq!(download(&provider, &gradients.offset), vec![0.0; 3]);
}

fn extreme_normalization_adjoint_keeps_representable_gradients(precision: ProviderPrecision) {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    assert_eq!(
        provider.precision(),
        precision,
        "set RUNMAT_WGPU_FORCE_PRECISION to the test precision; F64 requires SHADER_F64"
    );
    let magnitudes = match precision {
        ProviderPrecision::F32 => [1e38, f64::from(f32::MAX)],
        ProviderPrecision::F64 => [1e300, f64::MAX],
    };
    for magnitude in magnitudes {
        let seed_scale = match precision {
            ProviderPrecision::F32 => 1e20,
            ProviderPrecision::F64 => 1e200,
        };
        let dy = [1.0, 0.5, -1.0 / 3.0, 0.25, -0.2, 1.0 / 6.0].map(|v| v * seed_scale);
        let gamma = [1.0; 3];
        let beta = [0.0; 3];
        let request = DeepLearningLayerNormRequest {
            shape: vec![2, 3, 1],
            reduce_mask: 0b011,
            channel_axis: 1,
            scale_shape: vec![1, 3],
            offset_shape: vec![1, 3],
            epsilon: 1e-5,
        };
        let scale = upload(&provider, &gamma, &[1, 3]);
        let offset = upload(&provider, &beta, &[1, 3]);
        let upstream = upload(&provider, &dy, &[2, 3, 1]);
        let alternating = [
            magnitude, -magnitude, magnitude, -magnitude, magnitude, -magnitude,
        ];
        let input = upload(&provider, &alternating, &[2, 3, 1]);
        let recorded = provider
            .deep_learning_layer_norm_recorded_exec(&input, &scale, &offset, &request)
            .unwrap();
        let gradients = provider
            .deep_learning_layer_norm_adjoint_exec(
                &input,
                &scale,
                &offset,
                &recorded.statistics,
                &upstream,
                &request,
            )
            .unwrap();
        let actual = download(&provider, &gradients.input);
        assert_eq!(actual.len(), dy.len());
        let signed_xhat = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let sum_dy = dy.iter().sum::<f64>();
        let sum_weighted = dy
            .iter()
            .zip(signed_xhat)
            .map(|(seed, xhat)| seed * xhat)
            .sum::<f64>();
        let tolerance = match precision {
            ProviderPrecision::F32 => 2e-2,
            ProviderPrecision::F64 => 2e-10,
        };
        for index in 0..dy.len() {
            let expected =
                (6.0 * dy[index] - sum_dy - signed_xhat[index] * sum_weighted) / magnitude / 6.0;
            assert!(
                expected != 0.0 && actual[index] != 0.0,
                "large finite dX must not underflow to zero"
            );
            assert!(
                ((actual[index] - expected) / expected).abs() < tolerance,
                "dX[{index}]={} != {expected}",
                actual[index]
            );
        }

        let small_upstream = upload(&provider, &[1., 2., 3., 4., 5., 6.], &[2, 3, 1]);
        if magnitude == magnitudes[1] {
            balanced_native_max_adjoint(&provider, magnitude, seed_scale, tolerance);
        }
        for constant_value in [magnitude, -magnitude] {
            let constant = [constant_value; 6];
            let constant_input = upload(&provider, &constant, &[2, 3, 1]);
            let constant_recorded = provider
                .deep_learning_layer_norm_recorded_exec(&constant_input, &scale, &offset, &request)
                .unwrap();
            let constant_gradients = provider
                .deep_learning_layer_norm_adjoint_exec(
                    &constant_input,
                    &scale,
                    &offset,
                    &constant_recorded.statistics,
                    &small_upstream,
                    &request,
                )
                .unwrap();
            let constant_dx = download(&provider, &constant_gradients.input);
            assert_eq!(
                download(&provider, &constant_recorded.statistics)[2],
                0.0,
                "exact constant groups must retain the zero-variance sentinel"
            );
            assert_eq!(download(&provider, &constant_recorded.output), vec![0.0; 6]);
            assert_eq!(download(&provider, &constant_gradients.scale), vec![0.0; 3]);
            assert_eq!(
                download(&provider, &constant_gradients.offset),
                vec![3.0, 7.0, 11.0]
            );
            assert_eq!(constant_dx.len(), 6);
            let inv_epsilon = request.epsilon.sqrt().recip();
            for (index, actual) in constant_dx.iter().enumerate() {
                let expected = (index as f64 + 1.0 - 3.5) * inv_epsilon;
                assert!(
            (actual - expected).abs()
                < match precision {
                    ProviderPrecision::F32 => 2e-2,
                    ProviderPrecision::F64 => 1e-10,
                },
            "constant {precision:?} dX[{index}]={actual} != {expected} at magnitude {constant_value}"
        );
            }
        }
    }
}

fn balanced_native_max_adjoint(
    provider: &WgpuProvider,
    magnitude: f64,
    seed_scale: f64,
    tolerance: f64,
) {
    // This exact balanced ordering makes a plain IEEE F32 shifted reduction
    // round Var(X)/M^2 to 1.000000119, beyond its mathematical upper bound.
    let signs = (0..50)
        .map(|index| {
            if (index * 3 + 11) % 50 < 25 {
                1.0
            } else {
                -1.0
            }
        })
        .collect::<Vec<_>>();
    let input_values = signs
        .iter()
        .map(|sign| sign * magnitude)
        .collect::<Vec<_>>();
    let seed = (0..50)
        .map(|index| (index as f64 % 7.0 - 2.3) * seed_scale)
        .collect::<Vec<_>>();
    let input = upload(provider, &input_values, &[2, 5, 5]);
    let upstream = upload(provider, &seed, &[2, 5, 5]);
    let scale = upload(provider, &[1.0; 5], &[1, 5]);
    let offset = upload(provider, &[0.0; 5], &[1, 5]);
    let request = DeepLearningLayerNormRequest {
        shape: vec![2, 5, 5],
        reduce_mask: 0b111,
        channel_axis: 1,
        scale_shape: vec![1, 5],
        offset_shape: vec![1, 5],
        epsilon: 1e-5,
    };
    let recorded = provider
        .deep_learning_layer_norm_recorded_exec(&input, &scale, &offset, &request)
        .unwrap();
    let gradients = provider
        .deep_learning_layer_norm_adjoint_exec(
            &input,
            &scale,
            &offset,
            &recorded.statistics,
            &upstream,
            &request,
        )
        .unwrap();
    let actual_dx = download(provider, &gradients.input);
    let actual_forward = download(provider, &recorded.output);
    let sum_seed = seed.iter().sum::<f64>();
    let sum_weighted = seed
        .iter()
        .zip(&signs)
        .map(|(seed, sign)| seed * sign)
        .sum::<f64>();
    for index in 0..50 {
        assert!((actual_forward[index] - signs[index]).abs() < tolerance);
        let expected =
            (50.0 * seed[index] - sum_seed - signs[index] * sum_weighted) / magnitude / 50.0;
        assert!(
            expected != 0.0 && actual_dx[index] != 0.0,
            "native MAX derivative must remain normal and nonzero"
        );
        assert!(
            ((actual_dx[index] - expected) / expected).abs() < tolerance,
            "balanced native MAX dx[{index}]={} != {expected}",
            actual_dx[index]
        );
    }
}

#[test]
#[ignore = "requires actual WGPU; RUNMAT_WGPU_FORCE_PRECISION=f32"]
fn extreme_normalization_adjoint_f32() {
    extreme_normalization_adjoint_keeps_representable_gradients(ProviderPrecision::F32);
}

fn adjacent_extreme_normalization_matches_spacing_oracle(precision: ProviderPrecision) {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    assert_eq!(provider.precision(), precision);
    let (values, spacing) = match precision {
        ProviderPrecision::F32 => {
            let bits = 1e38_f32.to_bits();
            let values = (0..6)
                .map(|step| f64::from(f32::from_bits(bits + step)))
                .collect::<Vec<_>>();
            let spacing = f64::from(f32::from_bits(bits + 1)) - f64::from(f32::from_bits(bits));
            (values, spacing)
        }
        ProviderPrecision::F64 => {
            let bits = 1e300_f64.to_bits();
            let values = (0..6)
                .map(|step| f64::from_bits(bits + step))
                .collect::<Vec<_>>();
            let spacing = f64::from_bits(bits + 1) - f64::from_bits(bits);
            (values, spacing)
        }
    };
    let seed = [1.0, 0.0, -2.0, 3.0, 0.5, -1.0];
    let request = DeepLearningLayerNormRequest {
        shape: vec![2, 3, 1],
        reduce_mask: 0b011,
        channel_axis: 1,
        scale_shape: vec![1, 3],
        offset_shape: vec![1, 3],
        epsilon: 1e-5,
    };
    let scale = upload(&provider, &[1.0; 3], &[1, 3]);
    let offset = upload(&provider, &[0.0; 3], &[1, 3]);
    let upstream = upload(&provider, &seed, &[2, 3, 1]);
    // This independent oracle uses exact native ulp offsets 0..5 and their
    // integer sum, not subtraction of a rounded enormous mean or x*x.
    let sigma_steps = (35.0_f64 / 12.0).sqrt();
    let xhat = [0., 1., 2., 3., 4., 5.].map(|step| (step - 2.5) / sigma_steps);
    let mean_seed = seed.iter().sum::<f64>() / 6.0;
    let mean_weighted = seed.iter().zip(xhat).map(|(dy, x)| dy * x).sum::<f64>() / 6.0;
    let tolerance = match precision {
        ProviderPrecision::F32 => 2e-4,
        ProviderPrecision::F64 => 2e-10,
    };
    for sign in [1.0, -1.0] {
        let input_values = values.iter().map(|value| sign * value).collect::<Vec<_>>();
        let input = upload(&provider, &input_values, &[2, 3, 1]);
        let recorded = provider
            .deep_learning_layer_norm_recorded_exec(&input, &scale, &offset, &request)
            .unwrap();
        let gradients = provider
            .deep_learning_layer_norm_adjoint_exec(
                &input,
                &scale,
                &offset,
                &recorded.statistics,
                &upstream,
                &request,
            )
            .unwrap();
        let actual_forward = download(&provider, &recorded.output);
        let actual_dx = download(&provider, &gradients.input);
        for index in 0..6 {
            assert!(
                (actual_forward[index] - sign * xhat[index]).abs() < tolerance,
                "adjacent {precision:?} forward[{index}]={} != {}",
                actual_forward[index],
                sign * xhat[index]
            );
            let expected_dx =
                (seed[index] - mean_seed - xhat[index] * mean_weighted) / spacing / sigma_steps;
            assert!(expected_dx != 0.0 && actual_dx[index] != 0.0);
            assert!(
                ((actual_dx[index] - expected_dx) / expected_dx).abs() < tolerance,
                "adjacent {precision:?} dx[{index}]={} != {expected_dx}",
                actual_dx[index]
            );
        }
    }
}

#[test]
#[ignore = "requires actual WGPU; RUNMAT_WGPU_FORCE_PRECISION=f32"]
fn adjacent_extreme_normalization_adjoint_f32() {
    adjacent_extreme_normalization_matches_spacing_oracle(ProviderPrecision::F32);
}

#[test]
#[ignore = "requires SHADER_F64 WGPU; RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn adjacent_extreme_normalization_adjoint_f64() {
    adjacent_extreme_normalization_matches_spacing_oracle(ProviderPrecision::F64);
}

#[test]
#[ignore = "requires SHADER_F64 WGPU; RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn extreme_normalization_adjoint_f64() {
    extreme_normalization_adjoint_keeps_representable_gradients(ProviderPrecision::F64);
}

#[test]
#[ignore = "requires actual WGPU; nonfinite native-precision normalization adjoint"]
fn nonfinite_normalization_keeps_undefined_derivatives_and_finite_offset_sum() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    let scale = upload(&provider, &[1.0, 1.0, 1.0], &[1, 3]);
    let offset = upload(&provider, &[0.0, 0.0, 0.0], &[1, 3]);
    let upstream = upload(&provider, &[1.0; 6], &[2, 3, 1]);
    let request = DeepLearningLayerNormRequest {
        shape: vec![2, 3, 1],
        reduce_mask: 0b011,
        channel_axis: 1,
        scale_shape: vec![1, 3],
        offset_shape: vec![1, 3],
        epsilon: 1e-5,
    };
    for special in [f64::NAN, f64::INFINITY] {
        let input = upload(&provider, &[1.0, 2.0, special, 4.0, 5.0, 6.0], &[2, 3, 1]);
        let recorded = provider
            .deep_learning_layer_norm_recorded_exec(&input, &scale, &offset, &request)
            .unwrap();
        let gradients = provider
            .deep_learning_layer_norm_adjoint_exec(
                &input,
                &scale,
                &offset,
                &recorded.statistics,
                &upstream,
                &request,
            )
            .unwrap();
        let dx = download(&provider, &gradients.input);
        let dgamma = download(&provider, &gradients.scale);
        let dbeta = download(&provider, &gradients.offset);
        assert_eq!(dx.len(), 6);
        assert_eq!(dgamma.len(), 3);
        assert_eq!(dbeta, vec![2.0; 3]);
        assert!(dx.iter().all(|value| value.is_nan()));
        assert!(dgamma.iter().all(|value| value.is_nan()));
    }
}
