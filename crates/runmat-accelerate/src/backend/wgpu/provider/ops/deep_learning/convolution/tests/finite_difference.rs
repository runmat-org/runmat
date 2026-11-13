use super::*;

fn weighted_gpu_output(
    provider: &WgpuProvider,
    request: &DeepLearningConv1DRequest,
    input: &[f64; 5],
    weights: &[f64; 3],
    bias: f64,
    upstream: &[f64; 2],
) -> f64 {
    let x = upload_native(provider, input, &[5, 1, 1]);
    let w = upload_native(provider, weights, &[3, 1, 1]);
    let b = upload_native(provider, &[bias], &[1, 1]);
    let y = provider
        .deep_learning_conv1d_exec(&x, &w, &b, request)
        .unwrap();
    let values = download_native(provider, &y);
    assert_eq!(values.len(), 2);
    values[0] * upstream[0] + values[1] * upstream[1]
}

fn boundary_adjoint_matches_actual_forward_finite_difference(precision: NumericPrecision) {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    assert_eq!(
        provider.precision, precision,
        "set RUNMAT_WGPU_FORCE_PRECISION"
    );
    let input = [1., 2., 3., 4., 5.];
    let weights = [0.3, -0.4, 0.8];
    let upstream = [0.7, -1.3];
    let epsilon = match precision {
        NumericPrecision::F32 => 0.01,
        NumericPrecision::F64 => 1e-5,
    };
    let tolerance = match precision {
        NumericPrecision::F32 => 0.02,
        NumericPrecision::F64 => 1e-7,
    };
    for (padding_value, expected_first) in [
        (DeepLearningConv1DPaddingValue::Constant(0.25), 1.875),
        (DeepLearningConv1DPaddingValue::Replicate, 2.1),
        (DeepLearningConv1DPaddingValue::SymmetricIncludeEdge, 2.4),
        (DeepLearningConv1DPaddingValue::SymmetricExcludeEdge, 2.7),
    ] {
        let request = DeepLearningConv1DRequest {
            input_shape: vec![5, 1, 1],
            output_shape: vec![2, 1, 1],
            weights_shape: [3, 1, 1],
            bias_shape: [1, 1],
            span_axis: 0,
            channel_axis: 1,
            batch_axis: Some(2),
            time_axis: None,
            stride: 2,
            dilation: 2,
            padding_left: 2,
            padding_right: 1,
            padding_value,
        };
        let x = upload_native(&provider, &input, &[5, 1, 1]);
        let w = upload_native(&provider, &weights, &[3, 1, 1]);
        let b = upload_native(&provider, &[-0.2], &[1, 1]);
        let dy = upload_native(&provider, &upstream, &[2, 1, 1]);
        let y = provider
            .deep_learning_conv1d_exec(&x, &w, &b, &request)
            .unwrap();
        let actual_y = download_native(&provider, &y);
        assert_eq!(actual_y.len(), 2);
        assert!(
            (actual_y[0] - expected_first).abs() < tolerance,
            "{precision:?} {padding_value:?}: first output {} != {expected_first}",
            actual_y[0]
        );
        assert!(
            (actual_y[1] - 2.9).abs() < tolerance,
            "{precision:?} {padding_value:?}: second output {} != 2.9",
            actual_y[1]
        );
        provider.reset_telemetry();
        let gradients = provider
            .deep_learning_conv1d_adjoint_exec(&x, &w, &b, &dy, &request)
            .unwrap();
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let dx = download_native(&provider, &gradients.input);
        let dw = download_native(&provider, &gradients.weights);
        let db = download_native(&provider, &gradients.bias);
        assert_eq!(dx.len(), 5);
        assert_eq!(dw.len(), 3);
        assert_eq!(db.len(), 1);
        for index in 0..input.len() {
            let mut plus = input;
            let mut minus = input;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical =
                (weighted_gpu_output(&provider, &request, &plus, &weights, -0.2, &upstream)
                    - weighted_gpu_output(&provider, &request, &minus, &weights, -0.2, &upstream))
                    / (2.0 * epsilon);
            assert!(
                (dx[index] - numerical).abs() < tolerance,
                "dX[{index}] for {padding_value:?}"
            );
        }
        for index in 0..weights.len() {
            let mut plus = weights;
            let mut minus = weights;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical =
                (weighted_gpu_output(&provider, &request, &input, &plus, -0.2, &upstream)
                    - weighted_gpu_output(&provider, &request, &input, &minus, -0.2, &upstream))
                    / (2.0 * epsilon);
            assert!(
                (dw[index] - numerical).abs() < tolerance,
                "dW[{index}] for {padding_value:?}"
            );
        }
        let numerical_bias = (weighted_gpu_output(
            &provider,
            &request,
            &input,
            &weights,
            -0.2 + epsilon,
            &upstream,
        ) - weighted_gpu_output(
            &provider,
            &request,
            &input,
            &weights,
            -0.2 - epsilon,
            &upstream,
        )) / (2.0 * epsilon);
        assert!((db[0] - numerical_bias).abs() < tolerance);
    }
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F32 precision"]
fn learned_conv1d_f32_stride_dilation_and_padding_adjoint() {
    boundary_adjoint_matches_actual_forward_finite_difference(NumericPrecision::F32);
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F64 precision"]
fn learned_conv1d_f64_stride_dilation_and_padding_adjoint() {
    boundary_adjoint_matches_actual_forward_finite_difference(NumericPrecision::F64);
}
