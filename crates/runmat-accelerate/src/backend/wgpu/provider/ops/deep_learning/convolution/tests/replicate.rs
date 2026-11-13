use super::*;

fn large_stride_single_sample_replication(precision: NumericPrecision) {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    assert_eq!(
        provider.precision, precision,
        "set RUNMAT_WGPU_FORCE_PRECISION"
    );
    let request = DeepLearningConv1DRequest {
        input_shape: vec![1, 1, 1],
        output_shape: vec![2, 1, 1],
        weights_shape: [2, 1, 1],
        bias_shape: [1, 1],
        span_axis: 0,
        channel_axis: 1,
        batch_axis: Some(2),
        time_axis: None,
        stride: 1_000_000,
        dilation: 3,
        padding_left: 1_000_000,
        padding_right: 4,
        padding_value: DeepLearningConv1DPaddingValue::Replicate,
    };
    let x = upload_native(&provider, &[2.], &[1, 1, 1]);
    let w = upload_native(&provider, &[0.3, -0.4], &[2, 1, 1]);
    let b = upload_native(&provider, &[0.1], &[1, 1]);
    let dy = upload_native(&provider, &[1.25, -0.5], &[2, 1, 1]);
    let y = provider
        .deep_learning_conv1d_exec(&x, &w, &b, &request)
        .unwrap();
    let gradients = provider
        .deep_learning_conv1d_adjoint_exec(&x, &w, &b, &dy, &request)
        .unwrap();
    let tolerance = if precision == NumericPrecision::F32 {
        1e-6
    } else {
        1e-12
    };
    for (actual, expected) in [
        (download_native(&provider, &y), vec![-0.1, -0.1]),
        (download_native(&provider, &gradients.input), vec![-0.075]),
        (
            download_native(&provider, &gradients.weights),
            vec![1.5, 1.5],
        ),
        (download_native(&provider, &gradients.bias), vec![0.75]),
    ] {
        assert_eq!(actual.len(), expected.len());
        for (observed, target) in actual.iter().zip(expected) {
            assert!((observed - target).abs() < tolerance);
        }
    }
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F32 precision"]
fn learned_conv1d_f32_single_sample_replication_does_not_walk_padding() {
    large_stride_single_sample_replication(NumericPrecision::F32);
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F64 precision"]
fn learned_conv1d_f64_single_sample_replication_does_not_walk_padding() {
    large_stride_single_sample_replication(NumericPrecision::F64);
}
