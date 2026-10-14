use super::*;

fn singleton_exclude_edge_without_padding(precision: NumericPrecision) {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    assert_eq!(
        provider.precision, precision,
        "set RUNMAT_WGPU_FORCE_PRECISION"
    );
    let request = DeepLearningConv1DRequest {
        input_shape: vec![1, 1, 1],
        output_shape: vec![1, 1, 1],
        weights_shape: [1, 1, 1],
        bias_shape: [1, 1],
        span_axis: 0,
        channel_axis: 1,
        batch_axis: Some(2),
        time_axis: None,
        stride: 1,
        dilation: 1,
        padding_left: 0,
        padding_right: 0,
        padding_value: DeepLearningConv1DPaddingValue::SymmetricExcludeEdge,
    };
    let x = upload_native(&provider, &[2.], &[1, 1, 1]);
    let w = upload_native(&provider, &[3.], &[1, 1, 1]);
    let b = upload_native(&provider, &[4.], &[1, 1]);
    let dy = upload_native(&provider, &[5.], &[1, 1, 1]);
    let y = provider
        .deep_learning_conv1d_exec(&x, &w, &b, &request)
        .unwrap();
    let gradients = provider
        .deep_learning_conv1d_adjoint_exec(&x, &w, &b, &dy, &request)
        .unwrap();
    assert_eq!(download_native(&provider, &y), vec![10.]);
    assert_eq!(download_native(&provider, &gradients.input), vec![15.]);
    assert_eq!(download_native(&provider, &gradients.weights), vec![10.]);
    assert_eq!(download_native(&provider, &gradients.bias), vec![5.]);
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F32 precision"]
fn learned_conv1d_f32_singleton_exclude_edge_without_padding() {
    singleton_exclude_edge_without_padding(NumericPrecision::F32);
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F64 precision"]
fn learned_conv1d_f64_singleton_exclude_edge_without_padding() {
    singleton_exclude_edge_without_padding(NumericPrecision::F64);
}
