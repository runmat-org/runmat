use super::*;

#[test]
#[ignore = "requires actual WGPU adapter; empty output and gradient buffers must be written"]
fn learned_conv1d_empty_batch_has_zero_parameter_adjoints() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    let request = DeepLearningConv1DRequest {
        input_shape: vec![5, 1, 0],
        output_shape: vec![2, 1, 0],
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
        padding_value: DeepLearningConv1DPaddingValue::Constant(0.0),
    };
    let x = upload_native(&provider, &[], &[5, 1, 0]);
    let w = upload_native(&provider, &[1., 2., 3.], &[3, 1, 1]);
    let b = upload_native(&provider, &[4.], &[1, 1]);
    let dy = upload_native(&provider, &[], &[2, 1, 0]);
    let y = provider
        .deep_learning_conv1d_exec(&x, &w, &b, &request)
        .unwrap();
    assert_eq!(y.shape(), &[2, 1, 0]);
    let gradients = provider
        .deep_learning_conv1d_adjoint_exec(&x, &w, &b, &dy, &request)
        .unwrap();
    assert_eq!(gradients.input.shape(), &[5, 1, 0]);
    assert_eq!(
        download_native(&provider, &gradients.weights),
        vec![0., 0., 0.]
    );
    assert_eq!(download_native(&provider, &gradients.bias), vec![0.]);
}
