use super::*;

fn exact_multichannel_forward_and_adjoint(precision: NumericPrecision) {
    use runmat_accelerate_api::AccelProvider;

    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    assert_eq!(
        provider.precision, precision,
        "set RUNMAT_WGPU_FORCE_PRECISION"
    );
    let request = DeepLearningConv1DRequest {
        input_shape: vec![4, 2, 1],
        output_shape: vec![3, 2, 1],
        weights_shape: [2, 2, 2],
        bias_shape: [1, 2],
        span_axis: 0,
        channel_axis: 1,
        batch_axis: Some(2),
        time_axis: None,
        stride: 1,
        dilation: 1,
        padding_left: 0,
        padding_right: 0,
        padding_value: DeepLearningConv1DPaddingValue::Constant(0.0),
    };
    let input = upload_native(&provider, &[1., 2., 3., 4., 5., 6., 7., 8.], &[4, 2, 1]);
    let weights = upload_native(&provider, &[1., 2., 3., 4., -1., 1., 2., -2.], &[2, 2, 2]);
    let bias = upload_native(&provider, &[0.5, -1.], &[1, 2]);
    let upstream = upload_native(&provider, &[1., 2., 3., -1., 0.5, 2.], &[3, 2, 1]);
    let valid = DeepLearningConv1DOperands {
        anchor: &input,
        input: Some(&input),
        weights: Some(&weights),
        bias: Some(&bias),
        upstream: Some(&upstream),
    };
    let before = provider.telemetry_snapshot();
    assert_eq!(
        provider
            .deep_learning_conv1d_capability_exec(valid, &request)
            .unwrap(),
        DeepLearningConv1DCapability::Ready
    );
    assert_eq!(provider.telemetry_snapshot(), before);
    let wrong_bias = upload_native(&provider, &[0.5, -1.], &[2, 1]);
    let before_invalid = provider.telemetry_snapshot();
    assert!(provider
        .deep_learning_conv1d_capability_exec(
            DeepLearningConv1DOperands {
                bias: Some(&wrong_bias),
                ..valid
            },
            &request,
        )
        .is_err());
    assert_eq!(provider.telemetry_snapshot(), before_invalid);
    let logical_upstream = provider
        .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
            data: &[1, 0, 1, 0, 1, 0],
            shape: &[3, 2, 1],
        })
        .unwrap();
    let before_logical = provider.telemetry_snapshot();
    assert_eq!(
        provider
            .deep_learning_conv1d_capability_exec(
                DeepLearningConv1DOperands {
                    upstream: Some(&logical_upstream),
                    ..valid
                },
                &request,
            )
            .unwrap(),
        DeepLearningConv1DCapability::UnsupportedOperation
    );
    assert_eq!(provider.telemetry_snapshot(), before_logical);
    let foreign_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let foreign = foreign_owner
        .allocate_zeroed(
            request.weights_shape.to_vec(),
            input.physical().element_type(),
            GpuTensorStorage::Real,
        )
        .unwrap();
    let before_foreign = provider.telemetry_snapshot();
    assert!(provider
        .deep_learning_conv1d_capability_exec(
            DeepLearningConv1DOperands {
                weights: Some(&foreign.handle),
                ..valid
            },
            &request,
        )
        .is_err());
    assert_eq!(provider.telemetry_snapshot(), before_foreign);
    assert_eq!(foreign.probe.retirements(), 0);
    assert!(
        provider
            .deep_learning_conv1d_capability_exec(
                DeepLearningConv1DOperands {
                    anchor: &logical_upstream,
                    weights: Some(&foreign.handle),
                    ..valid
                },
                &request,
            )
            .is_err(),
        "unsupported source must not hide a later foreign lease"
    );
    assert_eq!(provider.telemetry_snapshot(), before_foreign);
    let mut too_large = request.clone();
    too_large.padding_left = (i32::MAX as usize) / 4 + 1;
    too_large.output_shape[0] = too_large.padding_left + 3;
    assert!(Geometry::new(&too_large, precision).unwrap().is_none());
    let before_combined = provider.telemetry_snapshot();
    assert!(provider
        .deep_learning_conv1d_capability_exec(
            DeepLearningConv1DOperands {
                weights: Some(&foreign.handle),
                bias: None,
                upstream: None,
                ..valid
            },
            &too_large,
        )
        .is_err());
    assert_eq!(provider.telemetry_snapshot(), before_combined);
    assert_eq!(
        provider
            .deep_learning_conv1d_capability_exec(
                DeepLearningConv1DOperands {
                    weights: None,
                    bias: None,
                    upstream: None,
                    ..valid
                },
                &too_large,
            )
            .unwrap(),
        DeepLearningConv1DCapability::ResourceLimit
    );
    assert_eq!(provider.telemetry_snapshot(), before_combined);
    let mut forged = weights.wire_descriptor();
    forged.physical = runmat_accelerate_api::allocation::GpuPhysicalDescriptor::new(
        runmat_accelerate_api::NumericElementType::U8,
        GpuTensorStorage::Real,
        forged.physical.element_capacity(),
        forged.physical.payload_bytes(),
        forged.physical.requested_bytes(),
    )
    .unwrap();
    let before_forged = provider.telemetry_snapshot();
    assert!(provider.acquire_live(&forged).is_err());
    forged = weights.wire_descriptor();
    forged.view = forged.view.with_shape(vec![9, 1, 1]);
    assert!(provider.acquire_live(&forged).is_err());
    assert_eq!(provider.telemetry_snapshot(), before_forged);
    let mut duplicate_axes = request.clone();
    duplicate_axes.channel_axis = duplicate_axes.span_axis;
    let before_axes = provider.telemetry_snapshot();
    assert!(provider
        .deep_learning_conv1d_capability_exec(valid, &duplicate_axes)
        .is_err());
    assert_eq!(provider.telemetry_snapshot(), before_axes);
    let stale = upload_native(&provider, &[1., 2.], &[2, 1]);
    let stale_wire = stale.wire_descriptor();
    stale.release().unwrap();
    let before_stale = provider.telemetry_snapshot();
    assert!(provider.acquire_live(&stale_wire).is_err());
    assert_eq!(provider.telemetry_snapshot(), before_stale);
    provider.reset_telemetry();

    let output = provider
        .deep_learning_conv1d_exec(&input, &weights, &bias, &request)
        .unwrap();
    let gradients = provider
        .deep_learning_conv1d_adjoint_exec(&input, &weights, &bias, &upstream, &request)
        .unwrap();
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    for result in [
        &output,
        &gradients.input,
        &gradients.weights,
        &gradients.bias,
    ] {
        assert_eq!(result.device_id(), input.device_id());
        assert_eq!(
            result.physical().element_type(),
            input.physical().element_type()
        );
        assert!(!result.shares_allocation_with(&input));
        assert!(!result.shares_allocation_with(&weights));
        assert!(!result.shares_allocation_with(&upstream));
    }
    for (actual, expected) in [
        (
            download_native(&provider, &output),
            vec![44.5, 54.5, 64.5, -2., -2., -2.],
        ),
        (
            download_native(&provider, &gradients.input),
            vec![2., 2.5, 5.5, 8., 1., 13., 20., 8.],
        ),
        (
            download_native(&provider, &gradients.weights),
            vec![14., 20., 38., 44., 6., 7.5, 12., 13.5],
        ),
        (download_native(&provider, &gradients.bias), vec![6., 1.5]),
    ] {
        assert_eq!(actual.len(), expected.len());
        for (observed, target) in actual.iter().zip(expected) {
            assert!((observed - target).abs() < 1e-5, "{observed} != {target}");
        }
    }
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F32 precision"]
fn learned_conv1d_f32_multichannel_forward_and_adjoint() {
    exact_multichannel_forward_and_adjoint(NumericPrecision::F32);
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F64 precision"]
fn learned_conv1d_f64_multichannel_forward_and_adjoint() {
    exact_multichannel_forward_and_adjoint(NumericPrecision::F64);
}

fn non_square_spatial_channel_batch_time_oracle(precision: NumericPrecision) {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    assert_eq!(
        provider.precision, precision,
        "set RUNMAT_WGPU_FORCE_PRECISION"
    );
    const INPUT_SPAN: usize = 5;
    const INPUT_CHANNELS: usize = 2;
    const BATCHES: usize = 2;
    const TIMES: usize = 2;
    const FILTER_SIZE: usize = 2;
    const OUTPUT_CHANNELS: usize = 3;
    const OUTPUT_SPAN: usize = 3;
    let input_values: Vec<f64> = (0..INPUT_SPAN * INPUT_CHANNELS * BATCHES * TIMES)
        .map(|index| (index as f64 - 17.0) * 0.125)
        .collect();
    let weight_values: Vec<f64> = (0..FILTER_SIZE * INPUT_CHANNELS * OUTPUT_CHANNELS)
        .map(|index| ((index % 7) as f64 - 3.0) * 0.25)
        .collect();
    let bias_values = [0.5, -0.25, 1.0];
    let upstream_values: Vec<f64> = (0..OUTPUT_SPAN * OUTPUT_CHANNELS * BATCHES * TIMES)
        .map(|index| ((index % 9) as f64 - 4.0) * 0.125)
        .collect();
    let input_shape = [INPUT_SPAN, INPUT_CHANNELS, BATCHES, TIMES];
    let weight_shape = [FILTER_SIZE, INPUT_CHANNELS, OUTPUT_CHANNELS];
    let output_shape = [OUTPUT_SPAN, OUTPUT_CHANNELS, BATCHES, TIMES];
    let request = DeepLearningConv1DRequest {
        input_shape: input_shape.to_vec(),
        output_shape: output_shape.to_vec(),
        weights_shape: weight_shape,
        bias_shape: [1, OUTPUT_CHANNELS],
        span_axis: 0,
        channel_axis: 1,
        batch_axis: Some(2),
        time_axis: Some(3),
        stride: 2,
        dilation: 2,
        padding_left: 1,
        padding_right: 1,
        padding_value: DeepLearningConv1DPaddingValue::Constant(0.25),
    };
    let x = upload_native(&provider, &input_values, &input_shape);
    let w = upload_native(&provider, &weight_values, &weight_shape);
    let b = upload_native(&provider, &bias_values, &[1, OUTPUT_CHANNELS]);
    let dy = upload_native(&provider, &upstream_values, &output_shape);
    provider.reset_telemetry();
    let output = provider
        .deep_learning_conv1d_exec(&x, &w, &b, &request)
        .unwrap();
    let gradients = provider
        .deep_learning_conv1d_adjoint_exec(&x, &w, &b, &dy, &request)
        .unwrap();
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(output.shape(), output_shape);
    assert_eq!(gradients.input.shape(), input_shape);
    assert_eq!(gradients.weights.shape(), weight_shape);
    assert_eq!(gradients.bias.shape(), [1, OUTPUT_CHANNELS]);
    for result in [
        &output,
        &gradients.input,
        &gradients.weights,
        &gradients.bias,
    ] {
        assert_eq!(result.device_id(), x.device_id());
        assert_eq!(
            result.physical().element_type(),
            x.physical().element_type()
        );
        assert!(!result.shares_allocation_with(&x));
        assert!(!result.shares_allocation_with(&w));
        assert!(!result.shares_allocation_with(&dy));
    }

    let mut expected_output = vec![0.0; upstream_values.len()];
    let mut expected_dx = vec![0.0; input_values.len()];
    let mut expected_dw = vec![0.0; weight_values.len()];
    let mut expected_db = vec![0.0; OUTPUT_CHANNELS];
    for time in 0..TIMES {
        for batch in 0..BATCHES {
            for out_channel in 0..OUTPUT_CHANNELS {
                for out_span in 0..OUTPUT_SPAN {
                    let output_index = out_span
                        + OUTPUT_SPAN * (out_channel + OUTPUT_CHANNELS * (batch + BATCHES * time));
                    let derivative = upstream_values[output_index];
                    let mut value = bias_values[out_channel];
                    expected_db[out_channel] += derivative;
                    for in_channel in 0..INPUT_CHANNELS {
                        for tap in 0..FILTER_SIZE {
                            let source = out_span as isize * 2 + tap as isize * 2 - 1;
                            let weight_index =
                                tap + FILTER_SIZE * (in_channel + INPUT_CHANNELS * out_channel);
                            let input_index =
                                (0..INPUT_SPAN as isize).contains(&source).then(|| {
                                    source as usize
                                        + INPUT_SPAN
                                            * (in_channel
                                                + INPUT_CHANNELS * (batch + BATCHES * time))
                                });
                            let source_value =
                                input_index.map(|index| input_values[index]).unwrap_or(0.25);
                            value += source_value * weight_values[weight_index];
                            expected_dw[weight_index] += derivative * source_value;
                            if let Some(index) = input_index {
                                expected_dx[index] += derivative * weight_values[weight_index];
                            }
                        }
                    }
                    expected_output[output_index] = value;
                }
            }
        }
    }
    let tolerance = match precision {
        NumericPrecision::F32 => 1e-5,
        NumericPrecision::F64 => 1e-11,
    };
    for (name, actual, expected) in [
        ("Y", download_native(&provider, &output), expected_output),
        (
            "dX",
            download_native(&provider, &gradients.input),
            expected_dx,
        ),
        (
            "dW",
            download_native(&provider, &gradients.weights),
            expected_dw,
        ),
        (
            "dBias",
            download_native(&provider, &gradients.bias),
            expected_db,
        ),
    ] {
        assert_eq!(actual.len(), expected.len(), "{name} length");
        for (index, (observed, target)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (observed - target).abs() < tolerance,
                "{name}[{index}] {observed} != {target}"
            );
        }
    }
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F32 precision"]
fn learned_conv1d_f32_non_square_spatial_channel_batch_time() {
    non_square_spatial_channel_batch_time_oracle(NumericPrecision::F32);
}

#[test]
#[ignore = "requires actual WGPU adapter and forced F64 precision"]
fn learned_conv1d_f64_non_square_spatial_channel_batch_time() {
    non_square_spatial_channel_batch_time_oracle(NumericPrecision::F64);
}
