use super::*;

#[test]
fn physical_geometry_preserves_sc_b_t_positions_and_payload_counts() {
    let geometry = Geometry::new(&request(), NumericPrecision::F32)
        .unwrap()
        .expect("small physical geometry is supported");
    assert_eq!(geometry.input_elements, 7 * 3 * 2 * 4);
    assert_eq!(geometry.output_elements, 4 * 5 * 2 * 4);
    assert_eq!(geometry.weight_elements, 3 * 3 * 5);
    assert_eq!(geometry.bias_elements, 5);
    assert_eq!(geometry.params.input_shape, [7, 3, 2, 4]);
    assert_eq!(geometry.params.output_shape, [4, 5, 2, 4]);
    assert_eq!(geometry.params.input_strides, [1, 7, 21, 42]);
    assert_eq!(geometry.params.output_strides, [1, 4, 20, 40]);
    assert_eq!(geometry.params.padding_mode, 1);
    let uniform = std::mem::size_of::<ConvParams>() as u64;
    assert_eq!(uniform, 160, "Rust/WGSL convolution uniform layout");
    assert_eq!(std::mem::offset_of!(ConvParams, padding_bits), 128);
    assert_eq!(std::mem::offset_of!(ConvParams, chunk_offset), 136);
    assert_eq!(geometry.forward_owned_bytes(4), Some(160 * 4 + uniform));
    assert_eq!(
        geometry.adjoint_owned_bytes(4),
        Some((168 + 45 + 5) * 4 + 3 * uniform)
    );
}

#[test]
fn constant_padding_uses_native_uniform_bits_at_both_precisions() {
    let mut descriptor = request();
    descriptor.padding_value = DeepLearningConv1DPaddingValue::Constant(1.25);
    let single = Geometry::new(&descriptor, NumericPrecision::F32)
        .unwrap()
        .unwrap();
    let double = Geometry::new(&descriptor, NumericPrecision::F64)
        .unwrap()
        .unwrap();
    assert_eq!(single.params.padding_bits, u64::from(1.25f32.to_bits()));
    assert_eq!(double.params.padding_bits, 1.25f64.to_bits());
}

#[test]
fn reflection_stride_constants_follow_admitted_physical_geometry() {
    let mut descriptor = request();
    descriptor.stride = 5;
    descriptor.output_shape[0] = 2;
    descriptor.padding_value = DeepLearningConv1DPaddingValue::SymmetricIncludeEdge;
    let included = Geometry::new(&descriptor, NumericPrecision::F32)
        .unwrap()
        .unwrap();
    assert_eq!(
        (
            included.params.reflection_step,
            included.params.reflection_inverse
        ),
        (14, 3)
    );

    descriptor.padding_value = DeepLearningConv1DPaddingValue::SymmetricExcludeEdge;
    let excluded = Geometry::new(&descriptor, NumericPrecision::F64)
        .unwrap()
        .unwrap();
    assert_eq!(
        (
            excluded.params.reflection_step,
            excluded.params.reflection_inverse
        ),
        (12, 5)
    );
}

#[test]
fn malformed_geometry_and_supported_resource_limits_are_distinct() {
    let mut malformed = request();
    malformed.channel_axis = malformed.span_axis;
    assert!(Geometry::new(&malformed, NumericPrecision::F32).is_err());

    let mut unsupported = request();
    unsupported.padding_left = (i32::MAX as usize) / 4 + 1;
    unsupported.output_shape[0] = (7 + unsupported.padding_left + 1 - 3) / 2 + 1;
    assert!(Geometry::new(&unsupported, NumericPrecision::F32)
        .unwrap()
        .is_none());

    let mut wrong_shape = request();
    wrong_shape.output_shape[0] = 5;
    assert!(Geometry::new(&wrong_shape, NumericPrecision::F32).is_err());
}

#[test]
fn empty_batch_still_checks_preceding_physical_stride() {
    let mut request = request();
    request.input_shape = vec![65536, 65536, 0, 1];
    request.output_shape = vec![65536, 5, 0, 1];
    request.weights_shape = [3, 65536, 5];
    request.padding_left = 1;
    request.padding_right = 1;
    request.stride = 1;
    // The output span is the host geometry's 65536, while the zero batch
    // makes both physical allocation lengths zero. The channel-major prefix
    // still exceeds a u32 address before the batch coordinate is reached.
    assert!(Geometry::new(&request, NumericPrecision::F64)
        .unwrap()
        .is_none());
}

#[test]
fn enormous_valid_batch_is_resource_limited_not_malformed() {
    let mut request = request();
    request.input_shape = vec![7, 3, usize::MAX];
    request.output_shape = vec![4, 5, usize::MAX];
    request.time_axis = None;
    assert!(Geometry::new(&request, NumericPrecision::F64)
        .unwrap()
        .is_none());
    request.output_shape[0] = 5;
    assert!(Geometry::new(&request, NumericPrecision::F64).is_err());
}
