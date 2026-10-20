use super::*;

#[test]
fn volume_flow_constant_is_portable_typed_and_scalar_finite() {
    let value = block_on(volume_flow_constant("flow".into(), Value::Num(-0.25))).unwrap();
    let PhysicalSignal::VolumeFlowRate(signal) =
        physical_signal_from_value("test", &value).unwrap()
    else {
        panic!("expected generic volume-flow signal");
    };
    assert_eq!(signal.shape, SignalShape::Scalar);
    assert_eq!(
        signal.representation,
        SignalRepresentation::Constant(SignalValue::Scalar(VolumeFlowRate::new(-0.25)))
    );
    assert_eq!(
        crate::execution::value_codec::decode_inline_value(
            &crate::execution::value_codec::encode_inline_value(&value).unwrap()
        )
        .unwrap(),
        value
    );
    for bad in [
        Value::Num(f64::NAN),
        Value::Num(f64::INFINITY),
        Value::Tensor(Tensor::new_2d(vec![1.0, 2.0], 1, 2).unwrap()),
    ] {
        assert!(block_on(volume_flow_constant("bad".into(), bad)).is_err());
    }
    let Value::Object(mut object) = value else {
        unreachable!()
    };
    object = crate::builtins::common::test_support::reclassify_object_unbounded(
        &object,
        MASS_FLOW_SIGNAL_CLASS,
    );
    assert!(physical_signal_from_value("test", &Value::Object(object)).is_err());
}
