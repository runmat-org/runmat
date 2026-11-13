//! Regression tests for typed observation constructors: one-based mode
//! selection, absent choices, codec schemas, finite path positions, and the
//! row orientation of vectors inside typed objects.

use futures::executor::block_on;
use runmat_value::Tensor;

use super::*;
use crate::execution::value_codec::{decode_inline_value, encode_inline_value};

fn row(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new_2d(values.to_vec(), 1, values.len()).expect("row tensor"))
}

fn sampling_payload(value: &Value) -> SamplingPolicy {
    from_typed_value("test", SAMPLING_CLASS, value).expect("typed sampling policy")
}

fn codec_round_trip(value: &Value) -> Value {
    let payload = encode_inline_value(value).expect("immutable value encodes");
    decode_inline_value(&payload).expect("immutable value decodes")
}

#[test]
fn mode_indices_keep_the_public_one_based_selection() {
    let policy = block_on(sampling::sampling_modes(row(&[1.0, 2.0, 3.0]))).unwrap();
    assert_eq!(
        sampling_payload(&policy),
        SamplingPolicy::ModeIndices(vec![1, 2, 3])
    );
    let error = block_on(sampling::sampling_modes(row(&[0.0, 1.0]))).unwrap_err();
    assert!(error.message.contains("one-based"));
}

#[test]
fn absent_choices_expose_an_empty_value_and_encode() {
    for absent in [
        block_on(storage::reduction_none()).unwrap(),
        block_on(requests::stop_none()).unwrap(),
    ] {
        let Value::Object(object) = &absent else {
            panic!("typed object")
        };
        assert!(matches!(
            object.properties().get("value"),
            Some(Value::Tensor(tensor)) if tensor.is_empty()
        ));
        let decoded = codec_round_trip(&absent);
        let Value::Object(decoded) = decoded else {
            panic!("decoded object")
        };
        assert_eq!(decoded.class_name(), object.class_name());
    }
    let reduction = block_on(storage::reduction_none()).unwrap();
    assert_eq!(
        optional_typed_value::<ReductionPolicy>("test", REDUCTION_CLASS, &reduction).unwrap(),
        None
    );
}

#[test]
fn monitor_with_progress_only_condition_decodes_without_a_stop() {
    let cadence = block_on(requests::cadence_accepted(Value::Num(5.0))).unwrap();
    let stop = block_on(requests::stop_none()).unwrap();
    let monitor = block_on(requests::monitor(
        "progress".into(),
        "tip_deflection".into(),
        cadence,
        stop,
    ))
    .unwrap();
    let request: ObservationRequest =
        from_typed_value("test", OBSERVATION_CLASS, &monitor).unwrap();
    let ObservationRequest::Monitor(monitor) = request else {
        panic!("monitor request")
    };
    assert_eq!(monitor.stop, None);
}

#[test]
fn observation_cost_encodes_through_its_registered_schema() {
    let cost = block_on(storage::cost(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(25000.0),
            Value::Num(400000.0),
            Value::Num(24000.0),
            Value::Num(24000.0),
            Value::Num(0.0),
        ]),
    ))
    .unwrap();
    let decoded = codec_round_trip(&cost);
    let policy: runmat_analysis_core::ObservationCostPolicy =
        object_payload("test", &decoded, COST_CLASS).unwrap();
    assert_eq!(
        policy.upper_bound().map(|limit| limit.stored_bytes),
        Some(24000)
    );
}

#[test]
fn path_targets_reject_nonfinite_coordinates() {
    let positions = Value::Tensor(
        Tensor::new_2d(vec![0.0, 0.1, 0.0, 0.0, f64::NAN, 0.0], 2, 3).expect("positions"),
    );
    let error = block_on(targets::target_path(positions)).unwrap_err();
    assert!(error.message.contains("finite"));
}

#[test]
fn typed_vectors_are_rows_and_still_decode() {
    let policy = block_on(sampling::sampling_time(row(&[0.0, 0.01, 0.02]))).unwrap();
    let Value::Object(object) = &policy else {
        panic!("typed object")
    };
    let Some(Value::Struct(value)) = object.properties().get("value") else {
        panic!("typed payload struct")
    };
    let Some(Value::Tensor(times)) = value.fields.get("time_seconds") else {
        panic!("time samples")
    };
    assert_eq!(times.shape(), [1, 3]);
    assert_eq!(
        sampling_payload(&policy),
        SamplingPolicy::TimeSeconds(vec![0.0, 0.01, 0.02])
    );
    let single = block_on(sampling::sampling_time(Value::Num(0.5))).unwrap();
    assert_eq!(
        sampling_payload(&single),
        SamplingPolicy::TimeSeconds(vec![0.5])
    );
}
