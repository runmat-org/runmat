use super::*;
use futures::executor::block_on;
use runmat_value::{CharArray, IntegerStorage, ObjectInstance, Tensor};

fn run(input: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(stepinfo_builtin(
        None,
        input,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}
fn samples(values: Vec<f64>) -> Value {
    Value::Tensor(Tensor::new(values.clone(), vec![1, values.len()]).unwrap())
}
fn tf_model(numerator: Value, denominator: Value) -> Value {
    let mut object = ObjectInstance::new("tf".to_string());
    object.replace_property_unbounded("Numerator".into(), numerator);
    object.replace_property_unbounded("Denominator".into(), denominator);
    object.replace_property_unbounded("Variable".into(), Value::CharArray(CharArray::new_row("s")));
    object.replace_property_unbounded("Ts".into(), Value::Num(0.0));
    object.replace_property_unbounded("InputDelay".into(), Value::Num(0.0));
    object.replace_property_unbounded("OutputDelay".into(), Value::Num(0.0));
    Value::Object(object)
}
fn field(value: &Value, name: &str) -> f64 {
    let Value::Struct(info) = value else {
        panic!("expected struct")
    };
    let Some(Value::Num(number)) = info.fields.get(name) else {
        panic!("expected numeric field {name}")
    };
    *number
}

#[test]
fn sampled_forms_return_metrics_and_threshold_changes_settling_time() {
    let y = samples(vec![0.0, 0.5, 0.9, 1.0]);
    let t = samples(vec![0.0, 1.0, 2.0, 3.0]);
    let info = run(y.clone(), vec![t.clone(), Value::Num(1.0)]).unwrap();
    assert!((field(&info, "RiseTime") - 1.8).abs() < 1e-12);
    assert_eq!(field(&info, "SteadyStateValue"), 1.0);
    assert_eq!(field(&info, "Overshoot"), 0.0);
    let changed = run(
        y,
        vec![
            t,
            Value::Num(1.0),
            Value::from("SettlingTimeThreshold"),
            Value::Num(0.1),
        ],
    )
    .unwrap();
    assert_eq!(field(&changed, "SettlingTime"), 2.0);
    let padded = run(
        samples(vec![0.0, 0.5, 0.9, 1.0]),
        vec![
            samples(vec![0.0, 1.0, 2.0, 3.0]),
            Value::Num(1.0),
            Value::from(" SettlingTimeThreshold "),
            Value::Num(0.1),
        ],
    )
    .unwrap();
    assert_eq!(field(&padded, "SettlingTime"), 2.0);
}

#[test]
fn model_form_uses_typed_pure_step_service_and_dc_gain() {
    let model = tf_model(Value::Num(1.0), samples(vec![1.0, 1.0]));
    let info = run(model, vec![]).unwrap();
    assert!((field(&info, "SteadyStateValue") - 1.0).abs() < 1e-12);
    assert_eq!(field(&info, "Overshoot"), 0.0);
}

#[test]
fn typed_integer_roles_are_independently_gated_and_exact() {
    let y =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 1]), vec![1, 3]).unwrap());
    let _mode = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = run(y.clone(), vec![]).expect_err("typed sampled data gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:StepinfoIntegerSampledDataExtension")
    );
    let threshold =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).unwrap());
    let error = run(
        samples(vec![0.0, 1.0]),
        vec![Value::from("SettlingTimeThreshold"), threshold],
    )
    .expect_err("typed threshold gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:StepinfoIntegerThresholdExtension")
    );
    drop(_mode);
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    let info = run(y, vec![]).unwrap();
    assert_eq!(field(&info, "SteadyStateValue"), 1.0);
    let inexact = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![0, (1_u64 << 53) + 1]), vec![1, 2]).unwrap(),
    );
    let error = run(inexact, vec![]).expect_err("inexact binary64 conversion");
    assert!(error.message().contains("exactly representable"));
}

#[test]
fn host_options_and_length_fail_before_provider_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap(),
        )
        .unwrap();
        provider.reset_telemetry();
        let error = run(
            Value::GpuTensor(resident.clone()),
            vec![Value::from("UnsupportedOption"), Value::Num(0.5)],
        )
        .expect_err("known invalid option");
        assert!(error.message().contains("unsupported option"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let error = run(
            samples(vec![0.0, 1.0, 2.0]),
            vec![samples(vec![0.0, 1.0]), Value::GpuTensor(resident)],
        )
        .expect_err("known length mismatch");
        assert!(error.message().contains("same number of samples"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn requested_output_count_is_checked_before_admission() {
    let _count = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = run(samples(vec![0.0, 1.0]), vec![]).expect_err("one output only");
    assert_eq!(error.identifier(), Some("RunMat:stepinfo:InvalidArgument"));
}

#[test]
fn model_nested_residency_checks_host_options_before_owner_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let numerator = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let resident =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &numerator).unwrap();
        let model = tf_model(Value::GpuTensor(resident), samples(vec![1.0, 1.0]));
        provider.reset_telemetry();
        let error = run(
            model.clone(),
            vec![Value::from("SettlingTimeThreshold"), Value::Num(-0.1)],
        )
        .expect_err("invalid host option");
        assert!(error.message().contains("SettlingTimeThreshold"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let info = run(model, vec![]).expect("admitted nested resident model");
        assert!((field(&info, "SteadyStateValue") - 1.0).abs() < 1e-12);
        assert_eq!(
            provider.telemetry_snapshot().download_bytes,
            u64::try_from(numerator.len() * std::mem::size_of::<f64>()).unwrap()
        );
    });
}
