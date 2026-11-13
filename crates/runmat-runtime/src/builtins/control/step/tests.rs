use super::*;
use crate::builtins::control::response::step::{DEFAULT_SAMPLES, MAX_DISCRETE_SAMPLES};
use futures::executor::block_on;
use runmat_builtins::{
    STEP_DESCRIPTOR, STEP_ERROR_DISCRETE_LIMIT, STEP_ERROR_INVALID_MODEL,
    STEP_ERROR_UNSUPPORTED_MODEL,
};
use runmat_value::{
    CharArray, ComplexTensor, IntegerComplexStorage, IntegerStorage, ObjectInstance, Tensor,
};

fn tf_object(num: Vec<f64>, den: Vec<f64>, sample_time: f64) -> Value {
    let mut object = ObjectInstance::new("tf".to_string());
    object.replace_property_unbounded(
        "Numerator".to_string(),
        Value::Tensor(Tensor::new(num.clone(), vec![1, num.len()]).unwrap()),
    );
    object.replace_property_unbounded(
        "Denominator".to_string(),
        Value::Tensor(Tensor::new(den.clone(), vec![1, den.len()]).unwrap()),
    );
    object.replace_property_unbounded(
        "Variable".to_string(),
        Value::CharArray(CharArray::new_row(if sample_time > 0.0 {
            "z"
        } else {
            "s"
        })),
    );
    object.replace_property_unbounded("Ts".to_string(), Value::Num(sample_time));
    object.replace_property_unbounded("InputDelay".into(), Value::Num(0.0));
    object.replace_property_unbounded("OutputDelay".into(), Value::Num(0.0));
    Value::Object(object)
}

fn run_step(sys: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(step_builtin(
        None,
        sys,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

fn tensor_data(value: Value) -> Vec<f64> {
    match value {
        Value::Tensor(tensor) => tensor.materialize_f64(),
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
}

fn poisoned_complex_integer_tensor(
    real: IntegerStorage,
    imag: IntegerStorage,
    shape: Vec<usize>,
) -> ComplexTensor {
    let storage = IntegerComplexStorage::new(real, imag).expect("complex integer storage");

    ComplexTensor::new_integer(storage, shape).expect("complex integer tensor")
}

#[test]
fn step_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = STEP_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"y = step(sys)"));
    assert!(labels.contains(&"y = step(sys, tFinal)"));
    assert!(labels.contains(&"y = step(sys, t)"));
    assert!(labels.contains(&"[y,t] = step(sys)"));
    assert!(labels.contains(&"[y,t] = step(sys, tFinal)"));
    assert!(labels.contains(&"[y,t] = step(sys, t)"));
    for label in [
        "step(sys)",
        "step(sys, tFinal)",
        "step(sys, t)",
        "step(sys1, style1, sys2, style2, tFinal)",
    ] {
        let signature = STEP_DESCRIPTOR
            .signatures
            .iter()
            .find(|signature| signature.label == label)
            .expect("statement plotting signature");
        assert!(signature.outputs.is_empty());
    }
}

#[test]
fn first_order_continuous_response_matches_closed_form_for_explicit_time() {
    let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
    let time = Value::Tensor(Tensor::new(vec![0.0, 0.5, 1.0, 2.0], vec![1, 4]).unwrap());
    let y = tensor_data(run_step(sys, vec![time]).expect("step"));
    for (actual, t) in y.iter().zip([0.0_f64, 0.5, 1.0, 2.0]) {
        let expected = 1.0 - (-t).exp();
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "t={t} actual={actual} expected={expected}"
        );
    }
}

#[test]
fn shared_step_simulation_matches_builtin_and_keeps_stepinfo_error_ownership() {
    use crate::builtins::control::response::step::{admit_time, simulate, StepSimulationCaller};
    use crate::builtins::control::response::{parse_admitted_model, ResponseKind};

    let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
    let parsed = block_on(parse_admitted_model(
        None,
        sys.clone(),
        ResponseKind::StepInfo,
    ))
    .expect("one model admission");
    assert_eq!(parsed.model.dc_gain().expect("dc gain").re, 1.0);
    let time = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
    let time_spec = admit_time(
        Some(&time),
        parsed.real.sample_time(),
        StepSimulationCaller::StepInfo,
    )
    .expect("time admission");
    let shared =
        simulate(&parsed.real, time_spec, StepSimulationCaller::StepInfo).expect("shared response");
    let direct = tensor_data(run_step(sys, vec![time]).expect("step response"));
    assert_eq!(shared.y, direct);
    assert_eq!(shared.t, vec![0.0, 1.0]);

    let bad_time = Value::Num(-1.0);
    let error = admit_time(
        Some(&bad_time),
        parsed.real.sample_time(),
        StepSimulationCaller::StepInfo,
    )
    .expect_err("invalid stepinfo time");
    assert_eq!(error.identifier(), Some("RunMat:stepinfo:InvalidData"));

    let mut delayed = parsed.real.clone();
    delayed.input_delay = 0.25;
    let automatic = admit_time(None, delayed.sample_time(), StepSimulationCaller::StepInfo)
        .expect("automatic time");
    let error = simulate(&delayed, automatic, StepSimulationCaller::StepInfo)
        .expect_err("shared simulation must reject delay");
    assert_eq!(error.identifier(), Some("RunMat:stepinfo:UnsupportedModel"));
}

#[test]
fn multi_output_returns_y_then_time() {
    let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let time = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap());
    let result = run_step(sys, vec![time]).expect("step");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 2);
    assert_eq!(tensor_data(outputs[1].clone()), vec![0.0, 1.0]);
    let y = tensor_data(outputs[0].clone());
    assert!((y[1] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5);
}

#[test]
fn single_requested_output_returns_only_response() {
    let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
    let _guard = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let time = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
    let result = run_step(sys, vec![time]).expect("step");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 1);
    let y = tensor_data(outputs[0].clone());
    assert_eq!(y.len(), 2);
    assert!((y[1] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5);
}

#[test]
fn scalar_final_time_generates_column_time_vector_ending_at_final_time() {
    let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = run_step(sys, vec![Value::Num(5.0)]).expect("step");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    let t = tensor_data(outputs[1].clone());
    assert_eq!(t.len(), DEFAULT_SAMPLES);
    assert_eq!(t[0], 0.0);
    assert!((t[t.len() - 1] - 5.0).abs() < 1.0e-12);
}

#[test]
fn discrete_response_uses_sample_time_grid() {
    let sys = tf_object(vec![1.0], vec![1.0, -0.5], 0.1);
    let time = Value::Tensor(Tensor::new(vec![0.0, 0.1, 0.2], vec![1, 3]).unwrap());
    let y = tensor_data(run_step(sys, vec![time]).expect("step"));
    assert_eq!(y, vec![0.0, 1.0, 1.5]);
}

#[test]
fn step_typed_integer_time_inputs_cross_double_boundary_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = run_step(
        sys,
        vec![integer_tensor(
            IntegerStorage::U64(vec![0, 1, 2]),
            vec![1, 3],
        )],
    )
    .expect("step");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    assert_eq!(tensor_data(outputs[1].clone()), vec![0.0, 1.0, 2.0]);
}

#[test]
fn step_scalar_final_time_reads_typed_integer_storage_length_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
    let final_time =
        Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("final time");
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = run_step(sys, vec![Value::Tensor(final_time)]).expect("step");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    let time = tensor_data(outputs[1].clone());
    assert_eq!(time.first().copied(), Some(0.0));
    assert_eq!(time.last().copied(), Some(2.0));
}

#[test]
fn step_typed_integer_time_is_mode_gated() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
    let error = run_step(
        sys,
        vec![integer_tensor(IntegerStorage::U16(vec![2]), vec![1, 1])],
    )
    .expect_err("MATLAB-compatible mode must reject typed integer time");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:StepIntegerTimeExtension")
    );
}

#[test]
fn step_rejects_wide_integer_time_before_binary64_rounding() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
    let error = run_step(
        sys,
        vec![integer_tensor(
            IntegerStorage::U64(vec![9_007_199_254_740_993]),
            vec![1, 1],
        )],
    )
    .expect_err("inexact binary64 conversion must fail");
    assert!(error.message().contains("exactly representable"));
}

#[test]
fn step_sample_time_parser_reads_native_integer_storage_for_all_classes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];

    for storage in storages {
        let sample_time = Tensor::new_integer(storage, vec![1, 1]).expect("sample time");
        let Value::Object(mut object) = tf_object(vec![1.0], vec![1.0, -0.5], 1.0) else {
            unreachable!()
        };
        object.replace_property_unbounded("Ts".to_string(), Value::Tensor(sample_time));
        let result =
            run_step(Value::Object(object), vec![Value::Num(2.0)]).expect("integer sample time");
        assert_eq!(tensor_data(result), vec![0.0, 1.0, 1.5]);
    }
}

#[test]
fn step_tf_coefficients_read_complex_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = poisoned_complex_integer_tensor(
        IntegerStorage::I16(vec![2, 4]),
        IntegerStorage::I16(vec![0, 0]),
        vec![1, 2],
    );

    let Value::Object(mut object) = tf_object(vec![2.0, 4.0], vec![1.0, 2.0, 1.0], 0.0) else {
        unreachable!()
    };
    object.replace_property_unbounded("Numerator".into(), Value::ComplexTensor(tensor));
    assert!(run_step(Value::Object(object), vec![Value::Num(1.0)]).is_ok());
}

#[test]
fn step_rejects_exact_nonreal_complex_typed_integer_coefficients() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = poisoned_complex_integer_tensor(
        IntegerStorage::I16(vec![2, 4]),
        IntegerStorage::I16(vec![0, 1]),
        vec![1, 2],
    );

    let Value::Object(mut object) = tf_object(vec![2.0, 4.0], vec![1.0, 2.0, 1.0], 0.0) else {
        unreachable!()
    };
    object.replace_property_unbounded("Numerator".into(), Value::ComplexTensor(tensor));
    let err = run_step(Value::Object(object), vec![Value::Num(1.0)]).expect_err("nonreal coeff");
    assert_eq!(err.identifier(), STEP_ERROR_UNSUPPORTED_MODEL.identifier);
    assert!(err.message().contains("complex tf Numerator coefficients"));
}

#[test]
fn discrete_final_time_rejects_excessive_sample_count() {
    let sys = tf_object(vec![1.0], vec![1.0, -0.5], 1.0e-6);
    let err = run_step(sys, vec![Value::Num(2.0)]).expect_err("should fail");
    assert!(err.message().contains("more than 1000000 samples"));
    assert_eq!(err.identifier(), STEP_ERROR_DISCRETE_LIMIT.identifier);
}

#[test]
fn discrete_time_vector_rejects_excessive_sample_index() {
    let sys = tf_object(vec![1.0], vec![1.0, -0.5], 1.0);
    let time =
        Value::Tensor(Tensor::new(vec![0.0, MAX_DISCRETE_SAMPLES as f64], vec![1, 2]).unwrap());
    let err = run_step(sys, vec![time]).expect_err("should fail");
    assert!(err.message().contains("more than 1000000 samples"));
}

#[test]
fn rejects_non_tf_input() {
    let err = run_step(Value::Num(1.0), Vec::new()).expect_err("expected error");
    assert!(err.message().contains("expected a tf object"));
    assert_eq!(err.identifier(), STEP_ERROR_INVALID_MODEL.identifier);
}

#[test]
fn invalid_host_time_and_model_precede_resident_coefficients() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let denominator = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &denominator)
            .expect("resident denominator");
        let Value::Object(mut model) = tf_object(vec![1.0], vec![1.0, 1.0], 0.0) else {
            unreachable!()
        };
        model.replace_property_unbounded("Denominator".into(), Value::GpuTensor(resident.clone()));
        provider.reset_telemetry();
        let error = run_step(Value::Object(model.clone()), vec![Value::Num(-1.0)])
            .expect_err("invalid time");
        assert_eq!(error.identifier(), STEP_ERROR_INVALID_TIME.identifier);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        model.replace_property_unbounded("Numerator".into(), Value::Num(f64::NAN));
        provider.reset_telemetry();
        let error =
            run_step(Value::Object(model), Vec::new()).expect_err("invalid host coefficient");
        assert_eq!(error.identifier(), STEP_ERROR_INVALID_MODEL.identifier);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let Value::Object(mut delayed) = tf_object(vec![1.0], vec![1.0, 1.0], 0.0) else {
            unreachable!()
        };
        delayed.replace_property_unbounded("Denominator".into(), Value::GpuTensor(resident));
        delayed.replace_property_unbounded("InputDelay".into(), Value::Num(1.0e-20));
        provider.reset_telemetry();
        let error = run_step(Value::Object(delayed), Vec::new()).expect_err("nonzero delay");
        assert_eq!(error.identifier(), STEP_ERROR_UNSUPPORTED_MODEL.identifier);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn consumed_resident_coefficient_gathers_but_unrelated_property_does_not() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let denominator = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &denominator)
            .expect("resident denominator");
        let Value::Object(mut model) = tf_object(vec![1.0], vec![1.0, 1.0], 0.0) else {
            unreachable!()
        };
        model.replace_property_unbounded("Unrelated".into(), Value::GpuTensor(resident.clone()));
        provider.reset_telemetry();
        run_step(Value::Object(model.clone()), Vec::new()).expect("host model");
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        model.replace_property_unbounded("Denominator".into(), Value::GpuTensor(resident));
        provider.reset_telemetry();
        run_step(Value::Object(model), Vec::new()).expect("resident coefficient");
        assert!(provider.telemetry_snapshot().download_bytes > 0);
    });
}

#[test]
fn scaled_nonzero_coefficients_preserve_model_order_and_response() {
    let model = tf_object(vec![1.0e-20], vec![1.0e-20, 1.0e-20], 0.0);
    let y = tensor_data(
        run_step(
            model,
            vec![Value::Tensor(
                Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap(),
            )],
        )
        .expect("scaled model"),
    );
    assert_eq!(y[0], 0.0);
    assert!((y[1] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5);
}

#[test]
fn tiny_nonzero_imaginary_coefficient_is_not_silently_real() {
    let Value::Object(mut model) = tf_object(vec![1.0], vec![1.0, 1.0], 0.0) else {
        unreachable!()
    };
    model.replace_property_unbounded(
        "Numerator".into(),
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 1.0e-20)], vec![1, 1]).unwrap()),
    );
    let error = run_step(Value::Object(model), Vec::new()).expect_err("complex model");
    assert_eq!(error.identifier(), STEP_ERROR_UNSUPPORTED_MODEL.identifier);
}

#[cfg(feature = "plot-core")]
#[test]
fn invalid_statement_style_is_rejected_before_model_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let denominator = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &denominator)
            .expect("resident denominator");
        let Value::Object(mut model) = tf_object(vec![1.0], vec![1.0, 1.0], 0.0) else {
            unreachable!()
        };
        model.replace_property_unbounded("Denominator".into(), Value::GpuTensor(resident));
        let _output = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        provider.reset_telemetry();
        let error = run_step(
            Value::Object(model),
            vec![Value::from("definitely-not-a-line-style")],
        )
        .expect_err("invalid style");
        assert_eq!(error.identifier(), STEP_ERROR_INVALID_ARGUMENT.identifier);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn exactly_representable_wide_integer_coefficients_are_admitted() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let exact = 1_u64 << 63;
    let Value::Object(mut model) = tf_object(vec![1.0], vec![1.0, 1.0], 0.0) else {
        unreachable!()
    };
    model.replace_property_unbounded(
        "Numerator".into(),
        integer_tensor(IntegerStorage::U64(vec![exact]), vec![1, 1]),
    );
    model.replace_property_unbounded(
        "Denominator".into(),
        integer_tensor(IntegerStorage::U64(vec![exact, exact]), vec![1, 2]),
    );
    let time = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
    let y = tensor_data(
        run_step(Value::Object(model.clone()), vec![time]).expect("exact wide integer"),
    );
    assert!((y[1] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5);

    model.replace_property_unbounded(
        "Numerator".into(),
        integer_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
    );
    let error = run_step(Value::Object(model), Vec::new()).expect_err("inexact wide integer");
    assert_eq!(error.identifier(), STEP_ERROR_INVALID_MODEL.identifier);
    assert!(error.message().contains("exactly representable"));
}

#[test]
fn tiny_nonreal_host_coefficient_precedes_resident_sample_time_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let ts = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &ts)
            .expect("resident sample time");
        let Value::Object(mut model) = tf_object(vec![1.0], vec![1.0, 1.0], 0.0) else {
            unreachable!()
        };
        model.replace_property_unbounded("Ts".into(), Value::GpuTensor(resident));
        model.replace_property_unbounded(
            "Numerator".into(),
            Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 1.0e-20)], vec![1, 1]).unwrap()),
        );
        provider.reset_telemetry();
        let error =
            run_step(Value::Object(model), Vec::new()).expect_err("nonreal host coefficient");
        assert_eq!(error.identifier(), STEP_ERROR_UNSUPPORTED_MODEL.identifier);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn resident_integer_time_is_downloaded_once_after_compatibility_admission() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let time = Tensor::new_integer(IntegerStorage::U64(vec![0, 1]), vec![1, 2]).unwrap();
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &time)
            .expect("resident time");
        provider.reset_telemetry();
        let y = tensor_data(
            run_step(
                tf_object(vec![1.0], vec![1.0, 1.0], 0.0),
                vec![Value::GpuTensor(resident)],
            )
            .expect("step"),
        );
        assert!((y[1] - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-5);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 16);
    });
}

#[test]
fn normalization_overflow_is_an_error_not_a_nonfinite_response() {
    let model = tf_object(vec![1.0e308], vec![1.0e-320, 1.0e-320], 0.0);
    let error =
        run_step(model, vec![Value::Num(1.0)]).expect_err("nonfinite normalized coefficient");
    assert_eq!(error.identifier(), STEP_ERROR_INVALID_MODEL.identifier);
}
