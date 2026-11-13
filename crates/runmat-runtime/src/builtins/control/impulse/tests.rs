use super::*;
use futures::executor::block_on;
use runmat_builtins::IMPULSE_DESCRIPTOR;
use runmat_value::{CharArray, IntValue, IntegerStorage, ObjectInstance, Tensor};

fn tf_object(num: Vec<f64>, den: Vec<f64>, ts: f64) -> Value {
    tf_object_with_delays(num, den, ts, 0.0, 0.0)
}

fn tf_object_with_delays(
    num: Vec<f64>,
    den: Vec<f64>,
    ts: f64,
    input_delay: f64,
    output_delay: f64,
) -> Value {
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
        Value::CharArray(CharArray::new_row(if ts > 0.0 { "z" } else { "s" })),
    );
    object.replace_property_unbounded("Ts".to_string(), Value::Num(ts));
    object.replace_property_unbounded("InputDelay".to_string(), Value::Num(input_delay));
    object.replace_property_unbounded("OutputDelay".to_string(), Value::Num(output_delay));
    Value::Object(object)
}

fn run_impulse(system: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(impulse_builtin(
        None,
        system,
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
    let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
    Value::Tensor(tensor)
}

#[test]
fn impulse_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = IMPULSE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"y = impulse(sys)"));
    assert!(labels.contains(&"y = impulse(sys, tFinal)"));
    assert!(labels.contains(&"y = impulse(sys, t)"));
    assert!(labels.contains(&"[y,t] = impulse(sys)"));
    assert!(labels.contains(&"[y,t] = impulse(sys, tFinal)"));
    assert!(labels.contains(&"[y,t] = impulse(sys, t)"));
    for label in ["impulse(sys)", "impulse(sys, tFinal)", "impulse(sys, t)"] {
        let signature = IMPULSE_DESCRIPTOR
            .signatures
            .iter()
            .find(|signature| signature.label == label)
            .expect("statement plotting signature");
        assert!(signature.outputs.is_empty());
    }
}

#[test]
fn fatal_plot_error_has_impulse_identifier_and_retains_source() {
    use std::error::Error;

    let source = build_runtime_error("plot: rendering failed")
        .with_builtin("plot")
        .with_identifier("RunMat:plot:RenderFailed")
        .build();
    let wrapped = impulse_plot_error(source);
    assert_eq!(wrapped.identifier(), Some("RunMat:impulse:PlotFailed"));
    assert!(wrapped.message().contains("rendering failed"));
    assert!(wrapped.source().is_some());
}

#[test]
fn impulse_first_order_continuous_explicit_time() {
    let sys = tf_object(vec![20.0], vec![1.0, 5.0], 0.0);
    let t = Value::Tensor(Tensor::new(vec![0.0, 0.1, 0.2], vec![1, 3]).unwrap());
    let y = tensor_data(run_impulse(sys, vec![t]).expect("impulse"));
    let expected = [20.0, 20.0 * (-0.5f64).exp(), 20.0 * (-1.0f64).exp()];
    for (actual, expected) in y.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-8);
    }
}

#[test]
fn impulse_second_order_continuous() {
    let sys = tf_object(vec![1.0], vec![1.0, 3.0, 2.0], 0.0);
    let t = Value::Tensor(Tensor::new(vec![0.0, 0.5, 1.0], vec![1, 3]).unwrap());
    let y = tensor_data(run_impulse(sys, vec![t]).expect("impulse"));
    for (actual, time) in y.iter().zip([0.0_f64, 0.5, 1.0]) {
        let expected = (-time).exp() - (-2.0 * time).exp();
        assert!((actual - expected).abs() < 1.0e-8);
    }
}

#[test]
fn impulse_multi_output_returns_y_and_t_columns() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let sys = tf_object(vec![20.0], vec![1.0, 5.0], 0.0);
    let t_arg = Value::Tensor(Tensor::new(vec![0.0, 0.1], vec![1, 2]).unwrap());
    let result = run_impulse(sys, vec![t_arg]).expect("impulse");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 2);
    match &outputs[0] {
        Value::Tensor(tensor) => assert_eq!(tensor.shape(), vec![2, 1]),
        other => panic!("expected y tensor, got {other:?}"),
    }
    match &outputs[1] {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), vec![2, 1]);
            assert_eq!(tensor.materialize_f64(), vec![0.0, 0.1]);
        }
        other => panic!("expected t tensor, got {other:?}"),
    }
}

#[cfg(feature = "plot-core")]
#[test]
fn impulse_zero_output_count_emits_no_values() {
    let _guard = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
    let result = run_impulse(sys, Vec::new()).expect("impulse");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    assert!(outputs.is_empty());
}

#[cfg(feature = "plot-core")]
#[test]
fn impulse_requested_zero_outputs_emits_no_values() {
    let _guard = crate::output_context::push_output_count(0).expect("output-count scope");
    let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
    let result = run_impulse(sys, Vec::new()).expect("impulse");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    assert!(outputs.is_empty());
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn impulse_numeric_output_survives_without_plotting_but_statement_fails_typed() {
    use std::error::Error;

    let sys = tf_object(vec![1.0], vec![1.0, 1.0], 0.0);
    let _numeric = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    assert!(matches!(
        run_impulse(sys.clone(), Vec::new()).expect("numeric response"),
        Value::OutputList(_)
    ));
    drop(_numeric);

    let _statement = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = run_impulse(sys, Vec::new()).expect_err("presentation capability absent");
    assert_eq!(error.identifier(), IMPULSE_ERROR_PLOT_FAILED.identifier);
    assert!(error
        .source()
        .and_then(|source| source.source())
        .and_then(|source| source.downcast_ref::<crate::plotting_hooks::PlottingUnavailable>())
        .is_some());
}

#[test]
fn impulse_discrete_siso_response() {
    let sys = tf_object(vec![1.0], vec![1.0, -0.5], 0.1);
    let t = Value::Tensor(Tensor::new(vec![0.0, 0.1, 0.2, 0.3], vec![1, 4]).unwrap());
    let y = tensor_data(run_impulse(sys, vec![t]).expect("impulse"));
    assert_eq!(y.len(), 4);
    assert!((y[0] - 0.0).abs() < 1.0e-12);
    assert!((y[1] - 10.0).abs() < 1.0e-12);
    assert!((y[2] - 5.0).abs() < 1.0e-12);
    assert!((y[3] - 2.5).abs() < 1.0e-12);
}

#[test]
fn impulse_typed_integer_coefficients_and_time_cross_double_boundary_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let mut object = ObjectInstance::new("tf".to_string());
    object.replace_property_unbounded(
        "Numerator".to_string(),
        integer_tensor(IntegerStorage::U16(vec![20]), vec![1, 1]),
    );
    object.replace_property_unbounded(
        "Denominator".to_string(),
        integer_tensor(IntegerStorage::I16(vec![1, 5]), vec![1, 2]),
    );
    object.replace_property_unbounded(
        "Variable".to_string(),
        Value::CharArray(CharArray::new_row("s")),
    );
    object.replace_property_unbounded("Ts".to_string(), Value::Num(0.0));
    object.replace_property_unbounded("InputDelay".to_string(), Value::Num(0.0));
    object.replace_property_unbounded("OutputDelay".to_string(), Value::Num(0.0));

    let t = integer_tensor(IntegerStorage::U64(vec![0, 1]), vec![1, 2]);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = run_impulse(Value::Object(object), vec![t]).expect("impulse");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    assert_eq!(tensor_data(outputs[1].clone()), vec![0.0, 1.0]);
}

#[test]
fn impulse_time_vector_parser_ignores_poisoned_integer_mirrors_for_all_classes() {
    let storages = [
        IntegerStorage::I8(vec![0, 1]),
        IntegerStorage::I16(vec![0, 1]),
        IntegerStorage::I32(vec![0, 1]),
        IntegerStorage::I64(vec![0, 1]),
        IntegerStorage::U8(vec![0, 1]),
        IntegerStorage::U16(vec![0, 1]),
        IntegerStorage::U32(vec![0, 1]),
        IntegerStorage::U64(vec![0, 1]),
    ];

    for storage in storages {
        assert_eq!(
            time_vector_from_value(integer_tensor(storage, vec![1, 2]))
                .expect("typed integer time vector"),
            vec![0.0, 1.0]
        );
    }
}

#[test]
fn impulse_scalar_final_time_reads_typed_integer_storage_length_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let mut object = ObjectInstance::new("tf".to_string());
    object.replace_property_unbounded(
        "Numerator".to_string(),
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
    );
    object.replace_property_unbounded(
        "Denominator".to_string(),
        Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
    );
    object.replace_property_unbounded(
        "Variable".to_string(),
        Value::CharArray(CharArray::new_row("s")),
    );
    object.replace_property_unbounded("Ts".to_string(), Value::Num(0.0));
    object.replace_property_unbounded("InputDelay".to_string(), Value::Num(0.0));
    object.replace_property_unbounded("OutputDelay".to_string(), Value::Num(0.0));

    let final_time =
        Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("final time");
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result =
        run_impulse(Value::Object(object), vec![Value::Tensor(final_time)]).expect("impulse");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    let time = tensor_data(outputs[1].clone());
    assert_eq!(time.first().copied(), Some(0.0));
    assert_eq!(time.last().copied(), Some(2.0));
}

#[test]
fn impulse_discrete_final_time_rejects_excessive_sample_count() {
    let sys = tf_object(vec![1.0], vec![1.0, -0.5], 1.0e-6);
    let err = run_impulse(sys, vec![Value::Num(2.0)]).expect_err("should fail");
    assert!(err.message().contains("more than 1000000 samples"));
    assert_eq!(err.identifier(), IMPULSE_ERROR_DISCRETE_LIMIT.identifier);
}

#[test]
fn impulse_discrete_time_vector_rejects_excessive_sample_index() {
    let sys = tf_object(vec![1.0], vec![1.0, -0.5], 1.0);
    let t = Value::Tensor(Tensor::new(vec![0.0, MAX_DISCRETE_SAMPLES as f64], vec![1, 2]).unwrap());
    let err = run_impulse(sys, vec![t]).expect_err("should fail");
    assert!(err.message().contains("more than 1000000 samples"));
}

#[test]
fn impulse_rejects_unsupported_model_type() {
    let object = ObjectInstance::new("ss".to_string());
    let err = run_impulse(Value::Object(object), Vec::new()).expect_err("should fail");
    assert!(err.message().contains("unsupported model class"));
    assert_eq!(err.identifier(), IMPULSE_ERROR_UNSUPPORTED_MODEL.identifier);
}

#[test]
fn impulse_rejects_direct_feedthrough_tf() {
    let sys = tf_object(vec![1.0, 1.0], vec![1.0, 2.0], 0.0);
    let err = run_impulse(sys, Vec::new()).expect_err("should fail");
    assert!(err.message().contains("strictly proper"));
}

#[test]
fn impulse_rejects_invalid_time_metadata() {
    let err = run_impulse(tf_object(vec![1.0], vec![1.0, -0.5], -0.1), Vec::new())
        .expect_err("negative sample time should fail");
    assert!(err.message().contains("sample time must be"), "{err}");
    assert_eq!(err.identifier(), IMPULSE_ERROR_INVALID_MODEL.identifier);

    let err = run_impulse(
        tf_object_with_delays(vec![1.0], vec![1.0, 5.0], 0.0, f64::NAN, 0.0),
        Vec::new(),
    )
    .expect_err("NaN input delay should fail");
    assert!(err.message().contains("InputDelay must be"));

    let err = run_impulse(
        tf_object_with_delays(vec![1.0], vec![1.0, 5.0], 0.0, 0.0, -1.0),
        Vec::new(),
    )
    .expect_err("negative output delay should fail");
    assert!(err.message().contains("OutputDelay must be"));
}

#[test]
fn impulse_runmat_numeric_extensions_are_gated_and_exact() {
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = run_impulse(
            tf_object(vec![1.0], vec![1.0, 1.0], 0.0),
            vec![Value::Int(IntValue::U16(2))],
        )
        .unwrap_err();
        assert_eq!(
            integer.identifier(),
            Some("RunMat:compatibility:ImpulseIntegerExtension")
        );
        let single = run_impulse(
            tf_object(vec![1.0], vec![1.0, 1.0], 0.0),
            vec![Value::Tensor(
                Tensor::from_f32(vec![0.0, 1.0], vec![1, 2]).unwrap(),
            )],
        )
        .unwrap_err();
        assert_eq!(
            single.identifier(),
            Some("RunMat:compatibility:ImpulseSingleExtension")
        );
    }

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let wide = run_impulse(
        tf_object(vec![1.0], vec![1.0, 1.0], 0.0),
        vec![Value::Int(IntValue::U64((1_u64 << 53) + 1))],
    )
    .unwrap_err();
    assert!(wide.message().contains("exactly representable as double"));
}

#[test]
fn impulse_resident_time_is_an_explicit_host_output_extension() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
            .expect("resident time");
        runmat_accelerate::fusion_residency::mark(&handle);
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = run_impulse(
                tf_object(vec![1.0], vec![1.0, 1.0], 0.0),
                vec![Value::GpuTensor(handle.clone())],
            )
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:ImpulseResidentExtension")
            );
        }
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let output = run_impulse(
            tf_object(vec![1.0], vec![1.0, 1.0], 0.0),
            vec![Value::GpuTensor(handle.clone())],
        )
        .expect("resident host fallback");
        assert!(matches!(output, Value::Tensor(_)));
        assert!(runmat_accelerate::fusion_residency::is_resident(&handle));
        handle.allocation_lease().validate_live().unwrap();
        let physical = handle.allocation_lease().downgrade();
        handle.release().unwrap();
        assert!(physical.acquire().is_none());
    });
}

#[test]
fn nested_resident_coefficients_are_compatibility_gated_before_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let coefficient = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &coefficient)
            .expect("resident denominator");
        let Value::Object(mut model) = tf_object(vec![1.0], vec![1.0, 1.0], 0.0) else {
            unreachable!()
        };
        model.replace_property_unbounded("Denominator".into(), Value::GpuTensor(resident.clone()));
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        provider.reset_telemetry();
        let error = run_impulse(Value::Object(model), Vec::new()).expect_err("MATLAB mode");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:ImpulseResidentExtension")
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn invalid_host_time_or_model_precedes_resident_coefficients() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let coefficient = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &coefficient)
            .expect("resident denominator");
        let Value::Object(mut model) = tf_object(vec![1.0], vec![1.0, 1.0], 0.0) else {
            unreachable!()
        };
        model.replace_property_unbounded("Denominator".into(), Value::GpuTensor(resident.clone()));
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        provider.reset_telemetry();
        let error = run_impulse(Value::Object(model.clone()), vec![Value::Num(-1.0)])
            .expect_err("invalid time");
        assert_eq!(error.identifier(), IMPULSE_ERROR_INVALID_TIME.identifier);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        model.replace_property_unbounded("Numerator".into(), Value::Num(f64::NAN));
        provider.reset_telemetry();
        let error = run_impulse(Value::Object(model), Vec::new()).expect_err("invalid model");
        assert_eq!(error.identifier(), IMPULSE_ERROR_INVALID_MODEL.identifier);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let Value::Object(mut delayed) = tf_object(vec![1.0], vec![1.0, 1.0], 0.0) else {
            unreachable!()
        };
        delayed.replace_property_unbounded("Denominator".into(), Value::GpuTensor(resident));
        delayed.replace_property_unbounded("OutputDelay".into(), Value::Num(1.0e-20));
        provider.reset_telemetry();
        let error = run_impulse(Value::Object(delayed), Vec::new()).expect_err("nonzero delay");
        assert_eq!(
            error.identifier(),
            IMPULSE_ERROR_UNSUPPORTED_MODEL.identifier
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn scaled_nonzero_coefficients_keep_first_order_impulse_response() {
    let model = tf_object(vec![1.0e-20], vec![1.0e-20, 1.0e-20], 0.0);
    let times = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
    let y = tensor_data(run_impulse(model, vec![times]).expect("scaled model"));
    assert!((y[0] - 1.0).abs() < 1.0e-12);
    assert!((y[1] - (-1.0_f64).exp()).abs() < 1.0e-8);
}

#[test]
fn tiny_nonzero_imaginary_coefficient_is_rejected() {
    let Value::Object(mut model) = tf_object(vec![1.0], vec![1.0, 1.0], 0.0) else {
        unreachable!()
    };
    model.replace_property_unbounded(
        "Numerator".into(),
        Value::ComplexTensor(
            runmat_value::ComplexTensor::new(vec![(1.0, 1.0e-20)], vec![1, 1]).unwrap(),
        ),
    );
    let error = run_impulse(Value::Object(model), Vec::new()).expect_err("complex model");
    assert_eq!(
        error.identifier(),
        IMPULSE_ERROR_UNSUPPORTED_MODEL.identifier
    );
}

#[test]
fn resident_wide_integer_coefficient_uses_exact_binary64_boundary() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let exact =
            Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63]), vec![1, 1]).unwrap();
        let exact_handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &exact)
            .expect("exact resident integer");
        let Value::Object(mut model) = tf_object(vec![1.0], vec![1.0, 1.0], 0.0) else {
            unreachable!()
        };
        model.replace_property_unbounded("Numerator".into(), Value::GpuTensor(exact_handle));
        model.replace_property_unbounded(
            "Denominator".into(),
            integer_tensor(
                IntegerStorage::U64(vec![1_u64 << 63, 1_u64 << 63]),
                vec![1, 2],
            ),
        );
        let time = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
        provider.reset_telemetry();
        let y = tensor_data(
            run_impulse(Value::Object(model.clone()), vec![time]).expect("exact resident integer"),
        );
        assert!((y[1] - (-1.0_f64).exp()).abs() < 1.0e-8);
        assert!(provider.telemetry_snapshot().download_bytes > 0);

        let inexact =
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap();
        let inexact_handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &inexact)
                .expect("inexact resident integer");
        model.replace_property_unbounded("Numerator".into(), Value::GpuTensor(inexact_handle));
        let error =
            run_impulse(Value::Object(model), Vec::new()).expect_err("inexact resident integer");
        assert_eq!(
            error.identifier(),
            IMPULSE_ERROR_INVALID_ARGUMENT.identifier
        );
        assert!(error.message().contains("exactly representable"));
    });
}

#[test]
fn wide_exact_integer_time_is_accepted_but_inexact_time_is_rejected() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let exact = 1_u64 << 63;
    let model = tf_object(vec![1.0], vec![1.0, -0.5], exact as f64);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let Value::OutputList(values) =
        run_impulse(model.clone(), vec![Value::Int(IntValue::U64(exact))])
            .expect("exact wide final time")
    else {
        panic!("outputs")
    };
    assert_eq!(tensor_data(values[1].clone()), vec![0.0, exact as f64]);
    let error = run_impulse(
        model.clone(),
        vec![Value::Int(IntValue::U64((1_u64 << 53) + 1))],
    )
    .expect_err("inexact final time");
    assert_eq!(
        error.identifier(),
        IMPULSE_ERROR_INVALID_ARGUMENT.identifier
    );

    assert_eq!(
        time_vector_from_value(integer_tensor(
            IntegerStorage::U64(vec![0, exact]),
            vec![1, 2]
        ))
        .expect("exact vector"),
        vec![0.0, exact as f64]
    );
    assert!(time_vector_from_value(integer_tensor(
        IntegerStorage::U64(vec![0, (1_u64 << 53) + 1]),
        vec![1, 2]
    ))
    .is_err());

    crate::builtins::common::test_support::with_test_provider(|provider| {
        let exact_tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![exact]), vec![1, 1]).unwrap();
        let exact_handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &exact_tensor)
                .expect("resident exact time");
        provider.reset_telemetry();
        let Value::OutputList(values) =
            run_impulse(model.clone(), vec![Value::GpuTensor(exact_handle)])
                .expect("resident exact time")
        else {
            panic!("outputs")
        };
        assert_eq!(tensor_data(values[1].clone()), vec![0.0, exact as f64]);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 8);

        let inexact_tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap();
        let inexact_handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &inexact_tensor)
                .expect("resident inexact time");
        let error = run_impulse(model, vec![Value::GpuTensor(inexact_handle)])
            .expect_err("resident inexact time");
        assert_eq!(
            error.identifier(),
            IMPULSE_ERROR_INVALID_ARGUMENT.identifier
        );
    });
}

#[test]
fn normalization_overflow_is_an_error_not_a_nonfinite_response() {
    let model = tf_object(vec![1.0e308], vec![1.0e-320, 1.0e-320], 0.0);
    let error = run_impulse(model, Vec::new()).expect_err("nonfinite normalized coefficient");
    assert_eq!(error.identifier(), IMPULSE_ERROR_INVALID_MODEL.identifier);
}
