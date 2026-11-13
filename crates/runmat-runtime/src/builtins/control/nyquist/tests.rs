use super::*;
use futures::executor::block_on;
use runmat_value::{CharArray, ComplexTensor, IntegerStorage, ObjectInstance};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn frequency_admission_accepts_low_rank_vectors_but_rejects_matrices() {
    for (shape, values) in [(Vec::new(), vec![1.0]), (vec![2], vec![1.0, 2.0])] {
        let value = Value::Tensor(Tensor::new(values, shape).unwrap());
        frequency::validate_host(&value).unwrap();
    }
    let matrix = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
    assert!(frequency::validate_host(&matrix).is_err());
}

fn raw_tf(numerator: Value, denominator: Value, sample_time: f64) -> Value {
    let mut object = ObjectInstance::new("tf".to_string());
    object.replace_property_unbounded("Numerator".into(), numerator);
    object.replace_property_unbounded("Denominator".into(), denominator);
    object.replace_property_unbounded("Ts".into(), Value::Num(sample_time));
    object.replace_property_unbounded("InputDelay".into(), Value::Num(0.0));
    object.replace_property_unbounded("OutputDelay".into(), Value::Num(0.0));
    object.replace_property_unbounded(
        "Variable".into(),
        Value::CharArray(CharArray::new_row(if sample_time > 0.0 {
            "z"
        } else {
            "s"
        })),
    );
    Value::Object(object)
}
fn tf(num: Vec<f64>, den: Vec<f64>, ts: f64) -> Value {
    raw_tf(
        Value::Tensor(Tensor::new(num.clone(), vec![1, num.len()]).unwrap()),
        Value::Tensor(Tensor::new(den.clone(), vec![1, den.len()]).unwrap()),
        ts,
    )
}
fn run(model: Value, frequency: Option<Value>, outputs: usize) -> BuiltinResult<Value> {
    let _count = crate::output_count::push_output_count(Some(outputs)).expect("output-count scope");
    block_on(nyquist_builtin(
        None,
        model,
        crate::builtins::common::test_support::builtin_arguments(frequency.into_iter().collect()),
    ))
}
fn values(value: &Value) -> Vec<f64> {
    let Value::Tensor(tensor) = value else {
        panic!("expected tensor, got {value:?}")
    };
    tensor.materialize_f64()
}

#[cfg(feature = "plot-core")]
#[test]
fn fatal_plot_failure_has_nyquist_identifier_and_retains_cause() {
    use std::error::Error;

    let source = crate::build_runtime_error("plot: rendering failed")
        .with_builtin("plot")
        .with_identifier("RunMat:plot:RenderFailed")
        .build();
    let wrapped = plot::plot_error(source);
    assert_eq!(wrapped.identifier(), Some("RunMat:nyquist:PlotFailed"));
    assert!(wrapped.message().contains("rendering failed"));
    assert!(wrapped.source().is_some());
}
#[test]
fn continuous_and_discrete_responses_preserve_exact_output_shape() {
    let w = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap());
    let Value::OutputList(outputs) = run(tf(vec![1.0], vec![1.0, 1.0], 0.0), Some(w), 3).unwrap()
    else {
        panic!("expected outputs")
    };
    assert_eq!(outputs.len(), 3);
    for output in &outputs {
        let Value::Tensor(tensor) = output else {
            panic!("expected tensor")
        };
        assert_eq!(tensor.shape(), vec![3, 1]);
    }
    assert_eq!(values(&outputs[2]), vec![0.0, 1.0, 2.0]);
    assert!((values(&outputs[0])[1] - 0.5).abs() < 1e-12);
    assert!((values(&outputs[1])[1] + 0.5).abs() < 1e-12);
    let Value::OutputList(discrete) = run(
        tf(vec![1.0], vec![1.0, -0.5], 0.1),
        Some(Value::Num(0.0)),
        2,
    )
    .unwrap() else {
        panic!("expected outputs")
    };
    assert!((values(&discrete[0])[0] - 2.0).abs() < 1e-12);
    assert_eq!(values(&discrete[1]), vec![0.0]);
}
#[cfg(feature = "plot-core")]
#[test]
fn default_grid_and_no_output_plot_form() {
    let Value::OutputList(outputs) = run(tf(vec![1.0], vec![1.0, -1.0], 0.1), None, 3).unwrap()
    else {
        panic!("expected outputs")
    };
    assert_eq!(values(&outputs[2]).len(), DEFAULT_FREQUENCY_POINTS);
    assert!(values(&outputs[2])[0] > 0.0);
    assert!(values(&outputs[2]).last().unwrap() < &(std::f64::consts::PI / 0.1));
    let result = run(tf(vec![1.0], vec![1.0, 1.0], 0.0), None, 0).unwrap();
    assert!(matches!(result, Value::OutputList(values) if values.is_empty()));
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn numeric_nyquist_survives_without_plotting_and_statement_fails_typed() {
    use std::error::Error;

    let model = tf(vec![1.0], vec![1.0, 1.0], 0.0);
    assert!(matches!(
        run(model.clone(), Some(Value::Num(0.0)), 2).expect("numeric response"),
        Value::OutputList(values) if values.len() == 2
    ));
    let error = run(model, None, 0).expect_err("presentation unavailable");
    assert!(error
        .source()
        .and_then(|source| source.downcast_ref::<crate::plotting_hooks::PlottingUnavailable>())
        .is_some());
}
#[test]
fn complex_coefficients_are_evaluated_without_a_real_mirror() {
    let numerator = Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 1.0)], vec![1, 1]).unwrap());
    let model = raw_tf(numerator, Value::Num(1.0), 0.0);
    let Value::OutputList(outputs) = run(model, Some(Value::Num(0.0)), 2).unwrap() else {
        panic!("expected outputs")
    };
    assert_eq!(values(&outputs[0]), vec![1.0]);
    assert_eq!(values(&outputs[1]), vec![1.0]);
}
#[test]
fn tiny_nonzero_coefficients_and_responses_are_not_zeroed_by_a_tolerance() {
    let Value::OutputList(outputs) =
        run(tf(vec![1.0e-20], vec![1.0], 0.0), Some(Value::Num(0.0)), 2).unwrap()
    else {
        panic!("expected outputs")
    };
    assert_eq!(values(&outputs[0]), vec![1.0e-20]);
    assert_eq!(values(&outputs[1]), vec![0.0]);
    let mut model = tf(vec![1.0], vec![1.0], 0.0);
    let Value::Object(ref mut object) = model else {
        unreachable!()
    };
    object.replace_property_unbounded("InputDelay".into(), Value::Num(1.0e-20));
    let error = run(model, None, 1).expect_err("nonzero delay is structural");
    assert_eq!(error.identifier(), Some("RunMat:nyquist:UnsupportedModel"));
}
#[test]
fn exact_wide_integer_frequency_is_admitted_but_inexact_is_rejected() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    let exact = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![0, 1_u64 << 53]), vec![1, 2]).unwrap(),
    );
    let Value::OutputList(outputs) = run(tf(vec![1.0], vec![1.0], 0.0), Some(exact), 3).unwrap()
    else {
        panic!("expected outputs")
    };
    assert_eq!(values(&outputs[2])[1], (1_u64 << 53) as f64);
    let inexact = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap(),
    );
    let error = run(tf(vec![1.0], vec![1.0], 0.0), Some(inexact), 1).expect_err("inexact integer");
    assert!(error.message().contains("exactly representable"));
}
#[test]
fn compatibility_and_invalid_host_forms_precede_provider_downloads() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let resident =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        provider.reset_telemetry();
        let _mode = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = run(
            tf(vec![1.0], vec![1.0], 0.0),
            Some(Value::GpuTensor(resident.clone())),
            1,
        )
        .expect_err("resident extension gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:NyquistResidentExtension")
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        drop(_mode);
        let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = run(
            Value::Object(ObjectInstance::new("ss".to_string())),
            Some(Value::GpuTensor(resident)),
            1,
        )
        .expect_err("host invalid model first");
        assert_eq!(error.identifier(), Some("RunMat:nyquist:UnsupportedModel"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn valid_nested_resident_coefficients_gather_only_consumed_values() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let denominator = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let unused = Tensor::new(vec![99.0], vec![1, 1]).unwrap();
        let den_gpu =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &denominator).unwrap();
        let unused_gpu =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &unused).unwrap();
        let mut model = raw_tf(Value::Num(1.0), Value::GpuTensor(den_gpu), 0.0);
        let Value::Object(ref mut object) = model else {
            unreachable!()
        };
        object.replace_property_unbounded("Unrelated".into(), Value::GpuTensor(unused_gpu));
        let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
        provider.reset_telemetry();
        let Value::OutputList(outputs) = run(model, Some(Value::Num(0.0)), 1).unwrap() else {
            panic!("output list")
        };
        assert_eq!(values(&outputs[0]), vec![1.0]);
        assert_eq!(
            provider.telemetry_snapshot().download_bytes,
            u64::try_from(denominator.len() * std::mem::size_of::<f64>()).unwrap()
        );
    });
}

#[test]
fn singular_points_and_extra_outputs_fail_with_owned_errors() {
    let error =
        run(tf(vec![1.0], vec![1.0, 0.0], 0.0), Some(Value::Num(0.0)), 1).expect_err("exact pole");
    assert_eq!(error.identifier(), Some("RunMat:nyquist:InvalidArgument"));
    assert!(error.message().contains("singular"));
    let error = run(tf(vec![1.0], vec![1.0], 0.0), None, 4).expect_err("output count");
    assert_eq!(error.identifier(), Some("RunMat:nyquist:InvalidArgument"));
}

#[test]
fn malformed_frequency_vectors_and_inexact_model_integers_fail() {
    let matrix = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap());
    let error = run(tf(vec![1.0], vec![1.0], 0.0), Some(matrix), 1).expect_err("frequency matrix");
    assert_eq!(error.identifier(), Some("RunMat:nyquist:InvalidArgument"));
    let negative = Value::Tensor(Tensor::new(vec![0.0, -1.0], vec![1, 2]).unwrap());
    let error =
        run(tf(vec![1.0], vec![1.0], 0.0), Some(negative), 1).expect_err("negative frequency");
    assert_eq!(error.identifier(), Some("RunMat:nyquist:InvalidArgument"));
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    let numerator = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap(),
    );
    let error = run(
        raw_tf(numerator, Value::Num(1.0), 0.0),
        Some(Value::Num(0.0)),
        1,
    )
    .expect_err("inexact coefficient");
    assert!(error.message().contains("exactly representable"));
}
