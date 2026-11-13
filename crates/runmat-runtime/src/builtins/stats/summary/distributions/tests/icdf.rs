use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn icdf_single_precision_preserves_canonical_scalar_broadcast_output_policy() {
    for (probabilities, expected_shape) in [
        (
            Tensor::from_f32(vec![0.5], Vec::new()).expect("rank-zero probabilities"),
            vec![1, 1],
        ),
        (
            Tensor::from_f32(vec![0.5, 0.5], vec![2]).expect("rank-one probabilities"),
            vec![1, 2],
        ),
    ] {
        let Value::Tensor(output) = block_on(icdf::icdf_builtin(
            Value::from("Normal"),
            Value::Tensor(probabilities),
            vec![Value::Num(0.0), Value::Num(1.0)],
        ))
        .expect("single-precision normal inverse") else {
            panic!("single precision is carried by a tensor")
        };
        assert_eq!(output.shape(), expected_shape);
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
        assert!(output
            .materialize_f64()
            .iter()
            .all(|value| value.abs() <= f64::EPSILON));
    }
}

#[test]
fn canonical_icdf_binding_uses_invocation_cancellation() {
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::rc::Rc;

    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    runtime.set_error_namespace("Invocation");
    let cancelled = CancellationSource::new();
    cancelled.cancel(CancellationReason::User);
    let runtime = runtime.with_cancellation_token(cancelled.token());
    let args = [
        Value::from("Binomial"),
        Value::Num(0.75),
        Value::Num(10.0),
        Value::Num(0.5),
    ];
    let bindings = crate::builtin::runtime_builtin_bindings_by_name("icdf");
    assert_eq!(bindings.len(), 1);
    let error = block_on((bindings[0].implementation)(
        crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(runtime), Some(1)),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("Invocation:Cancelled"));
}

#[test]
fn icdf_dispatches_supported_distribution_names() {
    let normal = block_on(icdf::icdf_builtin(
        Value::from("Normal"),
        Value::Num(0.5),
        vec![Value::Num(10.0), Value::Num(2.0)],
    ))
    .unwrap();
    match normal {
        Value::Num(value) => assert_close(value, 10.0, 1e-10),
        other => panic!("expected scalar normal icdf, got {other:?}"),
    }

    let student_t = block_on(icdf::icdf_builtin(
        Value::from("Student t"),
        Value::Num(0.95),
        vec![Value::Num(50.0)],
    ))
    .unwrap();
    match student_t {
        Value::Num(value) => assert_close(value, 1.675_905, 1e-6),
        other => panic!("expected scalar t icdf, got {other:?}"),
    }

    let weibull = block_on(icdf::icdf_builtin(
        Value::from("Weibull"),
        Value::Num(0.5),
        vec![Value::Num(3.0), Value::Num(4.0)],
    ))
    .unwrap();
    match weibull {
        Value::Num(value) => assert_close(value, 3.0 * std::f64::consts::LN_2.powf(0.25), 1e-12),
        other => panic!("expected scalar weibull icdf, got {other:?}"),
    }
}

#[test]
fn icdf_broadcasts_and_inverts_chisquare_and_binomial() {
    let probabilities = Value::Tensor(Tensor::new(vec![0.5, 0.95], vec![1, 2]).unwrap());
    let normal = block_on(icdf::icdf_builtin(
        Value::from("norm"),
        probabilities,
        vec![Value::Num(1.0), Value::Num(2.0)],
    ))
    .unwrap();
    match normal {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1, 2]);
            assert_close(tensor.materialize_f64()[0], 1.0, 1e-10);
            assert_close(tensor.materialize_f64()[1], 4.289_707_253_902_944, 1e-10);
        }
        other => panic!("expected tensor normal icdf, got {other:?}"),
    }

    let chi_square = block_on(icdf::icdf_builtin(
        Value::from("Chi-square"),
        Value::Num(0.5),
        vec![Value::Num(2.0)],
    ))
    .unwrap();
    match chi_square {
        Value::Num(value) => assert_close(value, 2.0 * std::f64::consts::LN_2, 1e-12),
        other => panic!("expected scalar chi-square icdf, got {other:?}"),
    }

    let binomial = block_on(icdf::icdf_builtin(
        Value::from("Binomial"),
        Value::Num(0.75),
        vec![Value::Num(10.0), Value::Num(0.5)],
    ))
    .unwrap();
    match binomial {
        Value::Num(value) => assert_close(value, 6.0, 0.0),
        other => panic!("expected scalar binomial icdf, got {other:?}"),
    }
}

#[test]
fn icdf_dispatches_common_distribution_families() {
    assert_close(
        icdf_scalar("Exponential", 0.5, vec![Value::Num(4.0)]),
        4.0 * std::f64::consts::LN_2,
        1e-12,
    );
    assert_close(
        icdf_scalar("Uniform", 0.25, vec![Value::Num(10.0), Value::Num(20.0)]),
        12.5,
        1e-12,
    );
    assert_close(
        icdf_scalar("Lognormal", 0.5, vec![Value::Num(1.0), Value::Num(2.0)]),
        std::f64::consts::E,
        1e-10,
    );
    assert_close(icdf_scalar("Poisson", 0.8, vec![Value::Num(3.0)]), 4.0, 0.0);
    assert_close(
        icdf_scalar("Beta", 0.5, vec![Value::Num(2.0), Value::Num(2.0)]),
        0.5,
        1e-12,
    );

    let gamma = icdf_scalar("Gamma", 0.6, vec![Value::Num(2.0), Value::Num(3.0)]);
    assert_close(
        distribution_math::regularized_gamma_p(2.0, gamma / 3.0),
        0.6,
        1e-10,
    );

    let f = icdf_scalar("F", 0.6, vec![Value::Num(5.0), Value::Num(10.0)]);
    let beta_x = 5.0 * f / (5.0 * f + 10.0);
    assert_close(
        distribution_math::regularized_beta(beta_x, 2.5, 5.0),
        0.6,
        1e-10,
    );
}

#[test]
fn icdf_handles_boundaries_invalid_params_and_bad_broadcasts() {
    assert_close(
        icdf_scalar("Uniform", 0.0, vec![Value::Num(2.0), Value::Num(5.0)]),
        2.0,
        0.0,
    );
    assert_close(
        icdf_scalar("Uniform", 1.0, vec![Value::Num(2.0), Value::Num(5.0)]),
        5.0,
        0.0,
    );
    assert_close(
        icdf_scalar("Gamma", 0.0, vec![Value::Num(2.0), Value::Num(3.0)]),
        0.0,
        0.0,
    );
    assert!(icdf_scalar("Gamma", 1.0, vec![Value::Num(2.0), Value::Num(3.0)]).is_infinite());
    assert_close(
        icdf_scalar("Binomial", 0.0, vec![Value::Num(10.0), Value::Num(0.5)]),
        0.0,
        0.0,
    );
    assert_close(
        icdf_scalar("Binomial", 1.0, vec![Value::Num(10.0), Value::Num(0.5)]),
        10.0,
        0.0,
    );
    assert!(icdf_scalar("Normal", -0.1, Vec::new()).is_nan());
    assert!(icdf_scalar("Gamma", 0.5, vec![Value::Num(0.0), Value::Num(3.0)]).is_nan());
    assert!(icdf_scalar("Uniform", 0.5, vec![Value::Num(5.0), Value::Num(2.0)]).is_nan());

    let err = block_on(icdf::icdf_builtin(
        Value::from("Exponential"),
        Value::Tensor(Tensor::new(vec![0.1, 0.2], vec![1, 2]).unwrap()),
        vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
        )],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:icdf:InvalidArgument"));
}

#[test]
fn icdf_broadcasts_later_parameter_shapes() {
    let normal = block_on(icdf::icdf_builtin(
        Value::from("Normal"),
        Value::Num(0.5),
        vec![
            Value::Num(0.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        ],
    ))
    .unwrap();
    match normal {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1, 2]);
            assert_close(tensor.materialize_f64()[0], 0.0, 1e-10);
            assert_close(tensor.materialize_f64()[1], 0.0, 1e-10);
        }
        other => panic!("expected tensor normal icdf, got {other:?}"),
    }

    let uniform = block_on(icdf::icdf_builtin(
        Value::from("Uniform"),
        Value::Num(0.5),
        vec![
            Value::Num(0.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        ],
    ))
    .unwrap();
    match uniform {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1, 2]);
            assert_close(tensor.materialize_f64()[0], 0.5, 1e-12);
            assert_close(tensor.materialize_f64()[1], 1.0, 1e-12);
        }
        other => panic!("expected tensor uniform icdf, got {other:?}"),
    }
}

#[test]
fn icdf_poisson_large_lambda_is_bounded_and_finite() {
    let value = icdf_scalar("Poisson", 0.5, vec![Value::Num(1000.0)]);
    assert!(value.is_finite());
    assert!(value >= 0.0);
    assert!(poisscdf_scalar(value, 1000.0) >= 0.5);
    assert!(value == 0.0 || poisscdf_scalar(value - 1.0, 1000.0) < 0.5);
}

#[test]
fn icdf_rejects_unknown_distribution_name() {
    let err = block_on(icdf::icdf_builtin(
        Value::from("not_a_distribution"),
        Value::Num(0.5),
        Vec::new(),
    ))
    .unwrap_err();
    assert!(err.message().contains("unsupported distribution"));
    assert_eq!(err.identifier(), Some("RunMat:icdf:InvalidArgument"));
}

#[test]
fn icdf_gates_nonfloating_extensions_and_preserves_single() {
    assert_eq!(
        runmat_builtins::ICDF_INTEGER_CAPABILITIES[0].inputs.len(),
        2
    );
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = block_on(icdf::icdf_builtin(
            Value::from("Normal"),
            Value::Int(runmat_value::IntValue::U16(1)),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            integer.identifier(),
            Some("RunMat:compatibility:IcdfIntegerInputExtension")
        );
        let logical = block_on(icdf::icdf_builtin(
            Value::from("Normal"),
            Value::Bool(true),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            logical.identifier(),
            Some("RunMat:compatibility:IcdfLogicalInputExtension")
        );
    }

    let single = block_on(icdf::icdf_builtin(
        Value::from("Normal"),
        Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
        Vec::new(),
    ))
    .expect("single icdf");
    let Value::Tensor(single) = single else {
        panic!("expected single tensor output");
    };
    assert_eq!(single.numeric_dtype(), NumericDType::F32);

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let wide = block_on(icdf::icdf_builtin(
        Value::from("Normal"),
        Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1)),
        Vec::new(),
    ))
    .unwrap_err();
    assert!(wide.message().contains("exactly representable as double"));
}

#[test]
fn icdf_final_carrier_preserves_single_across_named_families() {
    let cases: [(&str, Vec<Value>); 12] = [
        ("Normal", vec![]),
        ("Student t", vec![Value::Num(5.0)]),
        ("Weibull", vec![Value::Num(2.0), Value::Num(3.0)]),
        ("Chi-square", vec![Value::Num(2.0)]),
        ("Binomial", vec![Value::Num(10.0), Value::Num(0.5)]),
        ("Gamma", vec![Value::Num(2.0), Value::Num(3.0)]),
        ("Exponential", vec![Value::Num(4.0)]),
        ("Poisson", vec![Value::Num(3.0)]),
        ("Uniform", vec![Value::Num(1.0), Value::Num(2.0)]),
        ("Lognormal", vec![]),
        ("Beta", vec![Value::Num(2.0), Value::Num(2.0)]),
        ("F", vec![Value::Num(5.0), Value::Num(10.0)]),
    ];
    for (family, parameters) in cases {
        let output = block_on(icdf::icdf_builtin(
            Value::from(family),
            Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
            parameters,
        ))
        .unwrap_or_else(|error| panic!("{family}: {error}"));
        let Value::Tensor(tensor) = output else {
            panic!("{family}: expected single tensor final carrier");
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32, "{family}");
    }
}

#[test]
fn icdf_exact_wide_integer_parameters_cross_binary64_boundary_without_rounding() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let exact = block_on(icdf::icdf_builtin(
        Value::from("Normal"),
        Value::Num(0.5),
        vec![Value::Int(runmat_value::IntValue::U64(1_u64 << 54))],
    ))
    .expect("2^54 is exactly representable in binary64");
    assert!(matches!(exact, Value::Num(value) if value == (1_u64 << 54) as f64));
    let inexact = block_on(icdf::icdf_builtin(
        Value::from("Normal"),
        Value::Num(0.5),
        vec![Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1))],
    ))
    .expect_err("2^53+1 cannot cross binary64 exactly");
    assert!(inexact
        .message()
        .contains("exactly representable as double"));
}

#[test]
fn icdf_fitted_object_form_uses_the_same_integer_admission_boundary() {
    let pd = block_on(crate::builtins::stats::summary::fitdist::fitdist_builtin(
        Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap().into(),
        Value::from("Normal"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("fitted distribution");
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(icdf::icdf_builtin(
            pd.clone(),
            Value::Int(runmat_value::IntValue::U16(1)),
            Vec::new(),
        ))
        .expect_err("fitted-object typed p must gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:IcdfIntegerInputExtension")
        );
    }
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(icdf::icdf_builtin(
        pd.clone(),
        Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1)),
        Vec::new(),
    ))
    .expect_err("fitted-object wide p must reject exactly");
    assert!(error.message().contains("exactly representable as double"));

    let single = block_on(icdf::icdf_builtin(
        pd,
        Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
        Vec::new(),
    ))
    .expect("fitted object with single probability");
    assert!(matches!(single, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32));
}

#[test]
fn icdf_resident_floating_input_returns_to_exact_owner() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![0.5], vec![1, 1]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &input).expect("resident p");
        runmat_accelerate::fusion_residency::mark(&source);
        let output = block_on(icdf::icdf_builtin(
            Value::from("Normal"),
            Value::GpuTensor(source.clone()),
            Vec::new(),
        ))
        .expect("resident icdf");
        let Value::GpuTensor(output_handle) = &output else {
            panic!("expected resident output");
        };
        assert_eq!(output_handle.device_id(), source.device_id());
        assert!(
            gpu_helpers::exact_provider_for_handle(output_handle).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        assert!(runmat_accelerate::fusion_residency::is_resident(&source));
        let gathered =
            crate::builtins::common::test_support::gather(output).expect("gather output");
        assert!(gathered.materialize_f64()[0].abs() < 1.0e-12);
        drop(source);
    });
}
