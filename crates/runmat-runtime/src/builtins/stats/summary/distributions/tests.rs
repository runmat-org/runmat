use super::*;
use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

fn assert_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "actual={actual} expected={expected}"
    );
}

fn icdf_scalar(name: &str, p: f64, rest: Vec<Value>) -> f64 {
    match block_on(icdf::icdf_builtin(Value::from(name), Value::Num(p), rest)).unwrap() {
        Value::Num(value) => value,
        other => panic!("expected scalar icdf result, got {other:?}"),
    }
}

fn mirrorless_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn all_integer_binocdf_storages(value: i8) -> Vec<IntegerStorage> {
    vec![
        IntegerStorage::I8(vec![value]),
        IntegerStorage::I16(vec![i16::from(value)]),
        IntegerStorage::I32(vec![i32::from(value)]),
        IntegerStorage::I64(vec![i64::from(value)]),
        IntegerStorage::U8(vec![value as u8]),
        IntegerStorage::U16(vec![value as u16]),
        IntegerStorage::U32(vec![value as u32]),
        IntegerStorage::U64(vec![value as u64]),
    ]
}

#[test]
fn distribution_broadcast_indexing_appends_trailing_singletons() {
    let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    assert_eq!(
        broadcast_tensor_to("normcdf", &tensor, &[2, 1, 3]).unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn normal_distribution_scalar_values() {
    let pdf = block_on(normpdf::normpdf_builtin(Value::Num(0.0), Vec::new())).unwrap();
    match pdf {
        Value::Num(value) => assert_close(value, INV_SQRT_2PI, 1e-12),
        other => panic!("expected scalar pdf, got {other:?}"),
    }
    let cdf = block_on(normcdf::normcdf_builtin(Value::Num(0.0), Vec::new())).unwrap();
    match cdf {
        Value::Num(value) => assert_close(value, 0.5, 1e-12),
        other => panic!("expected scalar cdf, got {other:?}"),
    }
    let inv = block_on(norminv::norminv_builtin(Value::Num(0.5), Vec::new())).unwrap();
    match inv {
        Value::Num(value) => assert_close(value, 0.0, 1e-10),
        other => panic!("expected scalar inv, got {other:?}"),
    }
}

#[test]
fn normal_distribution_broadcasts_parameters() {
    let x = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
    let out = block_on(normcdf::normcdf_builtin(
        x,
        vec![Value::Num(0.0), Value::Num(1.0)],
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 2]);
            assert_close(tensor.materialize_f64()[0], 0.5, 1e-12);
            assert_close(tensor.materialize_f64()[1], 0.841_344_746_068_543, 1e-12);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn normal_distribution_integer_roles_are_independently_gated() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let x = block_on(normcdf::normcdf_builtin(
        Value::Int(runmat_value::IntValue::I16(0)),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(
        x.identifier(),
        Some("RunMat:compatibility:NormcdfIntegerXExtension")
    );
    let mu = block_on(normcdf::normcdf_builtin(
        Value::Num(0.0),
        vec![Value::Int(runmat_value::IntValue::I16(0))],
    ))
    .unwrap_err();
    assert_eq!(
        mu.identifier(),
        Some("RunMat:compatibility:NormcdfIntegerMuExtension")
    );
    let sigma = block_on(normcdf::normcdf_builtin(
        Value::Num(0.0),
        vec![Value::Num(0.0), Value::Int(runmat_value::IntValue::U16(1))],
    ))
    .unwrap_err();
    assert_eq!(
        sigma.identifier(),
        Some("RunMat:compatibility:NormcdfIntegerSigmaExtension")
    );
}

#[test]
fn normal_distribution_extensions_preserve_single_and_reject_lossy_integers() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let single = Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap());
    let out = block_on(normpdf::normpdf_builtin(
        single,
        vec![
            mirrorless_int_tensor(IntegerStorage::I16(vec![0]), vec![1, 1]),
            mirrorless_int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
        ],
    ))
    .unwrap();
    let Value::Tensor(out) = out else {
        panic!("single normal inputs must retain tensor class");
    };
    assert_eq!(out.numeric_dtype(), NumericDType::F32);
    assert_close(out.materialize_f64()[0], INV_SQRT_2PI, 1.0e-6);

    for storage in all_integer_binocdf_storages(0) {
        let out = block_on(normcdf::normcdf_builtin(
            Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap()),
            Vec::new(),
        ))
        .unwrap();
        match out {
            Value::Num(value) => assert_close(value, 0.5, 1.0e-12),
            other => panic!("expected double scalar, got {other:?}"),
        }
    }

    let wide = mirrorless_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]);
    let error = block_on(norminv::norminv_builtin(wide, Vec::new())).unwrap_err();
    assert!(error.message.contains("exactly representable"));
}

#[test]
#[cfg(feature = "wgpu")]
fn normal_distribution_wgpu_integer_fallback_preserves_class_and_explicit_intent() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("wgpu provider");
    let input = Tensor::new_integer(IntegerStorage::U16(vec![0, 1]), vec![1, 2]).unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
    let out = block_on(normcdf::normcdf_builtin(
        Value::GpuTensor(handle.clone()),
        Vec::new(),
    ))
    .expect("normcdf");
    let Value::GpuTensor(out) = out else {
        panic!("expected automatic resident double output");
    };
    assert_eq!(
        runmat_accelerate_api::handle_precision(&out),
        Some(ProviderPrecision::F64)
    );
    let host = test_support::gather(Value::GpuTensor(out)).expect("gather automatic output");
    assert_close(host.materialize_f64()[0], 0.5, 1.0e-12);

    let handle = handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let out = block_on(normcdf::normcdf_builtin(
        Value::GpuTensor(handle),
        Vec::new(),
    ))
    .expect("explicit resident double output");
    let Value::GpuTensor(out) = out else {
        panic!("expected explicit resident output");
    };
    assert!(runmat_accelerate_api::handle_is_explicit(&out));
    assert_eq!(
        runmat_accelerate_api::handle_precision(&out),
        Some(ProviderPrecision::F64)
    );
}

#[test]
fn normal_distribution_accepts_mu_only_and_upper_tail() {
    let cdf = block_on(normcdf::normcdf_builtin(
        Value::Num(2.0),
        vec![Value::Num(1.0)],
    ))
    .unwrap();
    match cdf {
        Value::Num(value) => assert_close(value, 0.841_344_746_068_543, 1e-12),
        other => panic!("expected scalar cdf, got {other:?}"),
    }

    let upper = block_on(normcdf::normcdf_builtin(
        Value::Num(8.0),
        vec![Value::Num(0.0), Value::Num(1.0), Value::from("upper")],
    ))
    .unwrap();
    match upper {
        Value::Num(value) => assert_close(value, 6.220_960_574_271_784e-16, 1e-28),
        other => panic!("expected scalar upper cdf, got {other:?}"),
    }

    let pdf = block_on(normpdf::normpdf_builtin(
        Value::Num(2.0),
        vec![Value::Num(1.0)],
    ))
    .unwrap();
    match pdf {
        Value::Num(value) => assert_close(value, INV_SQRT_2PI * (-0.5f64).exp(), 1e-12),
        other => panic!("expected scalar pdf, got {other:?}"),
    }

    let inv = block_on(norminv::norminv_builtin(
        Value::Num(0.5),
        vec![Value::Num(2.0)],
    ))
    .unwrap();
    match inv {
        Value::Num(value) => assert_close(value, 2.0, 1e-10),
        other => panic!("expected scalar inv, got {other:?}"),
    }
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
            assert_eq!(tensor.shape, vec![1, 2]);
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
            assert_eq!(tensor.shape, vec![1, 2]);
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
            assert_eq!(tensor.shape, vec![1, 2]);
            assert_close(tensor.materialize_f64()[0], 0.5, 1e-12);
            assert_close(tensor.materialize_f64()[1], 1.0, 1e-12);
        }
        other => panic!("expected tensor uniform icdf, got {other:?}"),
    }
}

#[test]
fn icdf_type_uses_broadcasted_numeric_argument_shape() {
    let probability_type = Type::Tensor {
        shape: Some(vec![Some(1), Some(2)]),
    };
    let inferred = icdf_type(
        &[Type::String, probability_type.clone(), Type::Num, Type::Num],
        &ResolveContext::default(),
    );
    assert_eq!(inferred, probability_type);
    assert_eq!(
        icdf_type(&[Type::String, Type::Num], &ResolveContext::default()),
        Type::Num
    );
    assert_eq!(
        icdf_type(
            &[
                Type::String,
                Type::Num,
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(2)])
                }
            ],
            &ResolveContext::default()
        ),
        Type::Tensor {
            shape: Some(vec![Some(1), Some(2)])
        }
    );
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
fn normal_distribution_rejects_invalid_sigma_with_nan_outputs() {
    let pdf = block_on(normpdf::normpdf_builtin(
        Value::Num(0.0),
        vec![Value::Num(0.0), Value::Num(0.0)],
    ))
    .unwrap();
    assert!(matches!(pdf, Value::Num(value) if value.is_nan()));

    let inv = block_on(norminv::norminv_builtin(
        Value::Num(0.5),
        vec![Value::Num(0.0), Value::Num(0.0)],
    ))
    .unwrap();
    assert!(matches!(inv, Value::Num(value) if value.is_nan()));

    let cdf = block_on(normcdf::normcdf_builtin(
        Value::Num(0.0),
        vec![Value::Num(1.0), Value::Num(0.0)],
    ))
    .unwrap();
    assert_eq!(cdf, Value::Num(0.0));
}

#[test]
fn student_t_distribution_scalar_values() {
    let pdf = block_on(tpdf::tpdf_builtin(Value::Num(0.0), vec![Value::Num(1.0)])).unwrap();
    match pdf {
        Value::Num(value) => assert_close(value, std::f64::consts::FRAC_1_PI, 1e-12),
        other => panic!("expected scalar pdf, got {other:?}"),
    }

    let cdf = block_on(tcdf::tcdf_builtin(Value::Num(0.0), vec![Value::Num(10.0)])).unwrap();
    match cdf {
        Value::Num(value) => assert_close(value, 0.5, 1e-12),
        other => panic!("expected scalar cdf, got {other:?}"),
    }

    let inv = block_on(tinv::tinv_builtin(Value::Num(0.95), vec![Value::Num(50.0)])).unwrap();
    match inv {
        Value::Num(value) => assert_close(value, 1.675_905, 1e-6),
        other => panic!("expected scalar inv, got {other:?}"),
    }
}

#[test]
fn student_t_distribution_broadcasts_and_upper_tail() {
    let x = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap());
    let nu = Value::Tensor(Tensor::new(vec![1.0, 5.0, f64::INFINITY], vec![1, 3]).unwrap());
    let out = block_on(tcdf::tcdf_builtin(x, vec![nu])).unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 3]);
            assert_close(tensor.materialize_f64()[0], 0.5, 1e-12);
            assert_close(tensor.materialize_f64()[1], 0.818_391_266, 1e-9);
            assert_close(tensor.materialize_f64()[2], 0.977_249_868, 1e-9);
        }
        other => panic!("expected tensor cdf, got {other:?}"),
    }

    let upper = block_on(tcdf::tcdf_builtin(
        Value::Num(10.0),
        vec![Value::Num(99.0), Value::from("upper")],
    ))
    .unwrap();
    match upper {
        Value::Num(value) => assert_close(value, 5.469_9e-17, 1e-20),
        other => panic!("expected scalar upper cdf, got {other:?}"),
    }
}

#[test]
fn tpdf_preserves_single_and_gates_exact_integer_roles() {
    let single = block_on(tpdf::tpdf_builtin(
        Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
        vec![Value::Num(5.0)],
    ))
    .unwrap();
    assert!(matches!(single, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32));

    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(tpdf::tpdf_builtin(
            mirrorless_int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
            vec![Value::Num(5.0)],
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:TpdfIntegerXExtension")
        );
    }

    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(tpdf::tpdf_builtin(
        mirrorless_int_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]),
        vec![Value::Num(5.0)],
    ))
    .unwrap_err();
    assert!(error.message().contains("exactly representable"));
}

#[test]
fn tcdf_integer_roles_follow_compatibility_mode_and_exact_double_boundary() {
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer_x = block_on(tcdf::tcdf_builtin(
            mirrorless_int_tensor(IntegerStorage::I16(vec![1]), vec![1, 1]),
            vec![Value::Num(5.0)],
        ))
        .unwrap_err();
        assert_eq!(
            integer_x.identifier(),
            Some("RunMat:compatibility:TcdfIntegerXExtension")
        );
        let integer_nu = block_on(tcdf::tcdf_builtin(
            Value::Num(1.0),
            vec![mirrorless_int_tensor(
                IntegerStorage::U16(vec![5]),
                vec![1, 1],
            )],
        ))
        .unwrap_err();
        assert_eq!(
            integer_nu.identifier(),
            Some("RunMat:compatibility:TcdfIntegerDegreesOfFreedomExtension")
        );
    }

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let exact = block_on(tcdf::tcdf_builtin(
        mirrorless_int_tensor(IntegerStorage::U64(vec![1]), vec![1, 1]),
        vec![mirrorless_int_tensor(
            IntegerStorage::U64(vec![5]),
            vec![1, 1],
        )],
    ))
    .expect("exact integer tcdf inputs");
    assert!(matches!(exact, Value::Num(value) if value > 0.8 && value < 0.9));

    let inexact = block_on(tcdf::tcdf_builtin(
        mirrorless_int_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]),
        vec![Value::Num(5.0)],
    ))
    .unwrap_err();
    assert!(inexact.message().contains("exactly representable"));
}

#[test]
fn tinv_integer_roles_gate_independently_and_use_the_exact_double_boundary() {
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer_p = block_on(tinv::tinv_builtin(
            mirrorless_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
            vec![Value::Num(5.0)],
        ))
        .expect_err("integer p must gate");
        assert_eq!(
            integer_p.identifier(),
            Some("RunMat:compatibility:TinvIntegerProbabilityExtension")
        );
        let integer_nu = block_on(tinv::tinv_builtin(
            Value::Num(0.5),
            vec![mirrorless_int_tensor(
                IntegerStorage::U16(vec![5]),
                vec![1, 1],
            )],
        ))
        .expect_err("integer nu must gate");
        assert_eq!(
            integer_nu.identifier(),
            Some("RunMat:compatibility:TinvIntegerDegreesOfFreedomExtension")
        );
    }

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let exact = block_on(tinv::tinv_builtin(
        mirrorless_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
        vec![mirrorless_int_tensor(
            IntegerStorage::U16(vec![5]),
            vec![1, 1],
        )],
    ))
    .expect("exact integer tinv inputs");
    assert!(matches!(exact, Value::Num(value) if value.is_infinite() && value.is_sign_positive()));

    let inexact = block_on(tinv::tinv_builtin(
        Value::Num(0.5),
        vec![mirrorless_int_tensor(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            vec![1, 1],
        )],
    ))
    .expect_err("wide integer nu must reject");
    assert!(inexact.message().contains("exactly representable"));
}

#[test]
fn tinv_preserves_native_single_output_precision() {
    let output = block_on(tinv::tinv_builtin(
        Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
        vec![Value::Num(5.0)],
    ))
    .expect("single tinv");
    let Value::Tensor(output) = output else {
        panic!("expected single tensor output")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.materialize_f64(), vec![0.0]);
}

#[test]
fn tinv_declares_independent_integer_role_metadata() {
    let builtin = runmat_builtins::builtin_function_by_name("tinv").unwrap();
    assert_eq!(builtin.integer_capabilities.len(), 2);
    assert_eq!(builtin.extensions.len(), 3);
    assert!(builtin
        .integer_capabilities
        .iter()
        .all(|capability| capability.inputs[0].classes.len() == 8));
}

#[test]
fn tinv_resident_floating_input_returns_to_the_exact_owner() {
    test_support::with_test_provider(|provider| {
        let source =
            gpu_helpers::upload_tensor(provider, &Tensor::new(vec![0.5], vec![1, 1]).unwrap())
                .expect("resident probability");
        runmat_accelerate::fusion_residency::mark(&source);
        let output = block_on(tinv::tinv_builtin(
            Value::GpuTensor(source.clone()),
            vec![Value::Num(5.0)],
        ))
        .expect("resident tinv");
        let Value::GpuTensor(output_handle) = &output else {
            panic!("expected resident output")
        };
        assert_eq!(output_handle.device_id(), source.device_id());
        assert!(
            gpu_helpers::exact_provider_for_handle(output_handle).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        assert!(runmat_accelerate::fusion_residency::is_resident(&source));
        let gathered = test_support::gather(output).expect("gather tinv output");
        assert_eq!(gathered.materialize_f64(), vec![0.0]);
        drop(source);
    });
}

#[test]
fn student_t_distribution_rejects_bad_shapes_and_returns_nan_for_bad_parameters() {
    let err = block_on(tpdf::tpdf_builtin(
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
        vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
        )],
    ))
    .unwrap_err();
    assert!(err.message().contains("tpdf"));
    assert_eq!(err.identifier(), Some("RunMat:tpdf:InvalidArgument"));

    let out = block_on(tcdf::tcdf_builtin(Value::Num(0.0), vec![Value::Num(-1.0)])).unwrap();
    assert!(matches!(out, Value::Num(value) if value.is_nan()));
}

#[test]
fn student_t_distribution_extreme_tails_remain_representable() {
    let lower = block_on(tinv::tinv_builtin(
        Value::Num(1.0e-20),
        vec![Value::Num(1.0)],
    ))
    .unwrap();
    match lower {
        Value::Num(value) => {
            assert!(value.is_finite());
            assert_close(value / 1.0e19, -3.183_098_861_837_907, 1e-10);
        }
        other => panic!("expected scalar inv, got {other:?}"),
    }

    let upper = block_on(tcdf::tcdf_builtin(
        Value::Num(1.0e200),
        vec![Value::Num(1.0), Value::from("upper")],
    ))
    .unwrap();
    match upper {
        Value::Num(value) => assert_close(value / 1.0e-201, 3.183_098_861_837_907, 1e-12),
        other => panic!("expected scalar upper cdf, got {other:?}"),
    }
}

#[test]
fn student_t_distribution_large_nu_uses_normal_limit() {
    let pdf = block_on(tpdf::tpdf_builtin(
        Value::Num(1.0),
        vec![Value::Num(1.0e12)],
    ))
    .unwrap();
    match pdf {
        Value::Num(value) => assert_close(value, INV_SQRT_2PI * (-0.5f64).exp(), 1e-12),
        other => panic!("expected scalar pdf, got {other:?}"),
    }

    let cdf = block_on(tcdf::tcdf_builtin(
        Value::Num(1.0),
        vec![Value::Num(1.0e12)],
    ))
    .unwrap();
    match cdf {
        Value::Num(value) => assert_close(value, 0.841_344_746_068_543, 1e-12),
        other => panic!("expected scalar cdf, got {other:?}"),
    }

    let inv = block_on(tinv::tinv_builtin(
        Value::Num(0.975),
        vec![Value::Num(1.0e12)],
    ))
    .unwrap();
    match inv {
        Value::Num(value) => assert_close(value, 1.959_963_984_540_053_8, 1e-12),
        other => panic!("expected scalar inv, got {other:?}"),
    }
}

#[test]
fn binomial_chi_square_and_weibull_distribution_values() {
    let binomial = block_on(binocdf::binocdf_builtin(
        Value::Num(55.0),
        vec![Value::Num(100.0), Value::Num(0.5)],
    ))
    .unwrap();
    match binomial {
        Value::Num(value) => assert_close(value, 0.864_373_487_963_083, 1.0e-12),
        other => panic!("expected scalar binocdf, got {other:?}"),
    }

    let upper = block_on(binocdf::binocdf_builtin(
        Value::Num(55.0),
        vec![Value::Num(100.0), Value::Num(0.5), Value::from("upper")],
    ))
    .unwrap();
    match upper {
        Value::Num(value) => assert_close(value, 0.135_626_512_036_917, 1.0e-12),
        other => panic!("expected scalar upper binocdf, got {other:?}"),
    }

    let chi2 = block_on(chi2cdf::chi2cdf_builtin(
        Value::Num(3.0),
        vec![Value::Num(5.0)],
    ))
    .unwrap();
    match chi2 {
        Value::Num(value) => assert_close(value, 0.300_014_164_121_372, 1.0e-12),
        other => panic!("expected scalar chi2cdf, got {other:?}"),
    }

    let weibull = block_on(wblinv::wblinv_builtin(
        Value::Num(0.5),
        vec![Value::Num(3.0), Value::Num(4.0)],
    ))
    .unwrap();
    match weibull {
        Value::Num(value) => assert_close(value, 3.0 * std::f64::consts::LN_2.powf(0.25), 1.0e-12),
        other => panic!("expected scalar wblinv, got {other:?}"),
    }
}

#[test]
fn distribution_helpers_accept_typed_integer_tensors_at_f64_boundary() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(normcdf::normcdf_builtin(
        mirrorless_int_tensor(IntegerStorage::I16(vec![0, 1]), vec![1, 2]),
        vec![
            mirrorless_int_tensor(IntegerStorage::I16(vec![0]), vec![1, 1]),
            mirrorless_int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
        ],
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 2]);
            assert_close(tensor.materialize_f64()[0], 0.5, 1.0e-12);
            assert_close(tensor.materialize_f64()[1], 0.841_344_746_068_543, 1.0e-12);
        }
        other => panic!("expected tensor normcdf, got {other:?}"),
    }

    let chi2 = block_on(chi2cdf::chi2cdf_builtin(
        mirrorless_int_tensor(IntegerStorage::U16(vec![3]), vec![1, 1]),
        vec![mirrorless_int_tensor(
            IntegerStorage::U16(vec![5]),
            vec![1, 1],
        )],
    ))
    .unwrap();
    match chi2 {
        Value::Num(value) => assert_close(value, 0.300_014_164_121_372, 1.0e-12),
        other => panic!("expected scalar chi2cdf, got {other:?}"),
    }
}

#[test]
fn chi2cdf_integer_roles_are_gated_and_wide_values_do_not_round() {
    assert_eq!(chi2cdf::INTEGER_CAPABILITIES.len(), 2);
    assert!(chi2cdf::INTEGER_CAPABILITIES
        .iter()
        .all(|capability| capability.inputs[0].classes.len() == 8));

    {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
        let x_error = block_on(chi2cdf::chi2cdf_builtin(
            Value::Int(runmat_value::IntValue::I16(3)),
            vec![Value::Num(5.0)],
        ))
        .unwrap_err();
        assert_eq!(
            x_error.identifier(),
            Some("RunMat:compatibility:Chi2cdfIntegerXExtension")
        );
        let nu_error = block_on(chi2cdf::chi2cdf_builtin(
            Value::Num(3.0),
            vec![Value::Int(runmat_value::IntValue::U16(5))],
        ))
        .unwrap_err();
        assert_eq!(
            nu_error.identifier(),
            Some("RunMat:compatibility:Chi2cdfIntegerDegreesOfFreedomExtension")
        );
    }

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let wide_error = block_on(chi2cdf::chi2cdf_builtin(
        Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1)),
        vec![Value::Num(5.0)],
    ))
    .unwrap_err();
    assert_eq!(
        wide_error.identifier(),
        Some("RunMat:chi2cdf:InvalidArgument")
    );
    assert!(wide_error
        .message()
        .contains("exactly representable as double"));

    let signed_wide_error = block_on(chi2cdf::chi2cdf_builtin(
        Value::Int(runmat_value::IntValue::I64(i64::MIN)),
        vec![Value::Num(5.0)],
    ))
    .unwrap_err();
    assert!(signed_wide_error
        .message()
        .contains("exactly representable as double"));
}

#[test]
fn chi2cdf_executes_every_integer_class_and_upper_tail() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for (x_storage, nu_storage) in all_integer_binocdf_storages(3)
        .into_iter()
        .zip(all_integer_binocdf_storages(5))
    {
        let out = block_on(chi2cdf::chi2cdf_builtin(
            Value::Tensor(Tensor::new_integer(x_storage, vec![1, 1]).unwrap()),
            vec![Value::Tensor(
                Tensor::new_integer(nu_storage, vec![1, 1]).unwrap(),
            )],
        ))
        .unwrap();
        assert!(
            matches!(out, Value::Num(value) if (value - 0.300_014_164_121_372).abs() < 1.0e-12)
        );
    }
    let upper = block_on(chi2cdf::chi2cdf_builtin(
        Value::Int(runmat_value::IntValue::U16(3)),
        vec![
            Value::Int(runmat_value::IntValue::U16(5)),
            Value::from("upper"),
        ],
    ))
    .unwrap();
    assert!(matches!(upper, Value::Num(value) if (value - 0.699_985_835_878_628).abs() < 1.0e-12));
}

#[test]
fn chi2cdf_logical_and_resident_integer_gates_precede_gather() {
    {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
        let logical = block_on(chi2cdf::chi2cdf_builtin(
            Value::Bool(true),
            vec![Value::Num(5.0)],
        ))
        .unwrap_err();
        assert_eq!(
            logical.identifier(),
            Some("RunMat:compatibility:Chi2cdfLogicalInputExtension")
        );
    }

    crate::builtins::common::test_support::with_test_provider(|provider| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
        {
            let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(chi2cdf::chi2cdf_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::Num(5.0)],
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:Chi2cdfIntegerXExtension")
            );
        }
        let result = block_on(chi2cdf::chi2cdf_builtin(
            Value::GpuTensor(handle.clone()),
            vec![Value::Num(5.0)],
        ))
        .unwrap();
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = crate::builtins::common::test_support::gather(result).unwrap();
        assert!((gathered.materialize_f64()[0] - 0.300_014_164_121_372).abs() < 1.0e-12);
        drop(handle);
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn chi2cdf_wgpu_integer_fallback_preserves_residency() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("wgpu provider");
    let input = Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
    let result = block_on(chi2cdf::chi2cdf_builtin(
        Value::GpuTensor(handle),
        vec![Value::Int(runmat_value::IntValue::U16(5))],
    ))
    .expect("chi2cdf");
    assert!(matches!(result, Value::GpuTensor(_)));
    let gathered = crate::builtins::common::test_support::gather(result).expect("gather");
    assert!((gathered.materialize_f64()[0] - 0.300_014_164_121_372).abs() < 1.0e-6);
}

#[test]
fn chi2cdf_preserves_single_and_uses_only_scalar_expansion() {
    let single = Tensor::from_f32(vec![3.0, 4.0], vec![1, 2]).unwrap();
    let out = block_on(chi2cdf::chi2cdf_builtin(
        Value::Tensor(single),
        vec![Value::Num(5.0)],
    ))
    .unwrap();
    let Value::Tensor(tensor) = out else {
        panic!("expected single tensor");
    };
    assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    assert_eq!(tensor.shape, vec![1, 2]);

    let err = block_on(chi2cdf::chi2cdf_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        vec![Value::Tensor(
            Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap(),
        )],
    ))
    .unwrap_err();
    assert!(err.message().contains("same size"));
}

#[test]
fn distribution_helpers_broadcast_and_return_nan_for_bad_parameters() {
    let out = block_on(chi2cdf::chi2cdf_builtin(
        Value::Num(3.0),
        vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 5.0], vec![1, 3]).unwrap(),
        )],
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 3]);
            assert_close(tensor.materialize_f64()[2], 0.300_014_164_121_372, 1.0e-12);
        }
        other => panic!("expected tensor chi2cdf, got {other:?}"),
    }

    let invalid = block_on(wblinv::wblinv_builtin(
        Value::Num(0.5),
        vec![Value::Num(-1.0), Value::Num(2.0)],
    ))
    .unwrap();
    assert!(matches!(invalid, Value::Num(value) if value.is_nan()));
}

#[test]
fn binocdf_boundaries_and_invalid_parameters_match_distribution_contract() {
    let below = block_on(binocdf::binocdf_builtin(
        Value::Num(-1.0),
        vec![Value::Num(10.0), Value::Num(0.5)],
    ))
    .unwrap();
    assert_eq!(below, Value::Num(0.0));

    let above = block_on(binocdf::binocdf_builtin(
        Value::Num(10.0),
        vec![Value::Num(10.0), Value::Num(0.5)],
    ))
    .unwrap();
    assert_eq!(above, Value::Num(1.0));

    let always_zero = block_on(binocdf::binocdf_builtin(
        Value::Num(0.0),
        vec![Value::Num(10.0), Value::Num(0.0)],
    ))
    .unwrap();
    assert_eq!(always_zero, Value::Num(1.0));

    let impossible = block_on(binocdf::binocdf_builtin(
        Value::Num(5.0),
        vec![Value::Num(10.5), Value::Num(0.5)],
    ))
    .unwrap();
    assert!(matches!(impossible, Value::Num(value) if value.is_nan()));
}

#[test]
fn binocdf_classifies_all_integer_input_positions_as_runmat_extensions() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_integer_binocdf_storages(1) {
        let x = block_on(binocdf::binocdf_builtin(
            mirrorless_int_tensor(storage.clone(), vec![1, 1]),
            vec![Value::Num(2.0), Value::Num(0.5)],
        ))
        .expect("integer x");
        assert!(matches!(x, Value::Num(value) if (value - 0.75).abs() < 1.0e-12));

        let n = block_on(binocdf::binocdf_builtin(
            Value::Num(0.0),
            vec![
                mirrorless_int_tensor(storage.clone(), vec![1, 1]),
                Value::Num(0.5),
            ],
        ))
        .expect("integer n");
        assert!(matches!(n, Value::Num(value) if (value - 0.5).abs() < 1.0e-12));

        let p = block_on(binocdf::binocdf_builtin(
            Value::Num(0.0),
            vec![Value::Num(1.0), mirrorless_int_tensor(storage, vec![1, 1])],
        ))
        .expect("integer p");
        assert_eq!(p, Value::Num(0.0));
    }
}

#[test]
fn binocdf_compatibility_mode_rejects_integer_and_logical_extensions() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer_x = block_on(binocdf::binocdf_builtin(
        mirrorless_int_tensor(IntegerStorage::I8(vec![1]), vec![1, 1]),
        vec![Value::Num(2.0), Value::Num(0.5)],
    ))
    .unwrap_err();
    assert_eq!(
        integer_x.identifier(),
        Some("RunMat:compatibility:BinocdfIntegerXExtension")
    );
    let integer_n = block_on(binocdf::binocdf_builtin(
        Value::Num(1.0),
        vec![
            mirrorless_int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
            Value::Num(0.5),
        ],
    ))
    .unwrap_err();
    assert_eq!(
        integer_n.identifier(),
        Some("RunMat:compatibility:BinocdfIntegerTrialsExtension")
    );
    let integer_p = block_on(binocdf::binocdf_builtin(
        Value::Num(1.0),
        vec![
            Value::Num(2.0),
            mirrorless_int_tensor(IntegerStorage::U32(vec![1]), vec![1, 1]),
        ],
    ))
    .unwrap_err();
    assert_eq!(
        integer_p.identifier(),
        Some("RunMat:compatibility:BinocdfIntegerProbabilityExtension")
    );
    let logical = block_on(binocdf::binocdf_builtin(
        Value::Bool(true),
        vec![Value::Num(2.0), Value::Num(0.5)],
    ))
    .unwrap_err();
    assert_eq!(
        logical.identifier(),
        Some("RunMat:compatibility:BinocdfLogicalInputExtension")
    );
}

#[test]
fn binocdf_preserves_single_output_and_rejects_inexact_wide_integers() {
    let single = block_on(binocdf::binocdf_builtin(
        Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap()),
        vec![Value::Num(2.0), Value::Num(0.5)],
    ))
    .expect("single binocdf");
    let Value::Tensor(single) = single else {
        panic!("single scalar output must retain its class");
    };
    assert_eq!(single.numeric_dtype(), NumericDType::F32);
    assert!((single.materialize_f64()[0] - 0.75).abs() < f64::from(f32::EPSILON));

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(binocdf::binocdf_builtin(
        mirrorless_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
        vec![Value::Num(2.0), Value::Num(0.5)],
    ))
    .unwrap_err();
    assert!(error.message().contains("exactly representable as double"));
}

#[test]
fn binocdf_gpu_fallback_preserves_residency_precision_and_integer_guard_order() {
    use crate::builtins::common::test_support;

    crate::builtins::common::test_support::with_test_provider(|provider| {
        let single = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single input");
        let handle = gpu_helpers::upload_tensor(provider, &single).expect("single upload");
        let result = block_on(binocdf::binocdf_builtin(
            Value::GpuTensor(handle),
            vec![Value::Num(2.0), Value::Num(0.5)],
        ))
        .expect("resident binocdf");
        let Value::GpuTensor(result_handle) = &result else {
            panic!("expected resident output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(result_handle),
            Some(ProviderPrecision::F32)
        );
        let gathered = test_support::gather(result).expect("gather result");
        assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
        assert!((gathered.materialize_f64()[0] - 0.75).abs() < f64::from(f32::EPSILON));

        let integer =
            Tensor::new_integer(IntegerStorage::I16(vec![1]), vec![1, 1]).expect("integer x");
        let handle = gpu_helpers::upload_tensor(provider, &integer).expect("integer upload");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(binocdf::binocdf_builtin(
            Value::GpuTensor(handle),
            vec![Value::Num(2.0), Value::Num(0.5)],
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:BinocdfIntegerXExtension")
        );
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn binocdf_wgpu_fallback_preserves_residency_for_all_integer_classes() {
    use crate::builtins::common::test_support;

    let _accel_guard = test_support::accel_test_lock();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_integer_binocdf_storages(1) {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer x");
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
        let result = block_on(binocdf::binocdf_builtin(
            Value::GpuTensor(handle),
            vec![Value::Num(2.0), Value::Num(0.5)],
        ))
        .expect("resident integer binocdf");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("gather result");
        assert!((gathered.materialize_f64()[0] - 0.75).abs() < 1.0e-12);
    }
}

#[test]
fn wblinv_preserves_tiny_positive_probabilities() {
    let out = block_on(wblinv::wblinv_builtin(
        Value::Num(1.0e-20),
        vec![Value::Num(3.0), Value::Num(4.0)],
    ))
    .unwrap();
    match out {
        Value::Num(value) => {
            assert!(value > 0.0);
            assert_close(value, 3.0e-5, 1.0e-16);
        }
        other => panic!("expected scalar wblinv, got {other:?}"),
    }
}

#[test]
fn icdf_gates_nonfloating_extensions_and_preserves_single() {
    assert_eq!(icdf::INTEGER_CAPABILITIES[0].inputs.len(), 2);
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
fn icdf_fitted_object_form_uses_the_same_integer_admission_boundary() {
    let pd = block_on(crate::builtins::stats::summary::fitdist::fitdist_builtin(
        Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap().into(),
        Value::from("Normal"),
        Vec::new(),
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
        pd,
        Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1)),
        Vec::new(),
    ))
    .expect_err("fitted-object wide p must reject exactly");
    assert!(error.message().contains("exactly representable as double"));
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

#[test]
fn wblinv_gates_integer_roles_and_rejects_lossy_values() {
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(wblinv::wblinv_builtin(
            Value::Int(runmat_value::IntValue::U8(1)),
            Vec::new(),
        ))
        .expect_err("integer probability gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:WblinvIntegerProbabilityExtension")
        );
        let error = block_on(wblinv::wblinv_builtin(
            Value::Num(0.5),
            vec![Value::Int(runmat_value::IntValue::U8(1)), Value::Num(1.0)],
        ))
        .expect_err("integer scale gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:WblinvIntegerScaleExtension")
        );
    }
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(wblinv::wblinv_builtin(
        Value::Num(0.5),
        vec![
            Value::Num(1.0),
            Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1)),
        ],
    ))
    .expect_err("lossy shape boundary");
    assert!(error.message().contains("exactly representable as double"));
}

#[test]
fn wblinv_preserves_documented_single_output() {
    let output = block_on(wblinv::wblinv_builtin(
        Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
        vec![Value::Num(3.0), Value::Num(4.0)],
    ))
    .expect("single wblinv");
    let Value::Tensor(output) = output else {
        panic!("expected single tensor output")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
}

#[test]
fn wblinv_distinguishes_automatic_and_explicit_residency() {
    test_support::with_test_provider(|provider| {
        let source = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![0.5], vec![1, 1]).expect("probability"),
        )
        .expect("resident probability");
        runmat_accelerate::fusion_residency::mark(&source);
        let automatic = block_on(wblinv::wblinv_builtin(
            Value::GpuTensor(source.clone()),
            Vec::new(),
        ))
        .expect("automatic fallback");
        let Value::GpuTensor(output) = automatic else {
            panic!("expected resident automatic output")
        };
        assert!(
            gpu_helpers::exact_provider_for_handle(&output).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );

        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(wblinv::wblinv_builtin(Value::GpuTensor(source), Vec::new()))
            .expect_err("explicit gpu gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:WblinvExplicitGpuInputExtension")
        );
    });
}
