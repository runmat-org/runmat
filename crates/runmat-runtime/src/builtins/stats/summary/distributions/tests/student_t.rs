use super::*;

#[test]
fn student_t_distribution_scalar_values() {
    let pdf = block_on(tpdf::tpdf_builtin(
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .unwrap();
    match pdf {
        Value::Num(value) => assert_close(value, std::f64::consts::FRAC_1_PI, 1e-12),
        other => panic!("expected scalar pdf, got {other:?}"),
    }

    let cdf = block_on(tcdf::tcdf_builtin(
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(10.0)]),
    ))
    .unwrap();
    match cdf {
        Value::Num(value) => assert_close(value, 0.5, 1e-12),
        other => panic!("expected scalar cdf, got {other:?}"),
    }

    let inv = block_on(tinv::tinv_builtin(
        Value::Num(0.95),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(50.0)]),
    ))
    .unwrap();
    match inv {
        Value::Num(value) => assert_close(value, 1.675_905, 1e-6),
        other => panic!("expected scalar inv, got {other:?}"),
    }
}

#[test]
fn student_t_distribution_broadcasts_and_upper_tail() {
    let x = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap());
    let nu = Value::Tensor(Tensor::new(vec![1.0, 5.0, f64::INFINITY], vec![1, 3]).unwrap());
    let out = block_on(tcdf::tcdf_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![nu]),
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1, 3]);
            assert_close(tensor.materialize_f64()[0], 0.5, 1e-12);
            assert_close(tensor.materialize_f64()[1], 0.818_391_266, 1e-9);
            assert_close(tensor.materialize_f64()[2], 0.977_249_868, 1e-9);
        }
        other => panic!("expected tensor cdf, got {other:?}"),
    }

    let upper = block_on(tcdf::tcdf_builtin(
        Value::Num(10.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(99.0),
            Value::from("upper"),
        ]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
    ))
    .unwrap();
    assert!(matches!(single, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32));

    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(tpdf::tpdf_builtin(
            mirrorless_int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
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
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
        ))
        .unwrap_err();
        assert_eq!(
            integer_x.identifier(),
            Some("RunMat:compatibility:TcdfIntegerXExtension")
        );
        let integer_nu = block_on(tcdf::tcdf_builtin(
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![mirrorless_int_tensor(
                IntegerStorage::U16(vec![5]),
                vec![1, 1],
            )]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![mirrorless_int_tensor(
            IntegerStorage::U64(vec![5]),
            vec![1, 1],
        )]),
    ))
    .expect("exact integer tcdf inputs");
    assert!(matches!(exact, Value::Num(value) if value > 0.8 && value < 0.9));

    let inexact = block_on(tcdf::tcdf_builtin(
        mirrorless_int_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
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
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
        ))
        .expect_err("integer p must gate");
        assert_eq!(
            integer_p.identifier(),
            Some("RunMat:compatibility:TinvIntegerProbabilityExtension")
        );
        let integer_nu = block_on(tinv::tinv_builtin(
            Value::Num(0.5),
            crate::builtins::common::test_support::builtin_arguments(vec![mirrorless_int_tensor(
                IntegerStorage::U16(vec![5]),
                vec![1, 1],
            )]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![mirrorless_int_tensor(
            IntegerStorage::U16(vec![5]),
            vec![1, 1],
        )]),
    ))
    .expect("exact integer tinv inputs");
    assert!(matches!(exact, Value::Num(value) if value.is_infinite() && value.is_sign_positive()));

    let inexact = block_on(tinv::tinv_builtin(
        Value::Num(0.5),
        crate::builtins::common::test_support::builtin_arguments(vec![mirrorless_int_tensor(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            vec![1, 1],
        )]),
    ))
    .expect_err("wide integer nu must reject");
    assert!(inexact.message().contains("exactly representable"));
}

#[test]
fn tinv_preserves_native_single_output_precision() {
    let output = block_on(tinv::tinv_builtin(
        Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
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
    let builtin = &runmat_builtins::TINV_CATALOG_ENTRY;
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
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap(),
        )]),
    ))
    .unwrap_err();
    assert!(err.message().contains("tpdf"));
    assert_eq!(err.identifier(), Some("RunMat:tpdf:InvalidArgument"));

    let out = block_on(tcdf::tcdf_builtin(
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(-1.0)]),
    ))
    .unwrap();
    assert!(matches!(out, Value::Num(value) if value.is_nan()));
}

#[test]
fn student_t_distribution_extreme_tails_remain_representable() {
    let lower = block_on(tinv::tinv_builtin(
        Value::Num(1.0e-20),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::from("upper"),
        ]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0e12)]),
    ))
    .unwrap();
    match pdf {
        Value::Num(value) => assert_close(value, INV_SQRT_2PI * (-0.5f64).exp(), 1e-12),
        other => panic!("expected scalar pdf, got {other:?}"),
    }

    let cdf = block_on(tcdf::tcdf_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0e12)]),
    ))
    .unwrap();
    match cdf {
        Value::Num(value) => assert_close(value, 0.841_344_746_068_543, 1e-12),
        other => panic!("expected scalar cdf, got {other:?}"),
    }

    let inv = block_on(tinv::tinv_builtin(
        Value::Num(0.975),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0e12)]),
    ))
    .unwrap();
    match inv {
        Value::Num(value) => assert_close(value, 1.959_963_984_540_053_8, 1e-12),
        other => panic!("expected scalar inv, got {other:?}"),
    }
}
