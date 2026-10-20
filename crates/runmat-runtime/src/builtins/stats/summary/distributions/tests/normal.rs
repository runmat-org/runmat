use super::*;

#[test]
fn normal_distribution_scalar_values() {
    let pdf = block_on(normpdf::normpdf_builtin(
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match pdf {
        Value::Num(value) => assert_close(value, INV_SQRT_2PI, 1e-12),
        other => panic!("expected scalar pdf, got {other:?}"),
    }
    let cdf = block_on(normcdf::normcdf_builtin(
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match cdf {
        Value::Num(value) => assert_close(value, 0.5, 1e-12),
        other => panic!("expected scalar cdf, got {other:?}"),
    }
    let inv = block_on(norminv::norminv_builtin(
        Value::Num(0.5),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::Num(1.0),
        ]),
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1, 2]);
            assert_close(tensor.materialize_f64()[0], 0.5, 1e-12);
            assert_close(tensor.materialize_f64()[1], 0.841_344_746_068_543, 1e-12);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn normal_distribution_canonicalizes_low_rank_inputs_through_scalar_broadcasting() {
    let scalar = block_on(normcdf::normcdf_builtin(
        Value::Tensor(Tensor::new(vec![0.0], Vec::new()).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert_eq!(scalar, Value::Num(0.5));

    let Value::Tensor(vector) = block_on(normcdf::normcdf_builtin(
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::Num(1.0),
        ]),
    ))
    .unwrap() else {
        panic!("rank-one input must produce a broadcast tensor");
    };
    assert_eq!(vector.shape(), &[1, 2]);
    assert_close(vector.materialize_f64()[0], 0.5, 1e-12);
    assert_close(vector.materialize_f64()[1], 0.841_344_746_068_543, 1e-12);
}

#[test]
fn normal_distribution_integer_roles_are_independently_gated() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let x = block_on(normcdf::normcdf_builtin(
        Value::Int(runmat_value::IntValue::I16(0)),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        x.identifier(),
        Some("RunMat:compatibility:NormcdfIntegerXExtension")
    );
    let mu = block_on(normcdf::normcdf_builtin(
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
            runmat_value::IntValue::I16(0),
        )]),
    ))
    .unwrap_err();
    assert_eq!(
        mu.identifier(),
        Some("RunMat:compatibility:NormcdfIntegerMuExtension")
    );
    let sigma = block_on(normcdf::normcdf_builtin(
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::Int(runmat_value::IntValue::U16(1)),
        ]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            mirrorless_int_tensor(IntegerStorage::I16(vec![0]), vec![1, 1]),
            mirrorless_int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
        ]),
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
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        match out {
            Value::Num(value) => assert_close(value, 0.5, 1.0e-12),
            other => panic!("expected double scalar, got {other:?}"),
        }
    }

    let wide = mirrorless_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]);
    let error = block_on(norminv::norminv_builtin(
        wide,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
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
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
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
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .unwrap();
    match cdf {
        Value::Num(value) => assert_close(value, 0.841_344_746_068_543, 1e-12),
        other => panic!("expected scalar cdf, got {other:?}"),
    }

    let upper = block_on(normcdf::normcdf_builtin(
        Value::Num(8.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::Num(1.0),
            Value::from("upper"),
        ]),
    ))
    .unwrap();
    match upper {
        Value::Num(value) => assert_close(value, 6.220_960_574_271_784e-16, 1e-28),
        other => panic!("expected scalar upper cdf, got {other:?}"),
    }

    let pdf = block_on(normpdf::normpdf_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .unwrap();
    match pdf {
        Value::Num(value) => assert_close(value, INV_SQRT_2PI * (-0.5f64).exp(), 1e-12),
        other => panic!("expected scalar pdf, got {other:?}"),
    }

    let inv = block_on(norminv::norminv_builtin(
        Value::Num(0.5),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
    ))
    .unwrap();
    match inv {
        Value::Num(value) => assert_close(value, 2.0, 1e-10),
        other => panic!("expected scalar inv, got {other:?}"),
    }
}

#[test]
fn normal_distribution_rejects_invalid_sigma_with_nan_outputs() {
    let pdf = block_on(normpdf::normpdf_builtin(
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::Num(0.0),
        ]),
    ))
    .unwrap();
    assert!(matches!(pdf, Value::Num(value) if value.is_nan()));

    let inv = block_on(norminv::norminv_builtin(
        Value::Num(0.5),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::Num(0.0),
        ]),
    ))
    .unwrap();
    assert!(matches!(inv, Value::Num(value) if value.is_nan()));

    let cdf = block_on(normcdf::normcdf_builtin(
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Num(0.0),
        ]),
    ))
    .unwrap();
    assert_eq!(cdf, Value::Num(0.0));
}
