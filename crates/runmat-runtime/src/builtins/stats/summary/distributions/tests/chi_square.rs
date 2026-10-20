use super::*;

#[test]
fn chi2cdf_integer_roles_are_gated_and_wide_values_do_not_round() {
    assert_eq!(runmat_builtins::CHI2CDF_INTEGER_CAPABILITIES.len(), 2);
    assert!(runmat_builtins::CHI2CDF_INTEGER_CAPABILITIES
        .iter()
        .all(|capability| capability.inputs[0].classes.len() == 8));

    {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
        let x_error = block_on(chi2cdf::chi2cdf_builtin(
            Value::Int(runmat_value::IntValue::I16(3)),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
        ))
        .unwrap_err();
        assert_eq!(
            x_error.identifier(),
            Some("RunMat:compatibility:Chi2cdfIntegerXExtension")
        );
        let nu_error = block_on(chi2cdf::chi2cdf_builtin(
            Value::Num(3.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                runmat_value::IntValue::U16(5),
            )]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
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
        Value::Int(runmat_value::IntValue::I64(i64::MIN + 1)),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
    ))
    .unwrap_err();
    assert!(signed_wide_error
        .message()
        .contains("exactly representable as double"));

    let signed_minimum = block_on(chi2cdf::chi2cdf_builtin(
        Value::Int(runmat_value::IntValue::I64(i64::MIN)),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
    ))
    .expect("the signed minimum is exactly representable as double");
    assert!(matches!(signed_minimum, Value::Num(value) if value == 0.0));
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
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                Tensor::new_integer(nu_storage, vec![1, 1]).unwrap(),
            )]),
        ))
        .unwrap();
        assert!(
            matches!(out, Value::Num(value) if (value - 0.300_014_164_121_372).abs() < 1.0e-12)
        );
    }
    let upper = block_on(chi2cdf::chi2cdf_builtin(
        Value::Int(runmat_value::IntValue::U16(3)),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Int(runmat_value::IntValue::U16(5)),
            Value::from("upper"),
        ]),
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
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
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
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:Chi2cdfIntegerXExtension")
            );
        }
        let result = block_on(chi2cdf::chi2cdf_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
            runmat_value::IntValue::U16(5),
        )]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(5.0)]),
    ))
    .unwrap();
    let Value::Tensor(tensor) = out else {
        panic!("expected single tensor");
    };
    assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    assert_eq!(tensor.shape(), &[1, 2]);

    let err = block_on(chi2cdf::chi2cdf_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
            Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap(),
        )]),
    ))
    .unwrap_err();
    assert!(err.message().contains("same size"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn chi2cdf_scalar_expansion_preserves_rank_one_output_shape() {
    let rank_one_nu = Tensor::new(vec![2.0, 4.0], vec![2]).unwrap();
    let Value::Tensor(output) = block_on(chi2cdf::chi2cdf_builtin(
        Value::Tensor(Tensor::new(vec![1.0], Vec::new()).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rank_one_nu)]),
    ))
    .expect("rank-zero x expands across rank-one nu") else {
        panic!("expected tensor output");
    };
    assert_eq!(output.shape(), &[2]);

    let rank_one_x = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
    let Value::Tensor(output) = block_on(chi2cdf::chi2cdf_builtin(
        Value::Tensor(rank_one_x),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
            Tensor::new(vec![2.0], Vec::new()).unwrap(),
        )]),
    ))
    .expect("rank-zero nu expands across rank-one x") else {
        panic!("expected tensor output");
    };
    assert_eq!(output.shape(), &[2]);
}
