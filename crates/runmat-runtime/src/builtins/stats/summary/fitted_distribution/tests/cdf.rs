use super::*;

#[test]
fn cdf_classifies_all_integer_input_positions_as_runmat_extensions() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_cdf_integer_storages(1) {
        let x = block_on(cdf_builtin(
            Value::String("Normal".into()),
            mirrorless_int_tensor(storage.clone(), vec![1, 1]),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                Value::Num(1.0),
            ]),
        ))
        .expect("integer x");
        assert!(matches!(x, Value::Num(value) if (value - 0.841_344_746_068_543).abs() < 1.0e-12));

        let parameter = block_on(cdf_builtin(
            Value::String("Poisson".into()),
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(vec![mirrorless_int_tensor(
                storage,
                vec![1, 1],
            )]),
        ))
        .expect("integer parameter");
        assert!(
            matches!(parameter, Value::Num(value) if (value - (-1.0_f64).exp()).abs() < 1.0e-12)
        );
    }

    let pd = block_on(fitdist_builtin(
        vec_tensor(&[0.0, 1.0, 2.0]),
        Value::String("Normal".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("fitted normal");
    let object_x = block_on(cdf_builtin(
        pd,
        mirrorless_int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("integer object evaluation point");
    assert_eq!(object_x, Value::Num(0.5));
}

#[test]
fn cdf_compatibility_mode_rejects_integer_and_logical_extensions() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer_x = block_on(cdf_builtin(
        Value::String("Normal".into()),
        mirrorless_int_tensor(IntegerStorage::I8(vec![1]), vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::Num(1.0),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        integer_x.identifier(),
        Some("RunMat:compatibility:CdfIntegerXExtension")
    );
    let pd = block_on(fitdist_builtin(
        vec_tensor(&[0.0, 1.0, 2.0]),
        Value::String("Normal".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("fitted normal");
    let object_integer_x = block_on(cdf_builtin(
        pd,
        mirrorless_int_tensor(IntegerStorage::I16(vec![1]), vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        object_integer_x.identifier(),
        Some("RunMat:compatibility:CdfIntegerXExtension")
    );
    let integer_parameter = block_on(cdf_builtin(
        Value::String("Poisson".into()),
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![mirrorless_int_tensor(
            IntegerStorage::U16(vec![1]),
            vec![1, 1],
        )]),
    ))
    .unwrap_err();
    assert_eq!(
        integer_parameter.identifier(),
        Some("RunMat:compatibility:CdfIntegerParametersExtension")
    );
    let logical = block_on(cdf_builtin(
        Value::String("Normal".into()),
        Value::Bool(true),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::Num(1.0),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        logical.identifier(),
        Some("RunMat:compatibility:CdfLogicalInputExtension")
    );
}

#[test]
fn cdf_broadcasts_parameters_supports_upper_and_preserves_single() {
    let result = block_on(cdf_builtin(
        Value::String("Normal".into()),
        Value::Tensor(Tensor::from_f32(vec![0.0, 2.0], vec![2, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Num(1.0),
            Value::String("upper".into()),
        ]),
    ))
    .expect("broadcast upper cdf");
    let Value::Tensor(result) = result else {
        panic!("single output must retain its class");
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);
    assert_eq!(result.shape(), &[2, 1]);
    let values = result.materialize_f64();
    assert!((values[0] - 0.5).abs() < f64::from(f32::EPSILON));
    assert!((values[1] - 0.158_655_253_931_457).abs() < f64::from(f32::EPSILON));

    let tail = block_on(cdf_builtin(
        Value::String("Exponential".into()),
        Value::Num(1000.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::String("upper".into()),
        ]),
    ))
    .expect("upper tail");
    assert_eq!(tail, Value::Num(0.0));
}

#[test]
fn cdf_rejects_inexact_wide_integers() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(cdf_builtin(
        Value::String("Normal".into()),
        mirrorless_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::Num(1.0),
        ]),
    ))
    .unwrap_err();
    assert!(error.message().contains("exactly representable as double"));
    for storage in [
        IntegerStorage::U64(vec![1_u64 << 54]),
        IntegerStorage::I64(vec![i64::MIN]),
    ] {
        let exact = block_on(cdf_builtin(
            Value::String("Normal".into()),
            mirrorless_int_tensor(storage, vec![1, 1]),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                Value::Num(1.0),
            ]),
        ));
        assert!(
            exact.is_ok(),
            "wide but exactly representable integer must pass"
        );
    }
}

#[test]
fn cdf_gpu_fallback_preserves_residency_precision_and_guard_order() {
    use crate::builtins::common::test_support;

    test_support::with_test_provider(|provider| {
        let single = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single input");
        let handle = gpu_helpers::upload_tensor(provider, &single).expect("single upload");
        let result = block_on(cdf_builtin(
            Value::String("Normal".into()),
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                Value::Num(1.0),
            ]),
        ))
        .expect("resident cdf");
        let Value::GpuTensor(result_handle) = &result else {
            panic!("expected resident output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(result_handle),
            Some(ProviderPrecision::F32)
        );
        let gathered = test_support::gather(result).expect("gather result");
        assert_eq!(gathered.numeric_dtype(), NumericDType::F32);

        let integer =
            Tensor::new_integer(IntegerStorage::I16(vec![1]), vec![1, 1]).expect("integer x");
        let handle = gpu_helpers::upload_tensor(provider, &integer).expect("integer upload");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(cdf_builtin(
            Value::String("Normal".into()),
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                Value::Num(1.0),
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:CdfIntegerXExtension")
        );
    });
}
