use super::*;

#[test]
fn binocdf_boundaries_and_invalid_parameters_match_distribution_contract() {
    let below = block_on(binocdf::binocdf_builtin(
        Value::Num(-1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(10.0),
            Value::Num(0.5),
        ]),
    ))
    .unwrap();
    assert_eq!(below, Value::Num(0.0));

    let above = block_on(binocdf::binocdf_builtin(
        Value::Num(10.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(10.0),
            Value::Num(0.5),
        ]),
    ))
    .unwrap();
    assert_eq!(above, Value::Num(1.0));

    let always_zero = block_on(binocdf::binocdf_builtin(
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(10.0),
            Value::Num(0.0),
        ]),
    ))
    .unwrap();
    assert_eq!(always_zero, Value::Num(1.0));

    let impossible = block_on(binocdf::binocdf_builtin(
        Value::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(10.5),
            Value::Num(0.5),
        ]),
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
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::Num(0.5),
            ]),
        ))
        .expect("integer x");
        assert!(matches!(x, Value::Num(value) if (value - 0.75).abs() < 1.0e-12));

        let n = block_on(binocdf::binocdf_builtin(
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                mirrorless_int_tensor(storage.clone(), vec![1, 1]),
                Value::Num(0.5),
            ]),
        ))
        .expect("integer n");
        assert!(matches!(n, Value::Num(value) if (value - 0.5).abs() < 1.0e-12));

        let p = block_on(binocdf::binocdf_builtin(
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                mirrorless_int_tensor(storage, vec![1, 1]),
            ]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(0.5),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        integer_x.identifier(),
        Some("RunMat:compatibility:BinocdfIntegerXExtension")
    );
    let integer_n = block_on(binocdf::binocdf_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            mirrorless_int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
            Value::Num(0.5),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        integer_n.identifier(),
        Some("RunMat:compatibility:BinocdfIntegerTrialsExtension")
    );
    let integer_p = block_on(binocdf::binocdf_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            mirrorless_int_tensor(IntegerStorage::U32(vec![1]), vec![1, 1]),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        integer_p.identifier(),
        Some("RunMat:compatibility:BinocdfIntegerProbabilityExtension")
    );
    let logical = block_on(binocdf::binocdf_builtin(
        Value::Bool(true),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(0.5),
        ]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(0.5),
        ]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(0.5),
        ]),
    ))
    .unwrap_err();
    assert!(error.message().contains("exactly representable as double"));

    let exact = block_on(binocdf::binocdf_builtin(
        mirrorless_int_tensor(IntegerStorage::U64(vec![1_u64 << 54]), vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(0.5),
        ]),
    ))
    .expect("2^54 is exactly representable as double");
    assert!(matches!(exact, Value::Num(value) if value == 1.0));
}

#[test]
fn binocdf_gpu_fallback_preserves_residency_precision_and_integer_guard_order() {
    use crate::builtins::common::test_support;

    crate::builtins::common::test_support::with_test_provider(|provider| {
        let single = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single input");
        let handle = gpu_helpers::upload_tensor(provider, &single).expect("single upload");
        let result = block_on(binocdf::binocdf_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::Num(0.5),
            ]),
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
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::Num(0.5),
            ]),
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
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::Num(0.5),
            ]),
        ))
        .expect("resident integer binocdf");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("gather result");
        assert!((gathered.materialize_f64()[0] - 0.75).abs() < 1.0e-12);
    }
}
