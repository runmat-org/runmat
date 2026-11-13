use super::*;

#[test]
fn fitdist_normal_fits_and_evaluates_object_methods() {
    let pd = block_on(fitdist_builtin(
        vec_tensor(&[1.0, 2.0, 3.0]),
        Value::String("Normal".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let object = object(pd.clone());
    assert_eq!(
        object.properties().get("DistributionName"),
        Some(&Value::String("Normal".into()))
    );
    let values = numeric_vector_property(FITDIST_NAME, &object, "ParameterValues").unwrap();
    assert!((values[0] - 2.0).abs() < 1.0e-12);
    assert!((values[1] - (2.0_f64 / 3.0).sqrt()).abs() < 1.0e-12);

    let density = block_on(pdf_builtin(
        pd.clone(),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let Value::Num(density) = density else {
        panic!("expected scalar pdf");
    };
    assert!(density > 0.0);

    let p = block_on(cdf_builtin(
        pd.clone(),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert_eq!(p, Value::Num(0.5));

    let upper = block_on(cdf_builtin(
        pd.clone(),
        Value::Num(10.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "upper".into(),
        )]),
    ))
    .unwrap();
    assert!(matches!(upper, Value::Num(value) if value > 0.0 && value < 1.0e-20));

    let fitted = admit_icdf_probability_distribution(&pd).unwrap();
    let x = block_on(icdf_probability_distribution(fitted, Value::Num(0.5), None)).unwrap();
    assert_eq!(x, Value::Num(2.0));
}

#[test]
fn fitdist_logical_observations_and_frequency_retain_existing_numeric_admission() {
    let data =
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![0, 1, 1], vec![3, 1]).unwrap());
    let frequency =
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap());
    let pd = block_on(fitdist_builtin(
        data,
        Value::from("Normal"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Frequency"),
            frequency,
        ]),
    ))
    .expect("logical observations and counts are numeric 0/1 at the fitting boundary");
    let object = object(pd);
    let values = numeric_vector_property(FITDIST_NAME, &object, "ParameterValues").unwrap();
    assert!((values[0] - 0.5).abs() < 1.0e-12);
}

#[test]
fn fitdist_accepts_typed_integer_sample_and_eval_points() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let pd = block_on(fitdist_builtin(
        int_vec_tensor(IntegerStorage::I16(vec![1, 2, 3]), 3),
        Value::String("Normal".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let object = object(pd.clone());
    let values = numeric_vector_property(FITDIST_NAME, &object, "ParameterValues").unwrap();
    assert!((values[0] - 2.0).abs() < 1.0e-12);

    let density = block_on(pdf_builtin(
        pd,
        mirrorless_int_tensor(IntegerStorage::U16(vec![2, 3]), vec![2, 1]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match density {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2, 1]);
            assert!(tensor
                .materialize_f64()
                .iter()
                .all(|value| value.is_finite()));
        }
        other => panic!("expected tensor density, got {other:?}"),
    }
}

#[test]
fn fitdist_integer_data_and_frequency_are_separate_gated_extensions() {
    for storage in all_fitdist_integer_storages() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(fitdist_builtin(
            mirrorless_int_tensor(storage.clone(), vec![3, 1]),
            Value::String("Normal".into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("integer observations");
        block_on(fitdist_builtin(
            vec_tensor(&[1.0, 2.0, 3.0]),
            Value::String("Normal".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Frequency"),
                mirrorless_int_tensor(storage, vec![3, 1]),
            ]),
        ))
        .expect("integer Frequency");
    }

    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let data_error = block_on(fitdist_builtin(
        mirrorless_int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]),
        Value::String("Normal".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        data_error.identifier(),
        Some("RunMat:compatibility:FitdistIntegerDataExtension")
    );
    let frequency_error = block_on(fitdist_builtin(
        vec_tensor(&[1.0, 2.0, 3.0]),
        Value::String("Normal".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Frequency"),
            mirrorless_int_tensor(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        frequency_error.identifier(),
        Some("RunMat:compatibility:FitdistIntegerFrequencyExtension")
    );
}

#[test]
fn fitdist_rejects_inexact_integer_boundaries_and_fractional_frequency() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let exact_wide = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 54]), vec![1, 1])
        .expect("exact wide integer");
    ensure_exact_fitdist_integer_boundary(&exact_wide, "test")
        .expect("wide powers of two remain exactly representable");
    let wide = block_on(fitdist_builtin(
        mirrorless_int_tensor(
            IntegerStorage::U64(vec![1, 2, (1_u64 << 53) + 1]),
            vec![3, 1],
        ),
        Value::String("Normal".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(wide.message().contains("exactly representable as double"));

    for invalid in [1.5, f64::NAN, f64::INFINITY, -1.0] {
        let error = block_on(fitdist_builtin(
            vec_tensor(&[1.0, 2.0]),
            Value::String("Normal".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Frequency"),
                vec_tensor(&[0.0, invalid]),
            ]),
        ))
        .unwrap_err();
        assert!(error.message().contains("integer counts"));
    }
}

#[test]
fn fitdist_strict_mode_gates_resident_fallback_before_gather() {
    use crate::builtins::common::test_support;

    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1])
            .expect("integer observations");
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

        let floating = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("observations");
        let floating_handle =
            gpu_helpers::upload_tensor(provider, &floating).expect("floating upload");
        let resident_error = block_on(fitdist_builtin(
            Value::GpuTensor(floating_handle.clone()),
            Value::String("Normal".into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(
            resident_error.identifier(),
            Some("RunMat:compatibility:FitdistResidentFallbackExtension")
        );
        assert!(
            runmat_accelerate_api::provider_for_handle(&floating_handle).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        let resident_option_error = block_on(fitdist_builtin(
            vec_tensor(&[1.0, 2.0, 3.0]),
            Value::String("Normal".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Frequency"),
                Value::GpuTensor(floating_handle.clone()),
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            resident_option_error.identifier(),
            Some("RunMat:compatibility:FitdistResidentFallbackExtension")
        );

        let error = block_on(fitdist_builtin(
            Value::GpuTensor(handle.clone()),
            Value::String("Normal".into()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FitdistIntegerDataExtension")
        );
        assert!(
            runmat_accelerate_api::provider_for_handle(&handle).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );

        let frequency = Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1])
            .expect("integer Frequency");
        let frequency_handle =
            gpu_helpers::upload_tensor(provider, &frequency).expect("integer upload");
        let error = block_on(fitdist_builtin(
            vec_tensor(&[1.0, 2.0, 3.0]),
            Value::String("Normal".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Frequency"),
                Value::GpuTensor(frequency_handle.clone()),
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FitdistIntegerFrequencyExtension")
        );
        assert!(
            runmat_accelerate_api::provider_for_handle(&frequency_handle).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
    });
}

#[test]
fn fitdist_random_shape_parser_ignores_all_typed_mirrors() {
    let storages = [
        IntegerStorage::I8(vec![2, 3]),
        IntegerStorage::I16(vec![2, 3]),
        IntegerStorage::I32(vec![2, 3]),
        IntegerStorage::I64(vec![2, 3]),
        IntegerStorage::U8(vec![2, 3]),
        IntegerStorage::U16(vec![2, 3]),
        IntegerStorage::U32(vec![2, 3]),
        IntegerStorage::U64(vec![2, 3]),
    ];

    for storage in storages {
        assert_eq!(
            block_on(parse_shape_args(
                crate::call::FunctionArgumentView::from_slice(&[mirrorless_int_tensor(
                    storage,
                    vec![2, 1]
                )])
            ))
            .unwrap(),
            vec![2, 3]
        );
    }
}

#[test]
fn fitdist_reads_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let pd = block_on(fitdist_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]),
        Value::String("Exponential".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Frequency".into()),
            mirrorless_int_tensor(IntegerStorage::U8(vec![1, 3, 2]), vec![3, 1]),
        ]),
    ))
    .unwrap();
    let object = object(pd.clone());
    let values = numeric_vector_property(FITDIST_NAME, &object, "ParameterValues").unwrap();
    assert!((values[0] - (13.0 / 6.0)).abs() < 1.0e-12);

    let mut typed_object = object;
    typed_object.replace_property_unbounded(
        "ParameterValues".to_string(),
        poisoned_int_tensor(IntegerStorage::U16(vec![7]), vec![1, 1]),
    );
    assert_eq!(
        numeric_vector_property(FITDIST_NAME, &typed_object, "ParameterValues").unwrap(),
        vec![7.0]
    );
}

#[test]
fn fitdist_sample_length_uses_typed_integer_storage_not_mirror() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let pd = block_on(fitdist_builtin(
        mirrorless_int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]),
        Value::String("Normal".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let object = object(pd);
    let values = numeric_vector_property(FITDIST_NAME, &object, "ParameterValues").unwrap();
    assert!((values[0] - 2.0).abs() < 1.0e-12);
}

#[test]
fn fitdist_frequency_and_range_validation() {
    let pd = object(
        block_on(fitdist_builtin(
            vec_tensor(&[1.0, 2.0]),
            Value::String("Exponential".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Frequency".into()),
                Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
            ]),
        ))
        .unwrap(),
    );
    let values = numeric_vector_property(FITDIST_NAME, &pd, "ParameterValues").unwrap();
    assert!((values[0] - 1.75).abs() < 1.0e-12);

    let err = block_on(fitdist_builtin(
        vec_tensor(&[-1.0, 2.0]),
        Value::String("Gamma".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("outside the supported range"));
}

#[test]
fn fitdist_gamma_weibull_and_poisson_smoke() {
    let _rng_guard = crate::builtins::common::random::test_guard();
    let gamma = block_on(fitdist_builtin(
        vec_tensor(&[1.0, 2.0, 4.0, 8.0]),
        Value::String("Gamma".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert!(matches!(
        block_on(cdf_builtin(gamma, Value::Num(2.0), crate::builtins::common::test_support::builtin_arguments(Vec::new()))).unwrap(),
        Value::Num(value) if value.is_finite()
    ));

    let weibull = block_on(fitdist_builtin(
        vec_tensor(&[1.0, 2.0, 3.0, 5.0, 8.0]),
        Value::String("Weibull".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert!(matches!(
        block_on(pdf_builtin(weibull, Value::Num(2.0), crate::builtins::common::test_support::builtin_arguments(Vec::new()))).unwrap(),
        Value::Num(value) if value.is_finite() && value >= 0.0
    ));

    let poisson = block_on(fitdist_builtin(
        vec_tensor(&[0.0, 1.0, 1.0, 2.0, 3.0]),
        Value::String("Poisson".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let samples = block_on(random_builtin(poisson, vec![Value::Num(2.0)])).unwrap();
    match samples {
        Value::Tensor(tensor) => assert_eq!(tensor.shape(), &[2, 2]),
        other => panic!("expected tensor, got {other:?}"),
    }
}
