use super::super::flagged::scalar_flag;
use super::*;

#[test]
fn singleton_dimension_retains_following_selector_and_missing_policy() {
    let input =
        || Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, f64::NAN, 6.0], vec![2, 3]).unwrap());
    let result = block_on(kurtosis::kurtosis_builtin(
        input(),
        vec![
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
            Value::from("all"),
            Value::from("includenan"),
            Value::from("omitmissing"),
        ],
    ))
    .unwrap();
    let expected = block_on(kurtosis::kurtosis_builtin(
        input(),
        vec![Value::Num(1.0), Value::from("all")],
    ))
    .unwrap();
    let (values, shape) = tensor_values(result);
    assert_eq!(shape, vec![1, 1]);
    // Five nonmissing observations have second moment2.96 and fourth17.4752.
    assert_close(values[0], 17.4752 / (2.96 * 2.96));
    assert_eq!((values, shape), tensor_values(expected));

    let along_second = block_on(kurtosis::kurtosis_builtin(
        input(),
        vec![
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap();
    assert_eq!(tensor_values(along_second).1, vec![2, 1]);
}

#[test]
fn floating_dimensions_and_small_scalar_flags_keep_distinct_tolerances() {
    let input = || Value::Tensor(Tensor::new(vec![1.0, 2.0, 10.0], vec![1, 3]).unwrap());
    let empty = || Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
    let mut dimensions = vec![1.0 + f64::EPSILON];
    if usize::BITS < 64 {
        dimensions.push(usize::MAX as f64);
    }
    for dimension in dimensions {
        let output = block_on(mad::mad_builtin(
            input(),
            vec![empty(), Value::Num(dimension)],
        ))
        .unwrap();
        let (values, shape) = tensor_values(output);
        assert_eq!(shape, vec![1, 3]);
        assert_eq!(values, vec![0.0; 3]);
    }
    let output = block_on(mad::mad_builtin(
        input(),
        vec![Value::Num(1e-13), Value::from("all")],
    ))
    .unwrap();
    assert_close(tensor_values(output).0[0], 34.0 / 9.0);
}

#[test]
fn scalar_flag_accepts_typed_integer_tensor_scalars() {
    let one =
        Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![1]), vec![1, 1]).expect("flag");
    assert_eq!(scalar_flag("mad", &Value::Tensor(one)).unwrap(), Some(1));

    let two =
        Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![2]), vec![1, 1]).expect("flag");
    assert!(scalar_flag("mad", &Value::Tensor(two)).is_err());
}

#[test]
fn mad_supports_mean_default_and_median_flag_modes() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 10.0], vec![1, 3]).unwrap());
    let default_mad = block_on(mad::mad_builtin(
        x.clone(),
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(tensor_values(default_mad).0[0], 34.0 / 9.0);

    let mean_mad = block_on(mad::mad_builtin(
        x.clone(),
        vec![
            Value::Num(0.0),
            Value::CharArray(runmat_value::CharArray::new_row("all")),
        ],
    ))
    .unwrap();
    assert_close(tensor_values(mean_mad).0[0], 34.0 / 9.0);

    let median_mad = block_on(mad::mad_builtin(
        x,
        vec![
            Value::Num(1.0),
            Value::CharArray(runmat_value::CharArray::new_row("all")),
        ],
    ))
    .unwrap();
    assert_close(tensor_values(median_mad).0[0], 1.0);
}

#[test]
fn empty_flag_placeholder_preserves_following_dimension() {
    let input = Value::Tensor(Tensor::new(vec![1.0, 2.0, 10.0], vec![1, 3]).unwrap());
    let empty = Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
    let result = block_on(mad::mad_builtin(input, vec![empty, Value::Num(2.0)])).unwrap();
    let (values, shape) = tensor_values(result);
    assert_eq!(shape, vec![1, 1]);
    assert_close(values[0], 34.0 / 9.0);
}

#[test]
fn flagged_statistics_keep_single_output_class_and_bound_far_dimensions() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let single = || Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    for output in [
        block_on(mad::mad_builtin(single(), Vec::new())).unwrap(),
        block_on(skewness::skewness_builtin(single(), Vec::new())).unwrap(),
        block_on(kurtosis::kurtosis_builtin(single(), Vec::new())).unwrap(),
    ] {
        let Value::Tensor(tensor) = output else {
            panic!("a single result must retain its tensor carrier")
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        assert_eq!(tensor.shape(), [1, 1]);
    }

    let data = || Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    let far_dim = || {
        vec![
            Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
            Value::Int(IntValue::U64(1_000_000_000)),
        ]
    };
    let mad_result = block_on(mad::mad_builtin(data(), far_dim())).unwrap();
    let (values, shape) = tensor_values(mad_result);
    assert_eq!(shape, vec![3, 1]);
    assert_eq!(values, vec![0.0; 3]);
    for output in [
        block_on(skewness::skewness_builtin(data(), far_dim())).unwrap(),
        block_on(kurtosis::kurtosis_builtin(data(), far_dim())).unwrap(),
    ] {
        let (values, shape) = tensor_values(output);
        assert_eq!(shape, vec![3, 1]);
        assert!(values.iter().all(|value| value.is_nan()));
    }
}

#[test]
fn mad_integer_data_and_controls_follow_compatibility_mode() {
    let integer_data = || int_tensor(IntegerStorage::I16(vec![1, 4, 9]), vec![1, 3]);
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(mad::mad_builtin(integer_data(), Vec::new())).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:MadIntegerDataExtension")
        );
        let floating = Value::Tensor(Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap());
        let error = block_on(mad::mad_builtin(
            floating,
            vec![Value::Int(IntValue::U8(1))],
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:MadTypedIntegerControlExtension")
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let out = block_on(mad::mad_builtin(
            integer_data(),
            vec![Value::Num(1.0), Value::from("all")],
        ))
        .unwrap();
        assert_close(tensor_values(out).0[0], 3.0);
    }
}

#[test]
fn remaining_descriptive_integer_extensions_follow_compatibility_mode() {
    let integer_data = || int_tensor(IntegerStorage::I16(vec![1, 4, 9]), vec![1, 3]);
    let floating_data = || Value::Tensor(Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap());
    let integer_dim = || Value::Int(IntValue::U8(2));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

    assert_extension_error(
        block_on(geomean::geomean_builtin_source(integer_data(), Vec::new())),
        "RunMat:compatibility:GeomeanIntegerDataExtension",
    );
    assert_extension_error(
        block_on(geomean::geomean_builtin_source(
            floating_data(),
            vec![integer_dim()],
        )),
        "RunMat:compatibility:GeomeanTypedIntegerControlExtension",
    );
    assert_extension_error(
        block_on(harmmean::harmmean_builtin_source(
            integer_data(),
            Vec::new(),
        )),
        "RunMat:compatibility:HarmmeanIntegerDataExtension",
    );
    assert_extension_error(
        block_on(harmmean::harmmean_builtin_source(
            floating_data(),
            vec![integer_dim()],
        )),
        "RunMat:compatibility:HarmmeanTypedIntegerControlExtension",
    );
    assert_extension_error(
        block_on(rms::rms_builtin_source(integer_data(), Vec::new())),
        "RunMat:compatibility:RmsIntegerDataExtension",
    );
    assert_extension_error(
        block_on(skewness::skewness_builtin(integer_data(), Vec::new())),
        "RunMat:compatibility:SkewnessIntegerDataExtension",
    );
    assert_extension_error(
        block_on(skewness::skewness_builtin(
            floating_data(),
            vec![integer_dim()],
        )),
        "RunMat:compatibility:SkewnessTypedIntegerControlExtension",
    );
    assert_extension_error(
        block_on(kurtosis::kurtosis_builtin(integer_data(), Vec::new())),
        "RunMat:compatibility:KurtosisIntegerDataExtension",
    );
    assert_extension_error(
        block_on(kurtosis::kurtosis_builtin(
            floating_data(),
            vec![integer_dim()],
        )),
        "RunMat:compatibility:KurtosisTypedIntegerControlExtension",
    );
    assert_extension_error(
        block_on(rmse::rmse_builtin_source(
            integer_data(),
            floating_data(),
            Vec::new(),
        )),
        "RunMat:compatibility:RmseIntegerDataExtension",
    );
    assert_extension_error(
        block_on(rmse::rmse_builtin_source(
            floating_data(),
            floating_data(),
            vec![Value::from("Weights"), integer_data()],
        )),
        "RunMat:compatibility:RmseIntegerWeightsExtension",
    );
    assert_extension_error(
        block_on(tabulate::tabulate_builtin(integer_data())),
        "RunMat:compatibility:TabulateIntegerDataExtension",
    );
}

#[test]
fn descriptive_gpu_extensions_reject_before_gather() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 4.0, 9.0], vec![3, 1]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input).unwrap();
        let vecdim = crate::builtins::common::gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(),
        )
        .unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        assert_extension_error(
            block_on(skewness::skewness_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::Num(1.0), Value::from("all")],
            )),
            "RunMat:compatibility:SkewnessGpuMultiAxisExtension",
        );
        assert_extension_error(
            block_on(kurtosis::kurtosis_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::Num(1.0), Value::GpuTensor(vecdim.clone())],
            )),
            "RunMat:compatibility:KurtosisGpuMultiAxisExtension",
        );
        assert_extension_error(
            block_on(tabulate::tabulate_builtin(Value::GpuTensor(handle.clone()))),
            "RunMat:compatibility:TabulateGpuInputExtension",
        );
        drop(handle);
        drop(vecdim);
    });
}

#[test]
fn skewness_and_kurtosis_match_biased_moments() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
    let skew = block_on(skewness::skewness_builtin(
        x.clone(),
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(tensor_values(skew).0[0], 0.0);

    let kurt = block_on(kurtosis::kurtosis_builtin(
        x,
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(tensor_values(kurt).0[0], 1.5);
}

#[test]
fn flagged_moments_omit_nans_by_default() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, f64::NAN], vec![1, 3]).unwrap());
    let mad_out = block_on(mad::mad_builtin(
        x.clone(),
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(tensor_values(mad_out).0[0], 0.5);

    let skew_out = block_on(skewness::skewness_builtin(
        x.clone(),
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(tensor_values(skew_out).0[0], 0.0);

    let kurt_out = block_on(kurtosis::kurtosis_builtin(
        x,
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(tensor_values(kurt_out).0[0], 1.0);
}
