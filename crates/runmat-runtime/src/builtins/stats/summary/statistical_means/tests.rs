use super::{data, geomean, harmmean, options, rms, rmse};
use futures::executor::block_on;
use runmat_value::{IntValue, NumericDType, Tensor, Value};

fn scalar(value: Value) -> f64 {
    match value {
        Value::Num(value) => value,
        Value::Tensor(value) => match value.numeric_value_at(0).expect("scalar result storage") {
            runmat_value::NumericScalar::F64(number) => number,
            runmat_value::NumericScalar::F32(number) => f64::from(number),
            other => other.into_int_value().expect("integer result").to_f64(),
        },
        other => panic!("unexpected scalar result: {other:?}"),
    }
}

fn tensor(values: Vec<f64>, shape: Vec<usize>, dtype: NumericDType) -> Value {
    Value::Tensor(Tensor::new_with_dtype(values, shape, dtype).unwrap())
}

#[test]
fn single_precision_is_preserved_by_all_four_reductions() {
    let input = tensor(vec![1.0, 4.0, 9.0, 16.0], vec![2, 2], NumericDType::F32);
    for output in [
        block_on(geomean::geomean_builtin_source(input.clone(), vec![])).unwrap(),
        block_on(harmmean::harmmean_builtin_source(input.clone(), vec![])).unwrap(),
        block_on(rms::rms_builtin_source(input.clone(), vec![])).unwrap(),
        block_on(rmse::rmse_builtin_source(
            input.clone(),
            input.clone(),
            vec![],
        ))
        .unwrap(),
    ] {
        let Value::Tensor(output) = output else {
            panic!("expected typed tensor result")
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
    }
}

#[test]
fn control_and_weight_extensions_are_classified_from_typed_values() {
    assert!(data::integer_control(&Value::Int(IntValue::U64(u64::MAX))));
    assert!(data::exact_integer_weight(&Value::Int(IntValue::I8(1))));
    assert!(!data::integer_data(&Value::Num(1.0)));
}

#[test]
fn malformed_weighted_selector_is_rejected() {
    let arguments = crate::builtins::common::test_support::builtin_arguments(vec![
        Value::from("all"),
        Value::from("Weights"),
        Value::Num(1.0),
    ]);
    let result = block_on(options::parse(
        options::ReductionKind::RootMeanSquaredError,
        &arguments,
    ));
    assert!(result.is_err());
}

#[test]
fn geometric_negative_is_nan_only_in_its_affected_slice() {
    let input = tensor(vec![-1.0, 4.0, 9.0, 16.0], vec![2, 2], NumericDType::F64);
    let result = block_on(geomean::geomean_builtin_source(input, vec![])).unwrap();
    let Value::Tensor(result) = result else {
        panic!("expected per-column means")
    };
    let values = result.materialize_f64();
    assert!(values[0].is_nan());
    assert_eq!(values[1], 12.0);
}

#[test]
fn geometric_omitnan_skips_missing_but_not_negative_observations() {
    let input = tensor(
        vec![f64::NAN, -1.0, 9.0, 16.0, 4.0, 25.0],
        vec![2, 3],
        NumericDType::F64,
    );
    let result = block_on(geomean::geomean_builtin_source(
        input,
        vec![Value::from("omitnan")],
    ))
    .unwrap();
    let Value::Tensor(result) = result else {
        panic!("expected per-column means")
    };
    let values = result.materialize_f64();
    assert!(values[0].is_nan());
    assert_eq!(values[1], 12.0);
    assert!((values[2] - 10.0).abs() <= 4.0 * f64::EPSILON * 10.0);
}

#[test]
fn geometric_and_harmonic_singleton_or_trailing_axis_are_identity() {
    let input = tensor(vec![-3.0, 4.0], vec![2, 1], NumericDType::F64);
    for geometric in [true, false] {
        for dimension in [2.0, 100.0] {
            let result = if geometric {
                block_on(geomean::geomean_builtin_source(
                    input.clone(),
                    vec![Value::Num(dimension)],
                ))
            } else {
                block_on(harmmean::harmmean_builtin_source(
                    input.clone(),
                    vec![Value::Num(dimension)],
                ))
            }
            .unwrap();
            let Value::Tensor(result) = result else {
                panic!("expected unchanged vector")
            };
            assert_eq!(result.shape(), [2, 1]);
            let values = result.materialize_f64();
            if geometric {
                assert!(values[0].is_nan());
            } else {
                assert_eq!(values[0], -3.0);
            }
            assert_eq!(values[1], 4.0);
        }
    }
}

#[test]
fn integer_extension_noop_still_crosses_the_documented_double_boundary() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = Value::Int(IntValue::U64(4));
    for result in [
        block_on(geomean::geomean_builtin_source(
            input.clone(),
            vec![Value::Num(100.0)],
        )),
        block_on(harmmean::harmmean_builtin_source(
            input,
            vec![Value::Num(100.0)],
        )),
    ] {
        let Value::Num(value) = result.unwrap() else {
            panic!("integer extension must return double")
        };
        assert_eq!(value, 4.0);
    }
}

#[test]
fn rms_accepts_character_code_units_in_column_major_order() {
    let input = Value::CharArray(runmat_value::CharArray::new_row("AB"));
    let result = block_on(rms::rms_builtin_source(input, vec![Value::from("all")])).unwrap();
    assert!((scalar(result) - ((65.0f64.powi(2) + 66.0f64.powi(2)) / 2.0).sqrt()).abs() < 1e-12);
}

#[test]
fn rms_scaling_avoids_squaring_overflow() {
    let large = f64::MAX / 2.0;
    let input = tensor(vec![large, large], vec![2, 1], NumericDType::F64);
    let result = block_on(rms::rms_builtin_source(input, vec![])).unwrap();
    assert_eq!(scalar(result), large);
}

#[test]
fn complex_rms_scales_components_before_forming_magnitudes() {
    let large = f64::MAX;
    let input = Value::ComplexTensor(
        runmat_value::ComplexTensor::new(vec![(large, large), (0.0, 0.0)], vec![2, 1]).unwrap(),
    );
    let result = block_on(rms::rms_builtin_source(input, vec![])).unwrap();
    assert_eq!(scalar(result), large);
}

#[test]
fn rmse_scales_residual_components_before_subtraction_overflows() {
    let large = f64::MAX;
    let forecast = tensor(vec![large, 0.0, 0.0, 0.0], vec![4, 1], NumericDType::F64);
    let actual = tensor(vec![-large, 0.0, 0.0, 0.0], vec![4, 1], NumericDType::F64);
    let result = block_on(rmse::rmse_builtin_source(forecast, actual, vec![])).unwrap();
    assert_eq!(scalar(result), large);
}

#[test]
fn mixed_wide_integer_and_exact_floating_observation_keep_unit_residual() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let wide = Value::Int(IntValue::U64(9_007_199_254_740_993));
    let nearby = Value::Num(9_007_199_254_740_992.0);
    let result = block_on(rmse::rmse_builtin_source(
        wide.clone(),
        nearby.clone(),
        vec![],
    ))
    .unwrap();
    assert_eq!(scalar(result), 1.0);
    let reversed = block_on(rmse::rmse_builtin_source(nearby, wide, vec![])).unwrap();
    assert_eq!(scalar(reversed), 1.0);
}

#[test]
fn weighted_rmse_broadcasts_weights_from_unexpanded_forecast() {
    let forecast = tensor(vec![1.0, 2.0], vec![2, 1], NumericDType::F64);
    let actual = tensor(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2], NumericDType::F64);
    let weights = tensor(vec![1.0, 3.0], vec![2, 1], NumericDType::F64);
    let result = block_on(rmse::rmse_builtin_source(
        forecast,
        actual,
        vec![Value::Num(1.0), Value::from("Weights"), weights],
    ))
    .unwrap();
    let Value::Tensor(output) = result else {
        panic!("expected per-column values")
    };
    assert_eq!(output.shape(), [1, 2]);
    for value in output.materialize_f64() {
        assert!((value - (13.0_f64 / 4.0).sqrt()).abs() < 1e-12);
    }
}

#[test]
fn negative_weights_fail_before_data_broadcast() {
    let forecast = tensor(vec![1.0, 2.0], vec![2, 1], NumericDType::F64);
    let actual = tensor(vec![1.0, 2.0, 3.0], vec![3, 1], NumericDType::F64);
    let error = block_on(rmse::rmse_builtin_source(
        forecast,
        actual,
        vec![Value::from("Weights"), Value::Num(-1.0)],
    ))
    .unwrap_err();
    assert!(error.message.contains("weights must be nonnegative"));
}

#[test]
fn nd_default_vector_and_all_preserve_full_rank() {
    let input = tensor(
        (1..=12).map(f64::from).collect(),
        vec![2, 2, 3],
        NumericDType::F64,
    );
    let default = block_on(rms::rms_builtin_source(input.clone(), vec![])).unwrap();
    let Value::Tensor(default) = default else {
        panic!("expected N-D result")
    };
    assert_eq!(default.shape(), [1, 2, 3]);
    let vecdim = Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap());
    let across_axes = block_on(rms::rms_builtin_source(input.clone(), vec![vecdim])).unwrap();
    let Value::Tensor(across_axes) = across_axes else {
        panic!("expected N-D result")
    };
    assert_eq!(across_axes.shape(), [1, 2, 1]);
    let all = block_on(rms::rms_builtin_source(input, vec![Value::from("all")])).unwrap();
    let expected = ((1..=12).map(|value| f64::from(value * value)).sum::<f64>() / 12.0).sqrt();
    assert!((scalar(all) - expected).abs() < 1e-12);
}

#[test]
fn empty_nd_shapes_and_far_trailing_axis_do_not_allocate_by_axis_value() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let empty = tensor(vec![], vec![0, 2, 3], NumericDType::F64);
    let result = block_on(rms::rms_builtin_source(empty.clone(), vec![])).unwrap();
    let Value::Tensor(result) = result else {
        panic!("expected empty N-D result")
    };
    assert_eq!(result.shape(), [1, 2, 3]);
    assert_eq!(result.len(), 6);
    assert!(result.materialize_f64().iter().all(|value| value.is_nan()));
    assert!(
        scalar(block_on(rms::rms_builtin_source(empty, vec![Value::from("all")])).unwrap())
            .is_nan()
    );
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::F64);
    let far_axis = Value::Int(IntValue::U64(u64::MAX));
    let result = block_on(rms::rms_builtin_source(input, vec![far_axis])).unwrap();
    let Value::Tensor(result) = result else {
        panic!("expected unchanged shape")
    };
    assert_eq!(result.shape(), [2, 2]);
}
