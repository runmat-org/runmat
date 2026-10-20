use super::*;

#[test]
fn geomean_reduces_columns_and_omits_nans() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 4.0, f64::NAN, 9.0], vec![2, 2]).unwrap());
    let out = block_on(geomean::geomean_builtin_source(
        x,
        vec![Value::CharArray(runmat_value::CharArray::new_row(
            "omitnan",
        ))],
    ))
    .unwrap();
    let (data, shape) = tensor_values(out);
    assert_eq!(shape, vec![1, 2]);
    assert_close(data[0], 2.0);
    assert_close(data[1], 9.0);
}

#[test]
fn harmmean_and_rms_support_all_selector() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 4.0], vec![1, 3]).unwrap());
    let harmonic = block_on(harmmean::harmmean_builtin_source(
        x.clone(),
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(tensor_values(harmonic).0[0], 12.0 / 7.0);

    let rms = block_on(rms::rms_builtin_source(
        x,
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(tensor_values(rms).0[0], (21.0_f64 / 3.0).sqrt());
}

#[test]
fn descriptive_reductions_read_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = int_tensor(IntegerStorage::U16(vec![1, 4, 9]), vec![1, 3]);
    let all = vec![Value::CharArray(runmat_value::CharArray::new_row("all"))];

    let geometric = block_on(geomean::geomean_builtin_source(x.clone(), all.clone())).unwrap();
    assert_close(tensor_values(geometric).0[0], (36.0_f64).powf(1.0 / 3.0));

    let harmonic = block_on(harmmean::harmmean_builtin_source(x.clone(), all.clone())).unwrap();
    assert_close(tensor_values(harmonic).0[0], 3.0 / (1.0 + 0.25 + 1.0 / 9.0));

    let rms = block_on(rms::rms_builtin_source(x.clone(), all.clone())).unwrap();
    assert_close(
        tensor_values(rms).0[0],
        ((1.0 + 16.0 + 81.0) / 3.0_f64).sqrt(),
    );

    let mad = block_on(mad::mad_builtin(x.clone(), all.clone())).unwrap();
    assert_close(tensor_values(mad).0[0], 26.0 / 9.0);

    let skew = block_on(skewness::skewness_builtin(x.clone(), all.clone())).unwrap();
    assert_close(tensor_values(skew).0[0], 0.294_799_620_144_828_63);

    let kurt = block_on(kurtosis::kurtosis_builtin(x, all)).unwrap();
    assert_close(tensor_values(kurt).0[0], 1.5);
}

#[test]
fn descriptive_integer_extensions_cover_all_storage_classes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = vec![
        IntegerStorage::I8(vec![1, 2, 4]),
        IntegerStorage::I16(vec![1, 2, 4]),
        IntegerStorage::I32(vec![1, 2, 4]),
        IntegerStorage::I64(vec![1, 2, 4]),
        IntegerStorage::U8(vec![1, 2, 4]),
        IntegerStorage::U16(vec![1, 2, 4]),
        IntegerStorage::U32(vec![1, 2, 4]),
        IntegerStorage::U64(vec![1, 2, 4]),
    ];
    for storage in storages {
        let input = || int_tensor(storage.clone(), vec![3, 1]);
        let all = || vec![Value::from("all")];
        assert!(block_on(geomean::geomean_builtin_source(input(), all())).is_ok());
        assert!(block_on(harmmean::harmmean_builtin_source(input(), all())).is_ok());
        assert!(block_on(rms::rms_builtin_source(input(), all())).is_ok());
        assert!(block_on(skewness::skewness_builtin(input(), all())).is_ok());
        assert!(block_on(kurtosis::kurtosis_builtin(input(), all())).is_ok());
        assert!(block_on(rmse::rmse_builtin_source(input(), Value::Num(0.0), all())).is_ok());
        assert!(matches!(
            block_on(tabulate::tabulate_builtin(input())).unwrap(),
            Value::Cell(_)
        ));
    }
}

#[test]
fn descriptive_reduction_shape_uses_typed_integer_storage_not_mirror() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = mirrorless_int_tensor(IntegerStorage::U16(vec![9]), Vec::new());

    let out = block_on(geomean::geomean_builtin_source(x, Vec::new())).unwrap();

    let (data, shape) = tensor_values(out);
    assert_eq!(shape, vec![1, 1]);
    assert_close(data[0], 9.0);
}

#[test]
fn rmse_broadcasts_and_reduces_residuals() {
    let x = Value::Tensor(Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]).unwrap());
    let y = Value::Num(1.0);
    let out = block_on(rmse::rmse_builtin_source(
        x,
        y,
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(
        tensor_values(out).0[0],
        ((1.0 + 9.0 + 25.0) / 3.0_f64).sqrt(),
    );
}

#[test]
fn rmse_reads_typed_integer_residual_inputs_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = int_tensor(IntegerStorage::I16(vec![2, 4, 6]), vec![3, 1]);
    let y = int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]);
    let out = block_on(rmse::rmse_builtin_source(
        x,
        y,
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(
        tensor_values(out).0[0],
        ((1.0 + 9.0 + 25.0) / 3.0_f64).sqrt(),
    );
}

#[test]
fn rmse_supports_weights_and_omitnan() {
    let x = Value::Tensor(Tensor::new(vec![2.0, f64::NAN, 6.0], vec![3, 1]).unwrap());
    let y = Value::Num(1.0);
    let weights = Value::Tensor(Tensor::new(vec![1.0, 100.0, 3.0], vec![3, 1]).unwrap());
    let out = block_on(rmse::rmse_builtin_source(
        x,
        y,
        vec![
            Value::Num(1.0),
            Value::CharArray(runmat_value::CharArray::new_row("omitnan")),
            Value::CharArray(runmat_value::CharArray::new_row("Weights")),
            weights,
        ],
    ))
    .unwrap();
    let (data, shape) = tensor_values(out);
    assert_eq!(shape, vec![1, 1]);
    assert_close(data[0], ((1.0 + 75.0) / 4.0_f64).sqrt());
}

#[test]
fn weighted_rmse_reads_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = int_tensor(IntegerStorage::I16(vec![2, 4, 6]), vec![3, 1]);
    let y = int_tensor(IntegerStorage::I16(vec![1]), vec![1, 1]);
    let weights = int_tensor(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]);
    let out = block_on(rmse::rmse_builtin_source(
        x,
        y,
        vec![
            Value::Num(1.0),
            Value::CharArray(runmat_value::CharArray::new_row("Weights")),
            weights,
        ],
    ))
    .unwrap();
    assert_close(
        tensor_values(out).0[0],
        ((1.0 + 18.0 + 75.0) / 6.0_f64).sqrt(),
    );
}

#[test]
fn weighted_rmse_weight_length_uses_typed_integer_storage_not_mirror() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = Value::Tensor(Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]).unwrap());
    let y = Value::Num(1.0);
    let weights = mirrorless_int_tensor(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]);

    let out = block_on(rmse::rmse_builtin_source(
        x,
        y,
        vec![
            Value::Num(1.0),
            Value::CharArray(runmat_value::CharArray::new_row("Weights")),
            weights,
        ],
    ))
    .unwrap();

    assert_close(
        tensor_values(out).0[0],
        ((1.0 + 18.0 + 75.0) / 6.0_f64).sqrt(),
    );
}

#[test]
fn rms_and_rmse_support_complex_magnitudes() {
    let x = Value::ComplexTensor(
        runmat_value::ComplexTensor::new(vec![(3.0, 4.0), (0.0, 12.0)], vec![2, 1]).unwrap(),
    );
    let out = block_on(rms::rms_builtin_source(
        x,
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(tensor_values(out).0[0], ((25.0 + 144.0) / 2.0_f64).sqrt());

    let out = block_on(rmse::rmse_builtin_source(
        Value::Complex(3.0, 4.0),
        Value::Num(0.0),
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(tensor_values(out).0[0], 5.0);
}

#[test]
fn rms_reads_typed_complex_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = complex_int_tensor(
        IntegerStorage::I16(vec![3, 0]),
        IntegerStorage::I16(vec![4, 12]),
        vec![2, 1],
    );
    let out = block_on(rms::rms_builtin_source(
        x,
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(tensor_values(out).0[0], ((25.0 + 144.0) / 2.0_f64).sqrt());
}

#[test]
fn rmse_reads_typed_complex_integer_residual_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = complex_int_tensor(
        IntegerStorage::I16(vec![3, 0]),
        IntegerStorage::I16(vec![4, 12]),
        vec![2, 1],
    );
    let y = complex_int_tensor(
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I16(vec![2]),
        vec![1, 1],
    );
    let out = block_on(rmse::rmse_builtin_source(
        x,
        y,
        vec![Value::CharArray(runmat_value::CharArray::new_row("all"))],
    ))
    .unwrap();
    assert_close(tensor_values(out).0[0], ((8.0 + 101.0) / 2.0_f64).sqrt());
}

#[test]
fn rmse_uses_implicit_expansion_and_rejects_weighted_all() {
    let lhs = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let rhs = Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap());
    let out = block_on(rmse::rmse_builtin_source(lhs, rhs, vec![Value::Num(1.0)])).unwrap();
    let (data, shape) = tensor_values(out);
    assert_eq!(shape, vec![1, 2]);
    assert_close(data[0], (0.5_f64).sqrt());
    assert_close(data[1], (0.5_f64).sqrt());

    let err = block_on(rmse::rmse_builtin_source(
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        Value::Num(0.0),
        vec![
            Value::CharArray(runmat_value::CharArray::new_row("all")),
            Value::CharArray(runmat_value::CharArray::new_row("Weights")),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap()),
        ],
    ))
    .unwrap_err();
    assert!(err.message.contains("weights are not supported"));
}
