use super::statistics::kendall_tau_b;
use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

#[test]
fn empty_observations_preserve_variable_count_and_row_mismatch() {
    let output = block_on(corr_builtin(
        tensor(vec![], vec![0, 3]),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let Value::Tensor(output) = output else {
        panic!("matrix")
    };
    assert_eq!(output.shape(), [3, 3]);
    assert!(output.materialize_f64().iter().all(|value| value.is_nan()));
    let error = block_on(corr_builtin(
        tensor(vec![], vec![2, 0]),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(vec![], vec![3, 0])]),
    ))
    .unwrap_err();
    assert!(error.message.contains("same number of observations"));
}

#[test]
fn empty_observation_matrix_checks_result_dimension_overflow() {
    let input = tensor(vec![], vec![0, usize::MAX]);
    let error = block_on(corr_builtin(
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert!(error.message.contains("platform limits"));
}

#[test]
fn rejects_extra_observation_pages_but_accepts_trailing_singletons() {
    let error = block_on(corr_builtin(
        tensor(vec![1.; 8], vec![2, 2, 2]),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert!(error.to_string().contains("two-dimensional"));
    assert!(block_on(corr_builtin(
        tensor(vec![1., 2., 3.], vec![3, 1, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![])
    ))
    .is_ok());
}

#[test]
fn resident_observations_gather_without_releasing_source() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1., 2., 3., 3., 2., 1.],
                shape: &[3, 2],
            })
            .unwrap();
        let result = block_on(corr_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let Value::Tensor(result) = result else {
            panic!("host correlation matrix")
        };
        assert_eq!(result.shape(), [2, 2]);
        assert!((result.materialize_f64()[1] + 1.).abs() < 1e-12);
        assert!(block_on(provider.download(&handle)).is_ok());
    });
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, _poison: f64) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

#[test]
fn corr_matrix_defaults_to_columns() {
    let out = block_on(corr_builtin(
        tensor(vec![1.0, 2.0, 1.0, 4.0], vec![2, 2]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), [2, 2]);
            assert!((tensor.materialize_f64()[0] - 1.0).abs() < 1e-12);
            assert!((tensor.materialize_f64()[1] - 1.0).abs() < 1e-12);
            assert!((tensor.materialize_f64()[2] - 1.0).abs() < 1e-12);
            assert!((tensor.materialize_f64()[3] - 1.0).abs() < 1e-12);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn corr_accepts_typed_integer_matrix_inputs() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(corr_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 1, 4]), vec![2, 2], 0.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), [2, 2]);
            assert!((tensor.materialize_f64()[0] - 1.0).abs() < 1.0e-12);
            assert!((tensor.materialize_f64()[3] - 1.0).abs() < 1.0e-12);
        }
        other => panic!("expected tensor correlation, got {other:?}"),
    }
}

#[test]
fn corr_complete_rows_reads_typed_integer_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = poisoned_int_tensor(
        IntegerStorage::I16(vec![1, 2, 3, 2, 4, 6]),
        vec![3, 2],
        f64::NAN,
    );
    let y = poisoned_int_tensor(IntegerStorage::U16(vec![5, 10, 15]), vec![3, 1], f64::NAN);

    let out = block_on(corr_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![
            y,
            Value::from("Rows"),
            Value::from("complete"),
        ]),
    ))
    .unwrap();

    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), [2, 1]);
            assert!((tensor.materialize_f64()[0] - 1.0).abs() < 1.0e-12);
            assert!((tensor.materialize_f64()[1] - 1.0).abs() < 1.0e-12);
        }
        other => panic!("expected tensor correlation, got {other:?}"),
    }
}

#[test]
fn corr_xy_returns_cross_correlation_matrix() {
    let x = tensor(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0], vec![3, 2]);
    let y = tensor(vec![2.0, 4.0, 6.0], vec![3, 1]);
    let out = block_on(corr_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![y]),
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), [2, 1]);
            assert!((tensor.materialize_f64()[0] - 1.0).abs() < 1e-12);
            assert!((tensor.materialize_f64()[1] + 1.0).abs() < 1e-12);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn corr_pairwise_omits_nan_pair() {
    let x = tensor(vec![1.0, 2.0, f64::NAN, 4.0], vec![4, 1]);
    let y = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
    let out = block_on(corr_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![
            y,
            Value::from("Rows"),
            Value::from("pairwise"),
        ]),
    ))
    .unwrap();
    match out {
        Value::Num(value) => assert!((value - 1.0).abs() < 1e-12),
        other => panic!("expected scalar, got {other:?}"),
    }
}

#[test]
fn corr_returns_pvalues_when_requested() {
    let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
    let y = tensor(vec![2.0, 4.0, 6.0, 8.0], vec![4, 1]);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(corr_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![y]),
    ))
    .unwrap();
    match out {
        Value::OutputList(values) => {
            assert_eq!(values.len(), 2);
            assert!(matches!(&values[0], Value::Num(value) if (*value - 1.0).abs() < 1e-12));
            assert!(matches!(&values[1], Value::Num(value) if *value <= 1e-12));
        }
        other => panic!("expected output list, got {other:?}"),
    }
}

#[test]
fn corr_supports_kendall_and_tail_option() {
    let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
    let y = tensor(vec![1.0, 3.0, 2.0, 4.0], vec![4, 1]);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(corr_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![
            y,
            Value::from("Type"),
            Value::from("Kendall"),
            Value::from("Tail"),
            Value::from("right"),
        ]),
    ))
    .unwrap();
    match out {
        Value::OutputList(values) => {
            assert_eq!(values.len(), 2);
            assert!(
                matches!(&values[0], Value::Num(value) if (*value - (2.0 / 3.0)).abs() < 1e-12)
            );
            assert!(matches!(&values[1], Value::Num(value) if *value > 0.0 && *value < 0.2));
        }
        other => panic!("expected output list, got {other:?}"),
    }
}

#[test]
fn corr_integer_extensions_are_independently_gated() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let data_error = block_on(corr_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1], 0.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        data_error.identifier(),
        Some("RunMat:compatibility:CorrIntegerDataExtension")
    );

    let weight_error = block_on(corr_builtin(
        tensor(vec![1.0, 2.0, 3.0], vec![3, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Weights"),
            poisoned_int_tensor(IntegerStorage::U8(vec![1, 2, 3]), vec![3, 1], 0.0),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        weight_error.identifier(),
        Some("RunMat:compatibility:CorrIntegerWeightsExtension")
    );
}

#[test]
fn corr_resident_integer_data_rejects_before_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input).unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(corr_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:CorrIntegerDataExtension")
        );
        drop(handle);
    });
}

#[test]
fn corr_supports_all_eight_integer_data_classes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = vec![
        IntegerStorage::I8(vec![1, 2, 3]),
        IntegerStorage::I16(vec![1, 2, 3]),
        IntegerStorage::I32(vec![1, 2, 3]),
        IntegerStorage::I64(vec![1, 2, 3]),
        IntegerStorage::U8(vec![1, 2, 3]),
        IntegerStorage::U16(vec![1, 2, 3]),
        IntegerStorage::U32(vec![1, 2, 3]),
        IntegerStorage::U64(vec![1, 2, 3]),
    ];
    for storage in storages {
        let class = storage.class_name();
        let weight_storage = storage.clone();
        let out = block_on(corr_builtin(
            poisoned_int_tensor(storage, vec![3, 1], 0.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_or_else(|error| panic!("{class}: {error}"));
        assert!(matches!(out, Value::Num(value) if (value - 1.0).abs() < 1.0e-12));

        let weighted = block_on(corr_builtin(
            tensor(vec![1.0, 2.0, 3.0], vec![3, 1]),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Weights"),
                poisoned_int_tensor(weight_storage, vec![3, 1], 0.0),
            ]),
        ))
        .unwrap_or_else(|error| panic!("{class} weights: {error}"));
        assert!(matches!(weighted, Value::Num(value) if (value - 1.0).abs() < 1.0e-12));
    }
}

#[test]
fn corr_wide_integer_rank_and_pearson_paths_remain_distinct() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let base = 1_u64 << 53;
    let x = poisoned_int_tensor(
        IntegerStorage::U64(vec![base, base + 1, base + 2]),
        vec![3, 1],
        0.0,
    );
    let y = poisoned_int_tensor(
        IntegerStorage::U64(vec![base + 10, base + 11, base + 12]),
        vec![3, 1],
        0.0,
    );
    for method in ["Pearson", "Spearman", "Kendall"] {
        let out = block_on(corr_builtin(
            x.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                y.clone(),
                Value::from("Type"),
                Value::from(method),
            ]),
        ))
        .unwrap();
        assert!(
            matches!(out, Value::Num(value) if (value - 1.0).abs() < 1.0e-12),
            "{method}: {out:?}"
        );
    }
}

#[test]
fn corr_weights_affect_all_types_and_force_nan_pvalues() {
    let x = tensor(vec![1.0, 2.0, 3.0], vec![3, 1]);
    let y = tensor(vec![1.0, 4.0, 2.0], vec![3, 1]);
    let weights = tensor(vec![1.0, 0.0, 1.0], vec![3, 1]);
    for method in ["Pearson", "Spearman", "Kendall"] {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let out = block_on(corr_builtin(
            x.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                y.clone(),
                Value::from("Type"),
                Value::from(method),
                Value::from("Weights"),
                weights.clone(),
            ]),
        ))
        .unwrap();
        let Value::OutputList(values) = out else {
            panic!("{method}: expected two outputs");
        };
        assert!(
            matches!(values[0], Value::Num(value) if (value - 1.0).abs() < 1.0e-12),
            "{method}: {:?}",
            values[0]
        );
        assert!(
            matches!(values[1], Value::Num(value) if value.is_nan()),
            "{method}: {:?}",
            values[1]
        );
    }
}

#[test]
fn corr_weighted_spearman_uses_documented_weighted_midrank_formula() {
    let column = CorrColumn::Floating(vec![1.0, 2.0, 2.0]);
    let ranks = column.ranks(&[0, 1, 2], Some(&[1.0, 2.0, 4.0]));
    assert_eq!(ranks, vec![1.0, 5.5, 5.5]);
}

#[test]
fn corr_kendall_excludes_joint_ties_from_tau_b_denominator() {
    let x = CorrColumn::Integer(vec![IntValue::U64(7), IntValue::U64(7)]);
    let y = CorrColumn::Integer(vec![IntValue::U64(9), IntValue::U64(9)]);
    assert!(kendall_tau_b(&x, &y, &[0, 1], None).is_nan());
}
