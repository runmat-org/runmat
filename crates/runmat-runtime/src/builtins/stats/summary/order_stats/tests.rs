use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

#[test]
fn invalid_option_precedes_any_provider_download() {
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(
            vec![3, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("unregistered resident fixture");
    let automatic = Value::GpuTensor(allocation.handle.clone());
    let invalid = block_on(quantile::quantile_builtin(
        automatic.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.5),
            Value::from("unsupported"),
        ]),
    ))
    .expect_err("option admission must precede gathering");
    assert!(
        invalid.message().contains("unsupported option"),
        "{invalid}"
    );
}

#[test]
fn oversized_structural_controls_fail_before_resident_gather() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(
            vec![3, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("unregistered resident fixture");
    let resident = || Value::GpuTensor(allocation.handle.clone());
    let dimension = Value::Int(IntValue::U64(u64::MAX));
    let error = block_on(quantile::quantile_builtin(
        resident(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.5), dimension]),
    ))
    .expect_err("oversized dimension must fail without a provider download");
    assert!(
        error.message().contains("structural range") || error.message().contains("host resources"),
        "{error}"
    );

    let count = Value::Int(IntValue::U64(u64::MAX));
    let error = block_on(quantile::quantile_builtin(
        resident(),
        crate::builtins::common::test_support::builtin_arguments(vec![count]),
    ))
    .expect_err("oversized scalar count must fail without a provider download");
    assert!(
        error.message().contains("quantile count is too large"),
        "{error}"
    );

    let rounded_boundary = Value::Num(usize::MAX as f64);
    let error = block_on(quantile::quantile_builtin(
        resident(),
        crate::builtins::common::test_support::builtin_arguments(vec![rounded_boundary]),
    ))
    .expect_err("binary64 cannot identify the largest usize exactly");
    assert!(
        error.message().contains("count must be an integer"),
        "{error}"
    );
}

#[test]
fn quantile_scalar_count_and_vecdim_have_distinct_shapes() {
    let sample =
        || Value::Tensor(Tensor::new((1..=8).map(f64::from).collect(), vec![2, 2, 2]).unwrap());
    let count = block_on(quantile::quantile_builtin(
        sample(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(4.0)]),
    ))
    .unwrap();
    let Value::Tensor(count) = count else {
        panic!("expected quantile array")
    };
    assert_eq!(count.shape(), [4, 2, 2]);
    assert_eq!(&count.materialize_f64()[..4], &[1.0, 1.3, 1.7, 2.0]);

    let dims = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let reduced = block_on(quantile::quantile_builtin(
        sample(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.5), dims]),
    ))
    .unwrap();
    let Value::Tensor(reduced) = reduced else {
        panic!("expected reduced array")
    };
    assert_eq!(reduced.shape(), [1, 1, 2]);
    assert_eq!(reduced.materialize_f64(), vec![2.5, 6.5]);
}

#[test]
fn tiedrank_kendall_and_ansari_modes_match_documented_vectors() {
    let sample = || Value::Tensor(Tensor::new(vec![-2.0, 1.0, 3.0, 1.0, 4.0], vec![5, 1]).unwrap());
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let kendall = block_on(tiedrank::tiedrank_builtin(
            sample(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .unwrap();
        let Value::OutputList(kendall) = kendall else {
            panic!("expected two outputs")
        };
        let Value::Tensor(ranks) = &kendall[0] else {
            panic!("expected ranked array")
        };
        let Value::Tensor(adjustment) = &kendall[1] else {
            panic!("expected correction array")
        };
        assert_eq!(ranks.materialize_f64(), vec![1.0, 2.5, 4.0, 2.5, 5.0]);
        assert_eq!(adjustment.shape(), [3, 1]);
        assert_eq!(adjustment.materialize_f64(), vec![3.0, 0.0, 18.0]);
    }
    let ansari = block_on(tiedrank::tiedrank_builtin(
        sample(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::Num(1.0),
        ]),
    ))
    .unwrap();
    let Value::Tensor(ansari) = ansari else {
        panic!("expected ranked array")
    };
    assert_eq!(ansari.materialize_f64(), vec![1.0, 2.5, 2.0, 2.5, 1.0]);
    let invalid = block_on(tiedrank::tiedrank_builtin(
        sample(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Num(1.0),
        ]),
    ))
    .unwrap_err();
    assert!(invalid.message().contains("unsupported mode"));
}

#[test]
fn typed_integer_request_range_is_checked_before_binary64_conversion() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = || Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    let outside = Value::Int(IntValue::U64((1_u64 << 53) + 1));
    let error = block_on(prctile::prctile_builtin(
        x(),
        crate::builtins::common::test_support::builtin_arguments(vec![outside]),
    ))
    .expect_err("out-of-range uint64 cannot round into an accepted percentage");
    assert!(error.message().contains("percentages must be"), "{error}");
    let error = block_on(quantile::quantile_builtin(
        x(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I64(
            -1,
        ))]),
    ))
    .expect_err("negative integer probability is invalid");
    assert!(error.message().contains("probabilities must be"), "{error}");
}

#[test]
fn quantile_vector_uses_midpoint_interpolation() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap());
    let out = block_on(quantile::quantile_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
            Tensor::new(vec![0.25, 0.5, 0.75], vec![1, 3]).unwrap(),
        )]),
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), [3, 1]);
            assert_eq!(tensor.materialize_f64(), vec![1.5, 3.0, 6.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn quantile_supports_compatibility_target_midpoint_inclusive_and_exclusive_methods() {
    let input = || Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 6.0, 10.0], vec![5, 1]).unwrap());
    let midpoint = block_on(quantile::quantile_builtin(
        input(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.4)]),
    ))
    .unwrap();
    let inclusive = block_on(quantile::quantile_builtin(
        input(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.4),
            Value::from("Method"),
            Value::from("inclusive"),
        ]),
    ))
    .unwrap();
    let exclusive = block_on(quantile::quantile_builtin(
        input(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.4),
            Value::from("Method"),
            Value::from("exclusive"),
        ]),
    ))
    .unwrap();
    assert!(matches!(midpoint, Value::Num(value) if value == 2.5));
    assert!(matches!(inclusive, Value::Num(value) if (value - 2.6).abs() < 1.0e-12));
    assert!(matches!(exclusive, Value::Num(value) if (value - 2.4).abs() < 1.0e-12));
}

#[test]
fn quantile_and_prctile_integer_extensions_follow_compatibility_mode() {
    let integer_data =
        || Value::Tensor(Tensor::new_integer(IntegerStorage::I16(vec![1, 3]), vec![2, 1]).unwrap());
    let floating = || Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap());
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(quantile::quantile_builtin(
            integer_data(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.5)]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:QuantileIntegerDataExtension")
        );
        let error = block_on(prctile::prctile_builtin(
            floating(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U8(50),
            )]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:PrctileTypedIntegerPercentageExtension")
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let out = block_on(quantile::quantile_builtin(
            integer_data(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.5)]),
        ))
        .unwrap();
        assert!(matches!(out, Value::Num(value) if value == 2.0));
    }
}

#[test]
fn tiedrank_integer_extension_follows_compatibility_mode() {
    let input = || {
        Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 1]), vec![3, 1]).unwrap())
    };
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(tiedrank::tiedrank_builtin(
            input(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:TiedrankIntegerDataExtension")
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let out = block_on(tiedrank::tiedrank_builtin(
            input(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        assert!(
            matches!(out, Value::Tensor(tensor) if tensor.materialize_f64() == vec![1.5, 3.0, 1.5])
        );
    }
}

#[test]
fn prctile_reduces_columns_by_default() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 3.0, 10.0, 20.0], vec![2, 2]).unwrap());
    let out = block_on(prctile::prctile_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(50.0)]),
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), [1, 2]);
            assert_eq!(tensor.materialize_f64(), vec![2.0, 15.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn tiedrank_averages_ties_and_preserves_nan() {
    let x = Value::Tensor(Tensor::new(vec![10.0, 20.0, 20.0, f64::NAN], vec![4, 1]).unwrap());
    let out = block_on(tiedrank::tiedrank_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), [4, 1]);
            assert_eq!(tensor.materialize_f64()[0], 1.0);
            assert_eq!(tensor.materialize_f64()[1], 2.5);
            assert_eq!(tensor.materialize_f64()[2], 2.5);
            assert!(tensor.materialize_f64()[3].is_nan());
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn quantile_omits_nan_by_default_and_rejects_approximate_method() {
    let x = Value::Tensor(Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap());
    let out = block_on(quantile::quantile_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.5)]),
    ))
    .unwrap();
    match out {
        Value::Num(value) => assert_eq!(value, 2.0),
        other => panic!("expected scalar, got {other:?}"),
    }

    let x = Value::Tensor(Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap());
    let out = block_on(quantile::quantile_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.5),
            Value::from("includenan"),
        ]),
    ))
    .unwrap();
    assert!(matches!(out, Value::Num(value) if value.is_nan()));

    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    let err = block_on(quantile::quantile_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.5),
            Value::from("Method"),
            Value::from("approximate"),
        ]),
    ))
    .unwrap_err();
    assert!(err.message().contains("approximate quantile method"));
}

#[test]
fn tiedrank_ranks_matrix_columns_and_returns_tieadj() {
    let x = Value::Tensor(Tensor::new(vec![3.0, 1.0, 1.0, 2.0, 2.0, 5.0], vec![3, 2]).unwrap());
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(tiedrank::tiedrank_builtin(
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    match out {
        Value::OutputList(values) => {
            assert_eq!(values.len(), 2);
            match &values[0] {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), [3, 2]);
                    assert_eq!(tensor.materialize_f64(), vec![3.0, 1.5, 1.5, 1.5, 1.5, 3.0]);
                }
                other => panic!("expected rank tensor, got {other:?}"),
            }
            match &values[1] {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), [1, 2]);
                    assert_eq!(tensor.materialize_f64(), vec![3.0, 3.0]);
                }
                other => panic!("expected tieadj tensor, got {other:?}"),
            }
        }
        other => panic!("expected output list, got {other:?}"),
    }
}

#[test]
fn quantile_typed_integer_input_and_probability_read_exact_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = Tensor::new_integer(IntegerStorage::I16(vec![1, 3, 9, 27]), vec![4, 1]).unwrap();
    let p = Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap();

    let out = block_on(quantile::quantile_builtin(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(p),
            Value::from("all"),
        ]),
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), [2, 1]);
            assert_eq!(tensor.materialize_f64(), vec![1.0, 27.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn prctile_typed_integer_input_and_percentiles_read_exact_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = Tensor::new_integer(
        IntegerStorage::I16(vec![10, 30, 50, 70, 20, 40, 60, 80]),
        vec![4, 2],
    )
    .unwrap();
    let p = Tensor::new_integer(IntegerStorage::U8(vec![25, 50, 75]), vec![1, 3]).unwrap();

    let out = block_on(prctile::prctile_builtin(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(p)]),
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), [3, 2]);
            let expected = [20.0, 40.0, 60.0, 30.0, 50.0, 70.0];
            for (actual, expect) in tensor.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1.0e-12, "{actual} vs {expect}");
            }
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn tiedrank_typed_integer_matrix_reads_exact_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = Tensor::new_integer(IntegerStorage::I16(vec![3, 1, 1, 2, 2, 5]), vec![3, 2]).unwrap();

    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(tiedrank::tiedrank_builtin(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    match out {
        Value::OutputList(values) => {
            assert_eq!(values.len(), 2);
            match &values[0] {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), [3, 2]);
                    assert_eq!(tensor.materialize_f64(), vec![3.0, 1.5, 1.5, 1.5, 1.5, 3.0]);
                }
                other => panic!("expected rank tensor, got {other:?}"),
            }
            match &values[1] {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), [1, 2]);
                    assert_eq!(tensor.materialize_f64(), vec![3.0, 3.0]);
                }
                other => panic!("expected tieadj tensor, got {other:?}"),
            }
        }
        other => panic!("expected output list, got {other:?}"),
    }
}

#[test]
fn tiedrank_uint64_distinguishes_values_beyond_f64_precision() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let base = 1_u64 << 53;
    let x =
        Tensor::new_integer(IntegerStorage::U64(vec![base, base + 1, base]), vec![3, 1]).unwrap();

    let out = block_on(tiedrank::tiedrank_builtin(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    match out {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.5, 3.0, 1.5]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn ordered_integer_comparison_covers_all_storage_classes() {
    macro_rules! assert_ordered {
        ($variant:ident, $lower:expr, $upper:expr) => {
            assert_eq!(
                compare_ordered_values(
                    &OrderedValue::Integer(IntValue::$variant($lower)),
                    &OrderedValue::Integer(IntValue::$variant($upper)),
                ),
                Ordering::Less,
            );
        };
    }

    assert_ordered!(I8, -1, 1);
    assert_ordered!(I16, -1, 1);
    assert_ordered!(I32, -1, 1);
    assert_ordered!(I64, -1, 1);
    assert_ordered!(U8, 1, 2);
    assert_ordered!(U16, 1, 2);
    assert_ordered!(U32, 1, 2);
    assert_ordered!(U64, 1, 2);
}

#[test]
fn tiedrank_registered_extension_covers_all_integer_classes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = vec![
        IntegerStorage::I8(vec![1, 2, 1]),
        IntegerStorage::I16(vec![1, 2, 1]),
        IntegerStorage::I32(vec![1, 2, 1]),
        IntegerStorage::I64(vec![1, 2, 1]),
        IntegerStorage::U8(vec![1, 2, 1]),
        IntegerStorage::U16(vec![1, 2, 1]),
        IntegerStorage::U32(vec![1, 2, 1]),
        IntegerStorage::U64(vec![1, 2, 1]),
    ];
    for storage in storages {
        let input = Tensor::new_integer(storage, vec![3, 1]).unwrap();
        let out = block_on(tiedrank::tiedrank_builtin(
            Value::Tensor(input),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        assert!(
            matches!(out, Value::Tensor(tensor) if tensor.materialize_f64() == vec![1.5, 3.0, 1.5])
        );
    }
}
