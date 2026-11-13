use super::*;
use futures::executor::block_on;
use runmat_value::{CharArray, IntegerStorage, Utf16CodeUnit};

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn logical(data: Vec<u8>, shape: Vec<usize>) -> Value {
    Value::LogicalArray(LogicalArray::new(data, shape).unwrap())
}

fn cleared_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

#[test]
fn scalar_counts_read_every_integer_storage_variant_not_the_float_mirror() {
    for storage in [
        IntegerStorage::I8(vec![2]),
        IntegerStorage::I16(vec![2]),
        IntegerStorage::I32(vec![2]),
        IntegerStorage::I64(vec![2]),
        IntegerStorage::U8(vec![2]),
        IntegerStorage::U16(vec![2]),
        IntegerStorage::U32(vec![2]),
        IntegerStorage::U64(vec![2]),
    ] {
        let value = poisoned_int_tensor(storage, vec![1, 1]);
        assert_eq!(PartitionInput::from_value(value.clone()).unwrap().n, 2);
        assert_eq!(holdout_count(&value, 6).unwrap(), 2);
    }
}

fn cv(first: Value, second: Value, rest: Vec<Value>) -> Value {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    block_on(cvpartition_builtin(
        None,
        first,
        second,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
    .expect("cvpartition")
}

fn logical_output(value: Value) -> LogicalArray {
    let Value::LogicalArray(array) = value else {
        panic!("expected logical array");
    };
    array
}

#[test]
fn holdout_fraction_requires_two_observations_without_panicking() {
    let error = holdout_count(&Value::Num(0.5), 1).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:cvpartition:InvalidArgument")
    );
    assert!(holdout_count(&Value::Num(0.5), 2).is_ok());
}

#[test]
fn kfold_partition_exposes_balanced_masks() {
    let _lock = random::test_guard();
    random::set_seed(2026).unwrap();
    let partition = cv(
        Value::Num(6.0),
        Value::String("KFold".into()),
        vec![Value::Num(3.0)],
    );
    let Value::Object(object) = &partition else {
        panic!("object");
    };
    assert_eq!(
        object.properties().get("NumTestSets"),
        Some(&Value::Num(3.0))
    );
    assert_eq!(
        object.properties().get("TestSize"),
        Some(&tensor(vec![2.0, 2.0, 2.0], vec![1, 3]))
    );

    let all_test = logical_output(
        block_on(test_builtin(
            None,
            partition.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        ))
        .expect("test all"),
    );
    assert_eq!(all_test.shape, vec![6, 3]);
    for fold in 0..3 {
        assert_eq!(
            all_test.data[fold * 6..(fold + 1) * 6]
                .iter()
                .filter(|flag| **flag != 0)
                .count(),
            2
        );
    }
    for row in 0..6 {
        assert_eq!(
            (0..3)
                .filter(|fold| all_test.data[row + fold * 6] != 0)
                .count(),
            1
        );
    }

    let train1 = logical_output(
        block_on(training_builtin(
            None,
            partition,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .expect("training 1"),
    );
    assert_eq!(train1.shape, vec![6, 1]);
    assert_eq!(train1.data.iter().filter(|flag| **flag != 0).count(), 4);
}

#[test]
fn numeric_group_order_keeps_seed_repeatability_and_each_class_covered() {
    assert_eq!(
        char_row_labels(
            runmat_value::CharArray::from_unicode_scalars(vec!['a', '1', 'b', '2'], 2, 2).unwrap()
        )
        .unwrap(),
        vec![
            Some(PartitionLabel::Text("a1".into())),
            Some(PartitionLabel::Text("b2".into())),
        ],
    );
    let _lock = random::test_guard();
    let partition = || {
        cv(
            tensor(vec![1.0, 1.0, 2.0, 2.0, 10.0, 10.0], vec![6, 1]),
            Value::String("KFold".into()),
            vec![Value::Num(2.0)],
        )
    };
    random::set_seed(4242).unwrap();
    let first = logical_output(
        block_on(test_builtin(
            None,
            partition(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        ))
        .unwrap(),
    );
    random::set_seed(4242).unwrap();
    let second = logical_output(
        block_on(test_builtin(
            None,
            partition(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        ))
        .unwrap(),
    );
    assert_eq!(first.data, second.data);
    assert_eq!(first.shape, vec![6, 2]);
    for row in 0..6 {
        assert_eq!(first.data[row] + first.data[row + 6], 1);
    }
    for group in [0usize, 2, 4] {
        assert_eq!(first.data[group] + first.data[group + 1], 1);
    }
}

#[test]
fn malformed_host_mask_metadata_rejects_before_resident_index_transfer() {
    let _lock = random::test_guard();
    let Value::Object(mut object) = cv(
        Value::Num(4.0),
        Value::String("Holdout".into()),
        vec![Value::Num(0.5)],
    ) else {
        panic!("expected partition object")
    };
    let resident_index = || {
        Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("unregistered resident index fixture")
                .handle,
        )
    };
    object.replace_property_unbounded("Type".into(), Value::String("not-a-partition".into()));
    let error = block_on(test_builtin(
        None,
        Value::Object(object.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![resident_index()]),
    ))
    .expect_err("invalid type must win before transfer");
    assert_eq!(
        error.identifier(),
        Some("RunMat:cvpartition:InvalidArgument")
    );
    object.replace_property_unbounded("Type".into(), Value::String("holdout".into()));
    crate::builtins::common::test_support::remove_object_property_unbounded(
        &mut object,
        ACTIVE_ROWS_PROPERTY,
    );
    let error = block_on(training_builtin(
        None,
        Value::Object(object),
        crate::builtins::common::test_support::builtin_arguments(vec![resident_index()]),
    ))
    .expect_err("missing active rows must win before transfer");
    assert_eq!(
        error.identifier(),
        Some("RunMat:cvpartition:InvalidArgument")
    );
}

#[test]
fn invalid_host_fold_control_wins_before_resident_observation_transfer() {
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![6, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("unregistered resident observation fixture")
            .handle,
    );
    let error = block_on(cvpartition_builtin(
        None,
        resident,
        Value::String("KFold".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .expect_err("invalid KFold control must win before transfer");
    assert_eq!(
        error.identifier(),
        Some("RunMat:cvpartition:InvalidArgument")
    );
}

#[test]
fn test_and_training_typed_indices_follow_compatibility_mode_and_decode_exactly() {
    let _guard = random::test_guard();
    random::set_seed(2030).unwrap();
    let partition = cv(
        Value::Num(8.0),
        Value::String("KFold".into()),
        vec![Value::Num(4.0)],
    );
    let index = cleared_int_tensor(IntegerStorage::U64(vec![1, 3]), vec![1, 2]);

    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let test_error = block_on(test_builtin(
            None,
            partition.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![index.clone()]),
        ))
        .expect_err("typed test index must be gated");
        assert_eq!(
            test_error.identifier(),
            TEST_INTEGER_INDEX_EXTENSION.error_identifier
        );
        let training_error = block_on(training_builtin(
            None,
            partition.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![index.clone()]),
        ))
        .expect_err("typed training index must be gated");
        assert_eq!(
            training_error.identifier(),
            TRAINING_INTEGER_INDEX_EXTENSION.error_identifier
        );
    }

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let test = logical_output(
        block_on(test_builtin(
            None,
            partition.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![index.clone()]),
        ))
        .expect("typed test indices"),
    );
    let training = logical_output(
        block_on(training_builtin(
            None,
            partition,
            crate::builtins::common::test_support::builtin_arguments(vec![index]),
        ))
        .expect("typed training indices"),
    );
    assert_eq!(test.shape, vec![8, 2]);
    assert_eq!(training.shape, vec![8, 2]);
    for (test, training) in test.data.iter().zip(training.data.iter()) {
        assert_eq!(*test == 0, *training != 0);
    }
}

#[test]
fn typed_stratify_aliases_have_one_runtime_decoder() {
    assert!(!scalar_bool(
        &Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        "Stratify"
    )
    .unwrap());
    assert!(scalar_bool(
        &Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        "Stratify"
    )
    .unwrap());
    assert!(scalar_bool(
        &cleared_int_tensor(IntegerStorage::U64(vec![1]), vec![1, 1]),
        "Stratify"
    )
    .unwrap());
    assert!(scalar_bool(
        &cleared_int_tensor(IntegerStorage::U64(vec![(1u64 << 53) + 1]), vec![1, 1]),
        "Stratify"
    )
    .is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn scalar_controls_do_not_fuse_surrogates_across_character_rows() {
    let rows = Value::CharArray(
        CharArray::from_code_units(
            vec![
                Utf16CodeUnit::from_raw(0xd83d),
                Utf16CodeUnit::from_raw(0xde80),
            ],
            2,
            1,
        )
        .expect("valid two-row character array"),
    );
    let bool_error = scalar_bool(&rows, "Stratify").expect_err("two rows are not a scalar");
    assert_eq!(
        bool_error.identifier(),
        Some("RunMat:cvpartition:InvalidArgument")
    );
    let text_error =
        scalar_text(&rows, "partition kind").expect_err("two rows are not a text scalar");
    assert_eq!(
        text_error.identifier(),
        Some("RunMat:cvpartition:InvalidArgument")
    );
}

#[test]
fn holdout_fraction_and_stratification_work() {
    let _lock = random::test_guard();
    random::set_seed(2027).unwrap();
    let labels = Value::StringArray(
        StringArray::new(
            vec!["a".into(), "a".into(), "b".into(), "b".into()],
            vec![4, 1],
        )
        .unwrap(),
    );
    let partition = cv(
        labels,
        Value::String("Holdout".into()),
        vec![
            Value::Num(0.5),
            Value::String("Stratify".into()),
            Value::Bool(true),
        ],
    );
    let test_mask = logical_output(
        block_on(test_builtin(
            None,
            partition,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("test"),
    );
    assert_eq!(test_mask.shape, vec![4, 1]);
    assert_eq!(test_mask.data.iter().filter(|flag| **flag != 0).count(), 2);
    assert_eq!(
        test_mask.data[..2]
            .iter()
            .filter(|flag| **flag != 0)
            .count(),
        1
    );
    assert_eq!(
        test_mask.data[2..]
            .iter()
            .filter(|flag| **flag != 0)
            .count(),
        1
    );
}

#[test]
fn leaveout_and_resubstitution_have_expected_shapes() {
    let leaveout = cv(
        Value::Num(3.0),
        Value::String("Leaveout".into()),
        Vec::new(),
    );
    let all = logical_output(
        block_on(test_builtin(
            None,
            leaveout,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        ))
        .expect("leaveout all"),
    );
    assert_eq!(all.shape, vec![3, 3]);
    assert_eq!(all.data, vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);

    let resub = cv(
        Value::Num(3.0),
        Value::String("Resubstitution".into()),
        Vec::new(),
    );
    let train = logical_output(
        block_on(training_builtin(
            None,
            resub,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("train"),
    );
    assert_eq!(train.shape, vec![3, 1]);
    assert_eq!(train.data, vec![1, 1, 1]);

    let resub = cv(
        Value::Num(3.0),
        Value::String("Resubstitution".into()),
        Vec::new(),
    );
    let test = logical_output(
        block_on(test_builtin(
            None,
            resub,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("test"),
    );
    assert_eq!(test.shape, vec![3, 1]);
    assert_eq!(test.data, vec![1, 1, 1]);
}

#[test]
fn custom_partition_preserves_columns() {
    let partition = cv(
        Value::String("CustomPartition".into()),
        logical(vec![1, 0, 0, 1, 0, 1], vec![3, 2]),
        Vec::new(),
    );
    let all = logical_output(
        block_on(test_builtin(
            None,
            partition,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        ))
        .expect("custom all"),
    );
    assert_eq!(all.shape, vec![3, 2]);
    assert_eq!(all.data, vec![1, 0, 0, 1, 0, 1]);
}

#[test]
fn custom_integer_vector_expands_to_test_set_columns() {
    let partition = cv(
        Value::String("CustomPartition".into()),
        tensor(vec![1.0, 2.0, 2.0, 1.0], vec![1, 4]),
        Vec::new(),
    );
    let all = logical_output(
        block_on(test_builtin(
            None,
            partition,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        ))
        .expect("custom integer all"),
    );
    assert_eq!(all.shape, vec![4, 2]);
    assert_eq!(all.data, vec![1, 0, 0, 1, 0, 1, 1, 0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn custom_numeric_singletons_preserve_rank_zero_and_rank_one_masks() {
    for shape in [vec![], vec![1]] {
        let partition = cv(
            Value::String("CustomPartition".into()),
            tensor(vec![1.0], shape.clone()),
            Vec::new(),
        );
        let mask = logical_output(
            block_on(test_builtin(
                None,
                partition,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
        );
        assert_eq!(mask.shape, vec![1, 1]);
        assert_eq!(mask.data, vec![1]);

        let partition = cv(
            Value::String("CustomPartition".into()),
            cleared_int_tensor(IntegerStorage::U8(vec![0]), shape),
            Vec::new(),
        );
        let mask = logical_output(
            block_on(test_builtin(
                None,
                partition,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
        );
        assert_eq!(mask.shape, vec![1, 1]);
        assert_eq!(mask.data, vec![0]);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn numeric_partition_input_preserves_scalar_count_and_rank_one_labels() {
    let scalar_count = cv(
        tensor(vec![2.0], vec![]),
        Value::String("Resubstitution".into()),
        Vec::new(),
    );
    let label_vector = cv(
        tensor(vec![8.0, 9.0], vec![2]),
        Value::String("Resubstitution".into()),
        Vec::new(),
    );
    for partition in [scalar_count, label_vector] {
        let mask = logical_output(
            block_on(test_builtin(
                None,
                partition,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
        );
        assert_eq!(mask.shape, vec![2, 1]);
        assert_eq!(mask.data, vec![1, 1]);
    }
}

#[test]
fn custom_integer_vector_reads_typed_integer_storage_exactly() {
    let partition = cv(
        Value::String("CustomPartition".into()),
        cleared_int_tensor(IntegerStorage::U8(vec![1, 2, 2, 1]), vec![1, 4]),
        Vec::new(),
    );
    let all = logical_output(
        block_on(test_builtin(
            None,
            partition,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        ))
        .expect("custom integer all"),
    );
    assert_eq!(all.shape, vec![4, 2]);
    assert_eq!(all.data, vec![1, 0, 0, 1, 0, 1, 1, 0]);

    let partition = cv(
        Value::String("CustomPartition".into()),
        cleared_int_tensor(IntegerStorage::U8(vec![1, 0, 1, 0]), vec![1, 4]),
        Vec::new(),
    );
    let mask = logical_output(
        block_on(test_builtin(
            None,
            partition,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("test"),
    );
    assert_eq!(mask.shape, vec![4, 1]);
    assert_eq!(mask.data, vec![1, 0, 1, 0]);

    let err = block_on(cvpartition_builtin(
        None,
        Value::String("CustomPartition".into()),
        tensor(vec![usize::MAX as f64], vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:cvpartition:InvalidArgument"));
}

#[test]
fn custom_logical_row_vector_is_single_test_set() {
    let partition = cv(
        Value::String("CustomPartition".into()),
        logical(vec![1, 0, 1, 0], vec![1, 4]),
        Vec::new(),
    );
    let Value::Object(object) = &partition else {
        panic!("object");
    };
    assert_eq!(
        object.properties().get("NumObservations"),
        Some(&Value::Num(4.0))
    );
    assert_eq!(
        object.properties().get("NumTestSets"),
        Some(&Value::Num(1.0))
    );
    let mask = logical_output(
        block_on(test_builtin(
            None,
            partition,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("test"),
    );
    assert_eq!(mask.shape, vec![4, 1]);
    assert_eq!(mask.data, vec![1, 0, 1, 0]);
}

#[test]
fn invalid_kfold_rejects_large_and_equal_fold_count() {
    let err = block_on(cvpartition_builtin(
        None,
        Value::Num(3.0),
        Value::String("KFold".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(4.0)]),
    ))
    .unwrap_err();
    assert!(err.message.contains("KFold"));

    let err = block_on(cvpartition_builtin(
        None,
        Value::Num(3.0),
        Value::String("KFold".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.0)]),
    ))
    .unwrap_err();
    assert!(err.message.contains("KFold"));
}

#[test]
fn selected_fold_vector_returns_one_column_per_index() {
    let _lock = random::test_guard();
    random::set_seed(2028).unwrap();
    let partition = cv(
        Value::Num(6.0),
        Value::String("KFold".into()),
        vec![Value::Num(3.0)],
    );
    let selected = logical_output(
        block_on(test_builtin(
            None,
            partition,
            crate::builtins::common::test_support::builtin_arguments(vec![tensor(
                vec![1.0, 3.0],
                vec![1, 2],
            )]),
        ))
        .expect("selected folds"),
    );
    assert_eq!(selected.shape, vec![6, 2]);
    assert_eq!(
        selected.data[..6].iter().filter(|flag| **flag != 0).count(),
        2
    );
    assert_eq!(
        selected.data[6..].iter().filter(|flag| **flag != 0).count(),
        2
    );
}

#[test]
fn missing_numeric_stratification_rows_are_discarded_from_masks() {
    let _lock = random::test_guard();
    random::set_seed(2029).unwrap();
    let partition = cv(
        tensor(vec![1.0, f64::NAN, 1.0, 2.0, 2.0], vec![5, 1]),
        Value::String("KFold".into()),
        vec![Value::Num(2.0)],
    );
    let all_test = logical_output(
        block_on(test_builtin(
            None,
            partition.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        ))
        .expect("test all"),
    );
    assert_eq!(all_test.shape, vec![5, 2]);
    assert_eq!(all_test.data[1], 0);
    assert_eq!(all_test.data[6], 0);

    let all_train = logical_output(
        block_on(training_builtin(
            None,
            partition,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        ))
        .expect("training all"),
    );
    assert_eq!(all_train.data[1], 0);
    assert_eq!(all_train.data[6], 0);
}

#[test]
fn typed_integer_partition_counts_and_indices_are_exact() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = PartitionInput::from_value(Value::Int(runmat_value::IntValue::U16(6))).unwrap();
    assert_eq!(input.n, 6);
    let input =
        PartitionInput::from_value(cleared_int_tensor(IntegerStorage::U16(vec![6]), vec![1, 1]))
            .unwrap();
    assert_eq!(input.n, 6);
    assert_eq!(
        positive_integer(&Value::Int(runmat_value::IntValue::U8(3)), "KFold").unwrap(),
        3
    );
    assert_eq!(
        positive_integer(
            &cleared_int_tensor(IntegerStorage::U8(vec![3]), vec![1, 1]),
            "KFold"
        )
        .unwrap(),
        3
    );
    if usize::BITS == 64 {
        let wide = (1u64 << 53) + 1;
        assert_eq!(
            positive_integer(
                &cleared_int_tensor(IntegerStorage::U64(vec![wide]), vec![1, 1]),
                "KFold",
            )
            .unwrap(),
            wide as usize,
        );
    }
    assert_eq!(
        holdout_count(&Value::Int(runmat_value::IntValue::U8(2)), 6).unwrap(),
        2
    );
    assert_eq!(
        holdout_count(
            &cleared_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
            6
        )
        .unwrap(),
        2
    );
    assert_eq!(
        selected_indices(&Value::Int(runmat_value::IntValue::U8(3)), 3).unwrap(),
        vec![2]
    );
    assert_eq!(
        selected_indices(
            &cleared_int_tensor(IntegerStorage::U8(vec![3]), vec![1, 1]),
            3
        )
        .unwrap(),
        vec![2]
    );

    let partition = cv(
        cleared_int_tensor(IntegerStorage::U16(vec![6]), vec![1, 1]),
        Value::from("KFold"),
        vec![cleared_int_tensor(IntegerStorage::U8(vec![3]), vec![1, 1])],
    );
    let selected = logical_output(
        block_on(test_builtin(
            None,
            partition,
            crate::builtins::common::test_support::builtin_arguments(vec![cleared_int_tensor(
                IntegerStorage::U8(vec![3]),
                vec![1, 1],
            )]),
        ))
        .unwrap(),
    );
    assert_eq!(selected.shape, vec![6, 1]);

    for value in [
        Value::Int(runmat_value::IntValue::I8(-1)),
        Value::Num(1.5),
        Value::Num(usize::MAX as f64),
        Value::Num(usize::MAX as f64 + 1.0),
    ] {
        assert!(positive_integer(&value, "KFold").is_err());
    }
    assert!(selected_indices(&Value::Int(runmat_value::IntValue::I8(-1)), 3).is_err());
}

#[test]
fn cvpartition_extensions_are_independently_mode_gated() {
    let integer_scalar = || Value::Int(runmat_value::IntValue::U16(6));
    let integer_labels = || cleared_int_tensor(IntegerStorage::U8(vec![1, 1, 2, 2]), vec![4, 1]);
    let cases = [
        (
            integer_scalar(),
            Value::from("KFold"),
            vec![Value::Num(2.0)],
            CVPARTITION_INTEGER_N_EXTENSION.error_identifier,
        ),
        (
            Value::Num(6.0),
            Value::from("KFold"),
            vec![Value::Int(runmat_value::IntValue::U8(2))],
            CVPARTITION_INTEGER_CONTROL_EXTENSION.error_identifier,
        ),
        (
            integer_labels(),
            Value::from("KFold"),
            vec![Value::Num(2.0)],
            CVPARTITION_INTEGER_STRATIFICATION_EXTENSION.error_identifier,
        ),
        (
            Value::from("CustomPartition"),
            integer_labels(),
            Vec::new(),
            CVPARTITION_INTEGER_CUSTOM_EXTENSION.error_identifier,
        ),
        (
            Value::Num(6.0),
            Value::from("KFold"),
            vec![Value::Num(2.0), Value::from("Stratify"), Value::Num(1.0)],
            CVPARTITION_BOOLEAN_ALIAS_EXTENSION.error_identifier,
        ),
    ];
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    for (first, second, rest, identifier) in cases {
        let error = block_on(cvpartition_builtin(
            None,
            first,
            second,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .expect_err("strict rejection");
        assert_eq!(error.identifier(), identifier);
    }
}

#[test]
fn scalar_string_array_kind_cannot_bypass_typed_control_gate() {
    let kind = Value::StringArray(
        StringArray::new(vec!["KFold".into()], vec![1, 1]).expect("scalar string"),
    );
    let control = cleared_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]);
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(cvpartition_builtin(
        None,
        Value::Num(6.0),
        kind,
        crate::builtins::common::test_support::builtin_arguments(vec![control]),
    ))
    .expect_err("typed control gate");
    assert_eq!(
        error.identifier(),
        CVPARTITION_INTEGER_CONTROL_EXTENSION.error_identifier
    );
}

#[test]
fn wide_integer_stratification_keys_remain_distinct_and_custom_validation_is_exact() {
    let wide = 9_007_199_254_740_992_i64;
    for value in [
        cleared_int_tensor(
            IntegerStorage::I64(vec![wide, wide + 1, wide, wide + 1]),
            vec![4, 1],
        ),
        cleared_int_tensor(
            IntegerStorage::U64(vec![
                wide as u64,
                wide as u64 + 1,
                wide as u64,
                wide as u64 + 1,
            ]),
            vec![4, 1],
        ),
    ] {
        let input = PartitionInput::from_value(value).expect("exact integer labels");
        let labels = input.labels.expect("stratification labels");
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[1], labels[3]);
        assert_ne!(labels[0], labels[1]);
    }

    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in [
        IntegerStorage::I64(vec![wide, wide + 1]),
        IntegerStorage::U64(vec![wide as u64, wide as u64 + 1]),
    ] {
        let error = block_on(cvpartition_builtin(
            None,
            Value::from("CustomPartition"),
            cleared_int_tensor(storage, vec![2, 1]),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("wide custom identifiers reject structurally");
        assert!(error.message().contains("CustomPartition"));
    }
}

#[test]
fn resident_integer_roles_gate_before_gather() {
    fn resident(shape: Vec<usize>, _fixture_id: u64) -> Value {
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                shape,
                runmat_accelerate_api::NumericElementType::U64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered integer fixture")
            .handle;
        Value::GpuTensor(handle)
    }
    let cases = [
        (
            resident(vec![1, 1], u64::MAX - 510),
            Value::from("KFold"),
            vec![Value::Num(2.0)],
            CVPARTITION_INTEGER_N_EXTENSION.error_identifier,
        ),
        (
            Value::Num(6.0),
            Value::from("KFold"),
            vec![resident(vec![1, 1], u64::MAX - 511)],
            CVPARTITION_INTEGER_CONTROL_EXTENSION.error_identifier,
        ),
        (
            resident(vec![4, 1], u64::MAX - 512),
            Value::from("KFold"),
            vec![Value::Num(2.0)],
            CVPARTITION_INTEGER_STRATIFICATION_EXTENSION.error_identifier,
        ),
        (
            Value::from("CustomPartition"),
            resident(vec![4, 1], u64::MAX - 513),
            Vec::new(),
            CVPARTITION_INTEGER_CUSTOM_EXTENSION.error_identifier,
        ),
    ];
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    for (first, second, rest, identifier) in cases {
        let error = block_on(cvpartition_builtin(
            None,
            first,
            second,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
        .expect_err("resident role rejects at extension gate");
        assert_eq!(error.identifier(), identifier);
    }
}

#[test]
fn cvpartition_logical_stratify_is_documented_and_rng_respects_seed() {
    let _lock = random::test_guard();
    let call = |seed| {
        random::set_seed(seed).unwrap();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let partition = block_on(cvpartition_builtin(
            None,
            Value::Num(12.0),
            Value::from("KFold"),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(3.0),
                Value::from("Stratify"),
                Value::Bool(false),
            ]),
        ))
        .unwrap();
        logical_output(
            block_on(test_builtin(
                None,
                partition,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::from("all")]),
            ))
            .unwrap(),
        )
        .data
    };
    let first = call(101);
    let same = call(101);
    let different = call(102);
    assert_eq!(first, same);
    assert_ne!(first, different);
}

#[test]
fn cvpartition_integer_capabilities_cover_all_classes() {
    assert_eq!(CVPARTITION_INTEGER_CAPABILITIES.len(), 4);
    assert!(CVPARTITION_INTEGER_CAPABILITIES
        .iter()
        .all(|capability| capability.inputs[0].classes.len() == 8));
}
