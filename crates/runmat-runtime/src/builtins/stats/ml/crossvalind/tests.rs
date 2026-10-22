use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

fn call(args: Vec<Value>, outputs: Option<usize>) -> BuiltinResult<Value> {
    let _guard = crate::output_count::push_output_count(outputs).expect("output-count scope");
    let args =
        crate::execution::RootedValues::new(args).map_err(|error| error.into_runtime_error())?;
    block_on(crossvalind_builtin(None, args.into()))
}

fn logical_data(value: Value) -> Vec<u8> {
    match value {
        Value::LogicalArray(array) => array.data.into_vec(),
        other => panic!("expected logical array, got {other:?}"),
    }
}

fn tensor_data(value: Value) -> Vec<f64> {
    match value {
        Value::Tensor(tensor) => tensor
            .into_numeric_storage()
            .expect("valid tensor storage")
            .materialize_f64(),
        Value::Num(number) => vec![number],
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn output_list(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique output list")
            .try_into_unbounded_vec()
            .expect("unbounded output list"),
        other => panic!("expected output list, got {other:?}"),
    }
}

fn int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new_integer(storage, vec![rows, cols]).unwrap())
}

#[test]
fn holdout_reads_every_integer_storage_variant() {
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
        assert_eq!(holdout_count(&int_tensor(storage, 1, 1), 6).unwrap(), 2);
    }
}

#[test]
fn holdout_fraction_requires_two_observations_without_panicking() {
    let error = holdout_count(&Value::Num(0.5), 1).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:crossvalind:InvalidArgument")
    );
    assert!(holdout_count(&Value::Num(0.5), 2).is_ok());
}

#[test]
fn holdout_default_output_is_training_mask() {
    let _rng_guard = random::test_guard();
    random::set_seed(2026).expect("seed");
    let train = logical_data(
        call(
            vec![Value::from("HoldOut"), Value::Num(10.0), Value::Num(0.3)],
            None,
        )
        .expect("crossvalind"),
    );
    assert_eq!(train.len(), 10);
    assert_eq!(train.iter().filter(|flag| **flag != 0).count(), 7);
}

#[test]
fn holdout_two_outputs_are_complements() {
    let _rng_guard = random::test_guard();
    random::set_seed(7).expect("seed");
    let outputs = output_list(
        call(
            vec![Value::from("holdout"), Value::Num(12.0), Value::Num(4.0)],
            Some(2),
        )
        .expect("crossvalind"),
    );
    let train = logical_data(outputs[0].clone());
    let test = logical_data(outputs[1].clone());
    assert_eq!(test.iter().filter(|flag| **flag != 0).count(), 4);
    for (a, b) in train.iter().zip(test.iter()) {
        assert_ne!(*a != 0, *b != 0);
    }
}

#[test]
fn kfold_returns_fold_indices_covering_all_folds() {
    let _rng_guard = random::test_guard();
    random::set_seed(11).expect("seed");
    let folds = tensor_data(
        call(
            vec![Value::from("KFold"), Value::Num(9.0), Value::Num(3.0)],
            None,
        )
        .expect("crossvalind"),
    );
    assert_eq!(folds.len(), 9);
    for fold in 1..=3 {
        assert!(folds
            .iter()
            .any(|value| (*value - fold as f64).abs() <= EPS));
    }
}

#[test]
fn classes_option_stratifies_holdout() {
    let _rng_guard = random::test_guard();
    random::set_seed(13).expect("seed");
    let classes = int_tensor(IntegerStorage::U8(vec![1, 1, 1, 1, 2, 2, 2, 2]), 8, 1);
    let outputs = output_list(
        call(
            vec![
                Value::from("HoldOut"),
                Value::Num(8.0),
                Value::Num(0.5),
                Value::from("Classes"),
                classes,
            ],
            Some(2),
        )
        .expect("crossvalind"),
    );
    let test = logical_data(outputs[1].clone());
    assert_eq!(test[..4].iter().filter(|flag| **flag != 0).count(), 2);
    assert_eq!(test[4..].iter().filter(|flag| **flag != 0).count(), 2);
}

#[test]
fn numeric_class_order_retains_seed_reproducibility_and_fold_coverage() {
    let _rng_guard = random::test_guard();
    let arguments = || {
        vec![
            Value::from("KFold"),
            Value::Num(6.0),
            Value::Num(2.0),
            Value::from("Classes"),
            Value::Tensor(Tensor::new(vec![1.0, 1.0, 2.0, 2.0, 10.0, 10.0], vec![6, 1]).unwrap()),
        ]
    };
    random::set_seed(4242).unwrap();
    let first = tensor::tensor_values_f64(&match call(arguments(), None).unwrap() {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected fold tensor, got {other:?}"),
    });
    random::set_seed(4242).unwrap();
    let second = tensor::tensor_values_f64(&match call(arguments(), None).unwrap() {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected fold tensor, got {other:?}"),
    });
    assert_eq!(first, second);
    for pair in first.chunks_exact(2) {
        assert_ne!(pair[0], pair[1]);
        assert!(pair.iter().all(|fold| *fold == 1.0 || *fold == 2.0));
    }
}

#[test]
fn invalid_host_fold_control_wins_before_resident_observation_transfer() {
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("unregistered resident count fixture")
            .handle,
    );
    let error = call(vec![Value::from("KFold"), resident, Value::Num(1.0)], None)
        .expect_err("invalid KFold control must win before transfer");
    assert_eq!(
        error.identifier(),
        Some("RunMat:crossvalind:InvalidArgument")
    );
}

#[test]
fn classes_keep_adjacent_wide_integer_labels_distinct() {
    let wide = 9_007_199_254_740_992_u64;
    let classes = int_tensor(
        IntegerStorage::U64(vec![wide, wide + 1, wide, wide + 1]),
        4,
        1,
    );
    let labels = classes_from_value(&classes, 4).expect("exact grouping labels");
    assert_eq!(labels[0], labels[2]);
    assert_eq!(labels[1], labels[3]);
    assert_ne!(labels[0], labels[1]);
    assert_eq!(grouped_indices(&labels).len(), 2);

    let character_classes =
        Value::CharArray(CharArray::from_unicode_scalars(vec!['a', '1', 'b', '2'], 2, 2).unwrap());
    assert_eq!(
        classes_from_value(&character_classes, 2).unwrap(),
        vec![
            Some(PartitionLabel::Text("a1".into())),
            Some(PartitionLabel::Text("b2".into())),
        ],
    );
}

#[test]
fn one_observation_accepts_scalar_numeric_class_labels() {
    assert_eq!(
        classes_from_value(&Value::Int(runmat_value::IntValue::U64(7)), 1).unwrap(),
        vec![Some(PartitionLabel::integer(&runmat_value::IntValue::U64(
            7
        )))],
    );
    assert_eq!(
        classes_from_value(&Value::Num(-0.0), 1).unwrap(),
        vec![PartitionLabel::number(0.0)],
    );
}

#[test]
fn resubstitution_returns_all_true_train_and_test() {
    let outputs = output_list(
        call(
            vec![Value::from("Resubstitution"), Value::Num(4.0)],
            Some(2),
        )
        .expect("crossvalind"),
    );
    assert_eq!(logical_data(outputs[0].clone()), vec![1, 1, 1, 1]);
    assert_eq!(logical_data(outputs[1].clone()), vec![1, 1, 1, 1]);
}

#[test]
fn invalid_forms_are_rejected() {
    let err = call(
        vec![Value::from("KFold"), Value::Num(5.0), Value::Num(2.0)],
        Some(2),
    )
    .expect_err("too many KFold outputs");
    assert_eq!(err.identifier(), Some("RunMat:crossvalind:InvalidArgument"));

    let err = call(
        vec![Value::from("HoldOut"), Value::Num(5.0), Value::Num(5.0)],
        None,
    )
    .expect_err("invalid holdout");
    assert_eq!(err.identifier(), Some("RunMat:crossvalind:InvalidArgument"));
}

#[test]
fn typed_integer_partition_counts_are_exact_and_lossy_f64_is_rejected() {
    let typed_six = Value::Int(runmat_value::IntValue::U16(6));
    assert_eq!(observation_count(&typed_six).unwrap(), 6);
    let typed_tensor_six = int_tensor(IntegerStorage::U16(vec![6]), 1, 1);
    assert_eq!(observation_count(&typed_tensor_six).unwrap(), 6);
    assert_eq!(
        positive_integer(&Value::Int(runmat_value::IntValue::U8(3)), "KFold").unwrap(),
        3
    );
    assert_eq!(
        positive_integer(&int_tensor(IntegerStorage::U8(vec![3]), 1, 1), "KFold").unwrap(),
        3
    );
    if usize::BITS == 64 {
        let wide = (1u64 << 53) + 1;
        assert_eq!(
            positive_integer(&int_tensor(IntegerStorage::U64(vec![wide]), 1, 1), "KFold").unwrap(),
            wide as usize,
        );
    }
    assert_eq!(
        holdout_count(&Value::Int(runmat_value::IntValue::U8(2)), 6).unwrap(),
        2
    );
    assert_eq!(
        holdout_count(&int_tensor(IntegerStorage::U8(vec![2]), 1, 1), 6).unwrap(),
        2
    );

    let fold_args = vec![
        Value::from("KFold"),
        typed_tensor_six,
        int_tensor(IntegerStorage::U8(vec![3]), 1, 1),
    ];
    let admitted =
        admit_host_controls(crate::call::FunctionArgumentView::from_slice(&fold_args)).unwrap();
    let folds = {
        let mut turn = random::try_acquire_turn(None, "crossvalind").unwrap();
        crossvalind_compute(
            crate::call::FunctionArgumentView::from_slice(&fold_args),
            admitted,
            Some(&mut turn),
        )
        .unwrap()
    };
    let PartitionOutput::Folds(Value::Tensor(folds)) = folds else {
        panic!("expected fold tensor");
    };
    assert_eq!(folds.shape(), &[6, 1]);

    for value in [
        Value::Int(runmat_value::IntValue::I8(-1)),
        Value::Num(1.5),
        Value::Num(usize::MAX as f64 + 1.0),
    ] {
        assert!(positive_integer(&value, "KFold").is_err());
    }
    assert!(holdout_count(&Value::Int(runmat_value::IntValue::I8(-1)), 6).is_err());
}
