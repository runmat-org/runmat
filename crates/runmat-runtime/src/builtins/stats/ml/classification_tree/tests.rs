use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

fn tensor_value(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn object(value: Value) -> ObjectInstance {
    match value {
        Value::Object(object) => object,
        other => panic!("expected object, got {other:?}"),
    }
}

fn tensor(value: &Value) -> &Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn fitctree_matrix_predicts_numeric_labels_and_scores() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1);
    let model = block_on(fitctree_builtin(
        x,
        vec![
            y,
            Value::from("MaxNumSplits"),
            Value::Num(1.0),
            Value::from("MinParentSize"),
            Value::Num(2.0),
        ],
    ))
    .unwrap();
    let output_count = crate::output_count::push_output_count(Some(4));
    let predicted = crate::builtins::stats::ml::linear_model::predict_builtin(
        model,
        tensor_value(vec![0.5, 2.5], 2, 1),
        Vec::new(),
    );
    let predicted = block_on(predicted).unwrap();
    drop(output_count);
    let Value::OutputList(values) = predicted else {
        panic!("expected output list");
    };
    assert_eq!(tensor(&values[0]).materialize_f64(), vec![0.0, 1.0]);
    assert_eq!(tensor(&values[1]).shape, vec![2, 2]);
    assert_eq!(tensor(&values[2]).shape, vec![2, 1]);
    assert_eq!(tensor(&values[3]).materialize_f64(), vec![1.0, 2.0]);
}

#[test]
fn fitctree_and_predict_read_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let model = block_on(fitctree_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1),
        vec![
            poisoned_int_tensor(IntegerStorage::U8(vec![0, 0, 1, 1]), 4, 1),
            Value::from("ClassNames"),
            poisoned_int_tensor(IntegerStorage::U8(vec![0, 1]), 2, 1),
            Value::from("Weights"),
            poisoned_int_tensor(IntegerStorage::U8(vec![1, 1, 1, 1]), 4, 1),
            Value::from("MaxNumSplits"),
            poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
            Value::from("MinLeafSize"),
            poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
            Value::from("MinParentSize"),
            poisoned_int_tensor(IntegerStorage::U8(vec![2]), 1, 1),
        ],
    ))
    .unwrap();
    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        model,
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 3]), 2, 1),
        Vec::new(),
    ))
    .unwrap();
    assert_eq!(tensor(&predicted).materialize_f64(), vec![0.0, 1.0]);
}

#[test]
fn integer_grouping_labels_round_trip_all_classes() {
    let storages = [
        IntegerStorage::I8(vec![-1, 1]),
        IntegerStorage::I16(vec![-1, 1]),
        IntegerStorage::I32(vec![-1, 1]),
        IntegerStorage::I64(vec![-1, 1]),
        IntegerStorage::U8(vec![0, 1]),
        IntegerStorage::U16(vec![0, 1]),
        IntegerStorage::U32(vec![0, 1]),
        IntegerStorage::U64(vec![0, 1]),
    ];
    for storage in storages {
        let expected = storage.clone();
        let labels = labels_from_value(
            Value::Tensor(Tensor::new_integer(storage, vec![2, 1]).unwrap()),
            "Y",
        )
        .unwrap();
        let Value::Tensor(round_trip) = labels_value(&labels.labels, labels.kind).unwrap() else {
            panic!("expected integer tensor");
        };
        assert_eq!(round_trip.integer_storage(), Some(&expected));
    }
}

#[test]
fn fitctree_preserves_adjacent_wide_uint64_classes() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let base = 9_007_199_254_740_992_u64;
    let model = block_on(fitctree_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        vec![
            poisoned_int_tensor(
                IntegerStorage::U64(vec![base, base, base + 1, base + 1]),
                4,
                1,
            ),
            Value::from("MaxNumSplits"),
            Value::Num(1.0),
            Value::from("MinParentSize"),
            Value::Num(2.0),
        ],
    ))
    .unwrap();
    let Value::Object(model_object) = &model else {
        panic!("expected model");
    };
    assert_eq!(
        model_object
            .properties
            .get("ClassNames")
            .and_then(|value| match value {
                Value::Tensor(tensor) => tensor.integer_storage(),
                _ => None,
            }),
        Some(&IntegerStorage::U64(vec![base, base + 1]))
    );
    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        model,
        tensor_value(vec![0.0, 3.0], 2, 1),
        Vec::new(),
    ))
    .unwrap();
    assert_eq!(
        match &predicted {
            Value::Tensor(tensor) => tensor.integer_storage(),
            _ => None,
        },
        Some(&IntegerStorage::U64(vec![base, base + 1]))
    );
}

#[test]
fn fitctree_strict_mode_rejects_integer_numeric_extensions() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(fitctree_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1),
        vec![tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1)],
    ))
    .expect_err("integer predictor extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FitctreeIntegerPredictorExtension")
    );

    let error = block_on(fitctree_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        vec![
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
            Value::from("MaxNumSplits"),
            poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
        ],
    ))
    .expect_err("integer control extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FitctreeIntegerNumericArgumentExtension")
    );

    let poison = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![4, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle,
    );
    let error = block_on(fitctree_builtin(
        poison,
        vec![tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1)],
    ))
    .expect_err("resident gate before gather");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FitctreeResidentInputExtension")
    );
}

#[test]
fn fitctree_rejects_inexact_integer_predictors_and_weights() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let inexact = 9_007_199_254_740_993_u64;
    let error = block_on(fitctree_builtin(
        poisoned_int_tensor(IntegerStorage::U64(vec![0, 1, 2, inexact]), 4, 1),
        vec![tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1)],
    ))
    .expect_err("inexact predictor");
    assert!(error.message.contains("exactly representable"));

    let error = block_on(fitctree_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        vec![
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
            Value::from("Weights"),
            poisoned_int_tensor(IntegerStorage::U64(vec![1, 1, 1, inexact]), 4, 1),
        ],
    ))
    .expect_err("inexact weights");
    assert!(error.message.contains("exactly representable"));
}

#[test]
fn fitctree_retains_rows_with_partially_missing_predictors() {
    let model = block_on(fitctree_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0, f64::NAN, 10.0, 11.0, 12.0], 4, 2),
        vec![
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
            Value::from("MaxNumSplits"),
            Value::Num(1.0),
            Value::from("MinParentSize"),
            Value::Num(2.0),
        ],
    ))
    .unwrap();
    let Value::Object(object) = model else {
        panic!("expected model");
    };
    assert_eq!(
        object.properties.get("NumObservations"),
        Some(&Value::Num(4.0))
    );
}

#[test]
fn fitctree_table_response_name_predicts_string_labels() {
    let table = crate::builtins::table::table_from_columns(
        vec!["A".into(), "Y".into()],
        vec![
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            Value::StringArray(StringArray {
                data: vec!["low".into(), "low".into(), "high".into(), "high".into()],
                shape: vec![4, 1],
                rows: 4,
                cols: 1,
            }),
        ],
    )
    .unwrap();
    let model = block_on(fitctree_builtin(
        table.clone(),
        vec![
            Value::String("Y".into()),
            Value::from("MaxNumSplits"),
            Value::Num(1.0),
            Value::from("MinParentSize"),
            Value::Num(2.0),
        ],
    ))
    .unwrap();
    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        model,
        table,
        Vec::new(),
    ))
    .unwrap();
    let Value::StringArray(labels) = predicted else {
        panic!("expected string labels");
    };
    assert_eq!(labels.shape, vec![4, 1]);
    assert_eq!(labels.data[0], "low");
    assert_eq!(labels.data[3], "high");
}

#[test]
fn fitctree_respects_class_names_and_depth_controls() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![2.0, 2.0, 5.0, 5.0], 4, 1);
    let model = object(
        block_on(fitctree_builtin(
            x,
            vec![
                y,
                Value::from("ClassNames"),
                tensor_value(vec![5.0, 2.0], 2, 1),
                Value::from("MaxNumSplits"),
                Value::Num(0.0),
            ],
        ))
        .unwrap(),
    );
    assert_eq!(model.properties.get("NumSplits"), Some(&Value::Num(0.0)));
    let classes = tensor(model.properties.get("ClassNames").unwrap());
    assert_eq!(classes.materialize_f64(), vec![5.0, 2.0]);
}

#[test]
fn fitctree_omits_nan_responses_and_requires_two_observed_classes() {
    let model = object(
        block_on(fitctree_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            vec![
                tensor_value(vec![0.0, f64::NAN, 1.0, 1.0], 4, 1),
                Value::from("MinParentSize"),
                Value::Num(2.0),
            ],
        ))
        .unwrap(),
    );
    assert_eq!(
        model.properties.get("NumObservations"),
        Some(&Value::Num(3.0))
    );
    assert_eq!(
        tensor(model.properties.get("ClassNames").unwrap()).materialize_f64(),
        vec![0.0, 1.0]
    );

    let err = block_on(fitctree_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        vec![
            tensor_value(vec![0.0, f64::NAN, f64::NAN, 0.0], 4, 1),
            Value::from("ClassNames"),
            tensor_value(vec![0.0, 1.0], 2, 1),
        ],
    ))
    .unwrap_err();
    assert!(err.message().contains("at least two observed classes"));
}

#[test]
fn fitctree_excludes_table_weight_variable_from_default_predictors() {
    let table = crate::builtins::table::table_from_columns(
        vec!["A".into(), "W".into(), "Y".into()],
        vec![
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            tensor_value(vec![1.0, 2.0, 1.0, 2.0], 4, 1),
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
        ],
    )
    .unwrap();
    let model = object(
        block_on(fitctree_builtin(
            table,
            vec![Value::from("Y"), Value::from("Weights"), Value::from("W")],
        ))
        .unwrap(),
    );
    assert_eq!(
        model.properties.get("NumPredictors"),
        Some(&Value::Num(1.0))
    );
    let Value::StringArray(names) = model.properties.get("PredictorNames").unwrap() else {
        panic!("expected predictor names");
    };
    assert_eq!(names.data, vec!["A"]);
}

#[test]
fn fitctree_predict_rejects_too_many_outputs() {
    let model = block_on(fitctree_builtin(
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        vec![
            tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1),
            Value::from("MaxNumSplits"),
            Value::Num(1.0),
            Value::from("MinParentSize"),
            Value::Num(2.0),
        ],
    ))
    .unwrap();
    let output_count = crate::output_count::push_output_count(Some(5));
    let err = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        model,
        tensor_value(vec![0.5, 2.5], 2, 1),
        Vec::new(),
    ))
    .unwrap_err();
    drop(output_count);
    assert!(err.message().contains("too many output"));
}

#[test]
fn fitctree_rejects_unsupported_cross_validation_options() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1);
    let err = block_on(fitctree_builtin(
        x,
        vec![y, Value::from("CrossVal"), Value::String("on".into())],
    ))
    .unwrap_err();
    assert!(err.message.contains("not supported"));
}
