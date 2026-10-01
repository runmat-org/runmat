use super::*;
use crate::builtins::table::table_from_columns;
use runmat_value::IntegerStorage;

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

#[test]
fn scalar_option_parsers_read_all_integer_storage_variants() {
    let storages = [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];
    for storage in storages {
        let value = poisoned_int_tensor(storage, vec![1, 1]);
        assert_eq!(numeric_scalar(&value, "Bias").unwrap(), 1.0);
        assert!(logical_scalar(&value, "FitBias").unwrap());
        assert_eq!(positive_integer(&value, "BatchLimit").unwrap(), 1);
    }
}

#[tokio::test]
async fn fitclinear_svm_predicts_binary_numeric_labels() {
    let model = fitclinear_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
        vec![
            tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
            Value::String("Lambda".into()),
            Value::Num(0.0),
        ],
    )
    .await
    .unwrap();
    let outputs = predict_classification_linear_object(
        match model {
            Value::Object(object) => object,
            _ => panic!("expected object"),
        },
        tensor(vec![0.2, 2.8], vec![2, 1]),
        Vec::new(),
    )
    .unwrap();
    match &outputs[0] {
        Value::Tensor(labels) => assert_eq!(labels.materialize_f64(), vec![0.0, 1.0]),
        other => panic!("expected labels, got {other:?}"),
    }
    match &outputs[1] {
        Value::Tensor(score) => assert_eq!(score.shape, vec![2, 2]),
        other => panic!("expected score, got {other:?}"),
    }
}

#[tokio::test]
async fn fitclinear_and_predict_read_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let model = fitclinear_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), vec![4, 1]),
        vec![
            poisoned_int_tensor(IntegerStorage::U8(vec![0, 0, 1, 1]), vec![4, 1]),
            Value::String("Lambda".into()),
            Value::Num(0.0),
            Value::String("FitBias".into()),
            poisoned_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
            Value::String("IterationLimit".into()),
            poisoned_int_tensor(IntegerStorage::U16(vec![25]), vec![1, 1]),
            Value::String("Bias".into()),
            poisoned_int_tensor(IntegerStorage::I16(vec![0]), vec![1, 1]),
        ],
    )
    .await
    .unwrap();
    let outputs = predict_classification_linear_object(
        match model {
            Value::Object(object) => object,
            _ => panic!("expected object"),
        },
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 3]), vec![2, 1]),
        Vec::new(),
    )
    .unwrap();
    match &outputs[0] {
        Value::Tensor(labels) => assert_eq!(labels.materialize_f64(), vec![0.0, 1.0]),
        other => panic!("expected labels, got {other:?}"),
    }
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

#[tokio::test]
async fn fitclinear_preserves_adjacent_wide_uint64_classes() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let base = 9_007_199_254_740_992_u64;
    let labels = IntegerStorage::U64(vec![base, base, base + 1, base + 1]);
    let model = fitclinear_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
        vec![
            poisoned_int_tensor(labels, vec![4, 1]),
            Value::from("Lambda"),
            Value::Num(0.0),
        ],
    )
    .await
    .unwrap();
    let Value::Object(object) = model else {
        panic!("expected model");
    };
    assert_eq!(
        object
            .properties
            .get("ClassNames")
            .and_then(|value| match value {
                Value::Tensor(tensor) => tensor.integer_storage(),
                _ => None,
            }),
        Some(&IntegerStorage::U64(vec![base, base + 1]))
    );
    let outputs = predict_classification_linear_object(
        object,
        tensor(vec![0.0, 3.0], vec![2, 1]),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(
        match &outputs[0] {
            Value::Tensor(tensor) => tensor.integer_storage(),
            _ => None,
        },
        Some(&IntegerStorage::U64(vec![base, base + 1]))
    );
}

#[tokio::test]
async fn fitclinear_strict_mode_rejects_integer_numeric_extensions() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = fitclinear_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), vec![4, 1]),
        vec![tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1])],
    )
    .await
    .expect_err("integer predictor extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FitclinearIntegerPredictorExtension")
    );

    let error = fitclinear_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
        vec![
            tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
            Value::from("IterationLimit"),
            poisoned_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
        ],
    )
    .await
    .expect_err("integer control extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FitclinearIntegerNumericArgumentExtension")
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
    let error = fitclinear_builtin(poison, vec![tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1])])
        .await
        .expect_err("resident gate before gather");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FitclinearResidentInputExtension")
    );
}

#[tokio::test]
async fn fitclinear_rejects_inexact_integer_predictors_and_weights() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let inexact = 9_007_199_254_740_993_u64;
    let error = fitclinear_builtin(
        poisoned_int_tensor(IntegerStorage::U64(vec![0, 1, 2, inexact]), vec![4, 1]),
        vec![tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1])],
    )
    .await
    .expect_err("inexact predictor");
    assert!(error.message.contains("exactly representable"));

    let error = fitclinear_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
        vec![
            tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
            Value::from("Weights"),
            poisoned_int_tensor(IntegerStorage::U64(vec![1, 1, 1, inexact]), vec![4, 1]),
        ],
    )
    .await
    .expect_err("inexact weights");
    assert!(error.message.contains("exactly representable"));
}

#[tokio::test]
async fn fitclinear_logistic_returns_fitinfo_and_probability_scores() {
    let _guard = crate::output_count::push_output_count(Some(2));
    let value = fitclinear_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
        vec![
            tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
            Value::String("Learner".into()),
            Value::String("logistic".into()),
            Value::String("Lambda".into()),
            Value::Num(0.0),
        ],
    )
    .await
    .unwrap();
    let (object, info) = match value {
        Value::OutputList(mut outputs) => {
            let info = outputs.pop().unwrap();
            let object = match outputs.pop().unwrap() {
                Value::Object(object) => object,
                other => panic!("expected object, got {other:?}"),
            };
            (object, info)
        }
        other => panic!("expected outputs, got {other:?}"),
    };
    assert!(matches!(info, Value::Struct(_)));
    let outputs = predict_classification_linear_object(
        object,
        tensor(vec![0.2, 2.8], vec![2, 1]),
        Vec::new(),
    )
    .unwrap();
    match &outputs[1] {
        Value::Tensor(score) => {
            assert_eq!(score.shape, vec![2, 2]);
            assert!((score.materialize_f64()[0] + score.materialize_f64()[2] - 1.0).abs() < 1.0e-8);
        }
        other => panic!("expected score, got {other:?}"),
    }
}

#[tokio::test]
async fn fitclinear_observations_in_columns_and_text_classnames() {
    let model = fitclinear_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]),
        vec![
            Value::StringArray(StringArray {
                data: vec!["low".into(), "low".into(), "high".into(), "high".into()],
                shape: vec![4, 1],
                rows: 4,
                cols: 1,
            }),
            Value::String("ObservationsIn".into()),
            Value::String("columns".into()),
            Value::String("ClassNames".into()),
            Value::StringArray(StringArray {
                data: vec!["low".into(), "high".into()],
                shape: vec![2, 1],
                rows: 2,
                cols: 1,
            }),
        ],
    )
    .await
    .unwrap();
    let outputs = predict_classification_linear_object(
        match model {
            Value::Object(object) => object,
            _ => panic!("expected object"),
        },
        tensor(vec![0.2, 2.8], vec![1, 2]),
        vec![
            Value::String("ObservationsIn".into()),
            Value::String("columns".into()),
        ],
    )
    .unwrap();
    match &outputs[0] {
        Value::StringArray(labels) => assert_eq!(labels.data, vec!["low", "high"]),
        other => panic!("expected string labels, got {other:?}"),
    }
}

#[tokio::test]
async fn fitclinear_rejects_multiclass_response() {
    let err = fitclinear_builtin(
        tensor(vec![0.0, 1.0, 2.0], vec![3, 1]),
        vec![tensor(vec![0.0, 1.0, 2.0], vec![3, 1])],
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("exactly two classes"));
}

#[tokio::test]
async fn fitclinear_table_external_y_honors_response_name() {
    let table = table_from_columns(
        vec!["A".into(), "B".into()],
        vec![
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
            tensor(vec![1.0, 1.0, 1.0, 1.0], vec![4, 1]),
        ],
    )
    .unwrap();
    let model = fitclinear_builtin(
        table,
        vec![
            tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
            Value::String("ResponseName".into()),
            Value::String("Outcome".into()),
        ],
    )
    .await
    .unwrap();
    match model {
        Value::Object(object) => {
            assert_eq!(
                object.properties.get("ResponseName"),
                Some(&Value::String("Outcome".into()))
            );
        }
        other => panic!("expected object, got {other:?}"),
    }
}
