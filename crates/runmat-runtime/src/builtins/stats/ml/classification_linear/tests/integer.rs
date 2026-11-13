use super::*;

#[test]
fn grouping_labels_round_trip_all_classes_and_text_carriers() {
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
    use runmat_value::StringElement;
    let elements = vec![
        StringElement::Missing,
        StringElement::text("<missing>"),
        StringElement::text("<MISSING>"),
        StringElement::text(""),
        StringElement::text("present"),
    ];
    let strings = || {
        Value::StringArray(
            StringArray::from_elements_unbounded(elements.clone(), vec![5, 1]).unwrap(),
        )
    };
    let scalar_cells = Value::Cell(
        runmat_value::CellArray::new(elements.iter().cloned().map(Value::String).collect(), 5, 1)
            .unwrap(),
    );
    let array_cells = Value::Cell(
        runmat_value::CellArray::new(
            elements
                .iter()
                .cloned()
                .map(|element| {
                    Value::StringArray(
                        StringArray::from_elements_unbounded(vec![element], vec![1, 1]).unwrap(),
                    )
                })
                .collect(),
            5,
            1,
        )
        .unwrap(),
    );
    for input in [strings(), scalar_cells, array_cells] {
        let labels = labels_from_value(input, "Y").unwrap();
        assert!(is_missing_label(&labels.labels[0]));
        assert!(labels.labels[1..]
            .iter()
            .all(|label| !is_missing_label(label)));
        assert_eq!(unique_labels(&labels.labels, labels.kind).len(), 4);
        let Value::StringArray(round_trip) = labels_value(&labels.labels, labels.kind).unwrap()
        else {
            panic!("expected string labels");
        };
        assert_eq!(round_trip.data(), elements.as_slice());
    }
}

#[tokio::test]
async fn fitclinear_preserves_adjacent_wide_uint64_classes() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let base = 9_007_199_254_740_992_u64;
    let labels = IntegerStorage::U64(vec![base, base, base + 1, base + 1]);
    let model = fitclinear_builtin(
        None,
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            poisoned_int_tensor(labels, vec![4, 1]),
            Value::from("Lambda"),
            Value::Num(0.0),
        ]),
    )
    .await
    .unwrap();
    let Value::Object(object) = model else {
        panic!("expected model");
    };
    assert_eq!(
        object
            .properties()
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
        crate::call::FunctionArgumentView::from_slice(&[]),
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
        None,
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), vec![4, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            vec![0.0, 0.0, 1.0, 1.0],
            vec![4, 1],
        )]),
    )
    .await
    .expect_err("integer predictor extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FitclinearIntegerPredictorExtension")
    );

    let error = fitclinear_builtin(
        None,
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
            Value::from("IterationLimit"),
            poisoned_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
        ]),
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
    let error = fitclinear_builtin(
        None,
        poison,
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            vec![0.0, 0.0, 1.0, 1.0],
            vec![4, 1],
        )]),
    )
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
        None,
        poisoned_int_tensor(IntegerStorage::U64(vec![0, 1, 2, inexact]), vec![4, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            vec![0.0, 0.0, 1.0, 1.0],
            vec![4, 1],
        )]),
    )
    .await
    .expect_err("inexact predictor");
    assert!(error.message.contains("exactly representable"));

    let error = fitclinear_builtin(
        None,
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
            Value::from("Weights"),
            poisoned_int_tensor(IntegerStorage::U64(vec![1, 1, 1, inexact]), vec![4, 1]),
        ]),
    )
    .await
    .expect_err("inexact weights");
    assert!(error.message.contains("exactly representable"));
}
