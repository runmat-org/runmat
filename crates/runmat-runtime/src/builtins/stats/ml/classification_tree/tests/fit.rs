use super::*;

#[test]
fn fitctree_matrix_predicts_numeric_labels_and_scores() {
    let x = tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let y = tensor_value(vec![0.0, 0.0, 1.0, 1.0], 4, 1);
    let model = block_on(fitctree_builtin(
        None,
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![
            y,
            Value::from("MaxNumSplits"),
            Value::Num(1.0),
            Value::from("MinParentSize"),
            Value::Num(2.0),
        ]),
    ))
    .unwrap();
    let output_count = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
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
    assert_eq!(tensor(&values[1]).shape(), [2, 2]);
    assert_eq!(tensor(&values[2]).shape(), [2, 1]);
    assert_eq!(tensor(&values[3]).materialize_f64(), vec![1.0, 2.0]);
}

#[test]
fn fitctree_and_predict_read_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let model = block_on(fitctree_builtin(
        None,
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
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
        ]),
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

#[test]
fn fitctree_preserves_adjacent_wide_uint64_classes() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let base = 9_007_199_254_740_992_u64;
    let model = block_on(fitctree_builtin(
        None,
        tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            poisoned_int_tensor(
                IntegerStorage::U64(vec![base, base, base + 1, base + 1]),
                4,
                1,
            ),
            Value::from("MaxNumSplits"),
            Value::Num(1.0),
            Value::from("MinParentSize"),
            Value::Num(2.0),
        ]),
    ))
    .unwrap();
    let Value::Object(model_object) = &model else {
        panic!("expected model");
    };
    assert_eq!(
        model_object
            .properties()
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
