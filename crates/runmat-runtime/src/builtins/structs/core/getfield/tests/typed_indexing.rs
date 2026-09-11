use super::*;

#[test]
fn getfield_vector_index_selector_reads_typed_integer_storage_exactly() {
    let mut st = StructValue::new();
    st.fields.insert(
        "values".to_string(),
        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
    );
    let selector =
        Tensor::new_integer(IntegerStorage::U64(vec![2, 1]), vec![1, 2]).expect("selector");
    let index =
        CellArray::new_with_shape(vec![Value::Tensor(selector)], vec![1, 1]).expect("index");
    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(index)],
    )
    .expect("vector index");
    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 2]);
            assert_eq!(tensor.materialize_f64(), vec![20.0, 10.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn getfield_vector_index_preserves_typed_integer_target_storage() {
    let values = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]),
        vec![1, 2],
    )
    .expect("typed target");
    let mut st = StructValue::new();
    st.fields
        .insert("values".to_string(), Value::Tensor(values));

    let selector =
        Tensor::new_integer(IntegerStorage::U64(vec![2, 1]), vec![1, 2]).expect("selector");
    let index =
        CellArray::new_with_shape(vec![Value::Tensor(selector)], vec![1, 1]).expect("index");

    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(index)],
    )
    .expect("vector index");
    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 2]);
            assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::U64(vec![u64::MAX, u64::MAX - 1]))
            );
            assert_eq!(
                tensor.materialize_f64(),
                vec![u64::MAX as f64, (u64::MAX - 1) as f64]
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn getfield_vector_index_preserves_single_target_storage() {
    let values = Tensor::from_numeric_storage(NumericStorage::F32(vec![1.25, 2.5]), vec![1, 2])
        .expect("single target");
    let mut st = StructValue::new();
    st.fields
        .insert("values".to_string(), Value::Tensor(values));
    let selector =
        Tensor::new_integer(IntegerStorage::U64(vec![2, 1]), vec![1, 2]).expect("selector");
    let index =
        CellArray::new_with_shape(vec![Value::Tensor(selector)], vec![1, 1]).expect("index");

    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(index)],
    )
    .expect("vector index");
    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 2]);
            assert_eq!(
                tensor.into_numeric_storage().expect("single storage"),
                NumericStorage::F32(vec![2.5, 1.25])
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn getfield_logical_index_selectors_preserve_integer_target_storage() {
    let values = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX - 2, u64::MAX - 1, u64::MAX]),
        vec![1, 3],
    )
    .expect("typed target");
    let mut st = StructValue::new();
    st.fields
        .insert("values".to_string(), Value::Tensor(values));
    let mask = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).expect("logical mask");
    let index =
        CellArray::new_with_shape(vec![Value::LogicalArray(mask)], vec![1, 1]).expect("index cell");

    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(index)],
    )
    .expect("logical index");
    let Value::Tensor(tensor) = result else {
        panic!("expected tensor result");
    };
    assert_eq!(tensor.shape, vec![1, 2]);
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX - 2, u64::MAX]))
    );

    let values = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX - 2, u64::MAX - 1, u64::MAX]),
        vec![1, 3],
    )
    .expect("typed target");
    let mut st = StructValue::new();
    st.fields
        .insert("values".to_string(), Value::Tensor(values));
    let index = CellArray::new_with_shape(vec![Value::Bool(true)], vec![1, 1])
        .expect("scalar logical index");
    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(index)],
    )
    .expect("short logical index");
    assert_eq!(result, Value::Int(IntValue::U64(u64::MAX - 2)));

    let values = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX - 2, u64::MAX - 1, u64::MAX]),
        vec![1, 3],
    )
    .expect("typed target");
    let mut st = StructValue::new();
    st.fields
        .insert("values".to_string(), Value::Tensor(values));
    let index = CellArray::new_with_shape(vec![Value::Bool(false)], vec![1, 1])
        .expect("scalar logical index");
    let result = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(index)],
    )
    .expect("false logical index");
    let Value::Tensor(tensor) = result else {
        panic!("expected empty typed tensor");
    };
    assert_eq!(tensor.shape, vec![1, 0]);
    assert_eq!(tensor.integer_storage(), Some(&IntegerStorage::U64(vec![])));
}
