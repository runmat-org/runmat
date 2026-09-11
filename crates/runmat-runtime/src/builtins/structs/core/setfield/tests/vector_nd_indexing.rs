use super::*;

#[test]
fn setfield_vector_and_logical_assignment_preserve_typed_storage() {
    let mut root = StructValue::new();
    root.fields.insert(
        "values".into(),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 3, 4]), vec![1, 4]).expect("target"),
        ),
    );
    let indices = Tensor::new_integer(IntegerStorage::U8(vec![2, 4]), vec![1, 2]).expect("indices");
    let selector =
        CellArray::new_with_shape(vec![Value::Tensor(indices)], vec![1, 1]).expect("selector");
    let rhs =
        Tensor::new_integer(IntegerStorage::U64(vec![20, 40]), vec![1, 2]).expect("replacement");
    let updated = run_setfield(
        Value::Struct(root),
        vec![
            Value::from("values"),
            Value::Cell(selector),
            Value::Tensor(rhs),
        ],
    )
    .expect("vector assignment");

    let mask = LogicalArray::new(vec![1, 0, 1, 0], vec![1, 4]).expect("mask");
    let selector =
        CellArray::new_with_shape(vec![Value::LogicalArray(mask)], vec![1, 1]).expect("selector");
    let rhs =
        Tensor::new_integer(IntegerStorage::U64(vec![10, 30]), vec![1, 2]).expect("replacement");
    let updated = run_setfield(
        updated,
        vec![
            Value::from("values"),
            Value::Cell(selector),
            Value::Tensor(rhs),
        ],
    )
    .expect("logical assignment");

    let Value::Struct(root) = updated else {
        panic!("expected structure result");
    };
    let Value::Tensor(values) = root.fields.get("values").expect("values field") else {
        panic!("expected numeric field");
    };
    assert_eq!(
        values.integer_storage(),
        Some(&IntegerStorage::U64(vec![10, 20, 30, 40]))
    );
}

#[test]
fn setfield_nd_assignment_uses_column_major_coordinates() {
    let mut root = StructValue::new();
    root.fields.insert(
        "values".into(),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16((1..=8).collect()), vec![2, 2, 2])
                .expect("target"),
        ),
    );
    let selector = CellArray::new_with_shape(
        vec![
            Value::Int(IntValue::U8(2)),
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U8(2)),
        ],
        vec![1, 3],
    )
    .expect("selector");
    let updated = run_setfield(
        Value::Struct(root),
        vec![
            Value::from("values"),
            Value::Cell(selector),
            Value::Int(IntValue::U16(99)),
        ],
    )
    .expect("N-D assignment");
    let Value::Struct(root) = updated else {
        panic!("expected structure result");
    };
    let Value::Tensor(values) = root.fields.get("values").expect("values field") else {
        panic!("expected numeric field");
    };
    assert_eq!(
        values.integer_storage(),
        Some(&IntegerStorage::U16(vec![1, 2, 3, 4, 5, 99, 7, 8]))
    );
}
