use super::*;

#[test]
fn setfield_index_selector_reads_typed_integer_storage_exactly() {
    let mut a = StructValue::new();
    a.fields
        .insert("id".to_string(), Value::Int(IntValue::I32(1)));
    let mut b = StructValue::new();
    b.fields
        .insert("id".to_string(), Value::Int(IntValue::I32(2)));
    let array = StructArray::new(vec![a, b], vec![1, 2]).unwrap();
    let index_tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).expect("index tensor");
    let indices = CellArray::new_with_shape(vec![Value::Tensor(index_tensor)], vec![1, 1])
        .expect("index cell");

    let updated = run_setfield(
        Value::StructArray(array),
        vec![
            Value::Cell(indices),
            Value::from("id"),
            Value::Int(IntValue::I32(42)),
        ],
    )
    .expect("setfield");
    match updated {
        Value::StructArray(array) => assert_eq!(
            array.get_linear(1).unwrap().fields.get("id"),
            Some(&Value::Int(IntValue::I32(42)))
        ),
        other => panic!("expected structure array, got {other:?}"),
    }
}

#[test]
fn setfield_scalar_assignment_reads_typed_integer_storage_exactly() {
    let mut root = StructValue::new();
    root.fields.insert(
        "values".to_string(),
        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
    );
    root.fields.insert(
        "mask".to_string(),
        Value::LogicalArray(LogicalArray::new(vec![0, 0], vec![1, 2]).unwrap()),
    );

    let index_tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).expect("index tensor");
    let index = CellArray::new_with_shape(vec![Value::Tensor(index_tensor)], vec![1, 1]).unwrap();
    let rhs = Tensor::new_integer(IntegerStorage::U64(vec![77]), vec![1, 1]).expect("rhs tensor");
    let updated = run_setfield(
        Value::Struct(root),
        vec![
            Value::from("values"),
            Value::Cell(index),
            Value::Tensor(rhs),
        ],
    )
    .expect("numeric setfield");

    let logical_index =
        Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).expect("logical index");
    let logical_index =
        CellArray::new_with_shape(vec![Value::Tensor(logical_index)], vec![1, 1]).unwrap();
    let logical_rhs =
        Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).expect("logical rhs");
    let updated = run_setfield(
        updated,
        vec![
            Value::from("mask"),
            Value::Cell(logical_index),
            Value::Tensor(logical_rhs),
        ],
    )
    .expect("logical setfield");

    match updated {
        Value::Struct(st) => {
            match st.fields.get("values").expect("values") {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.materialize_f64(), vec![10.0, 77.0])
                }
                other => panic!("expected tensor field, got {other:?}"),
            }
            match st.fields.get("mask").expect("mask") {
                Value::LogicalArray(array) => assert_eq!(array.data, vec![0, 1]),
                other => panic!("expected logical field, got {other:?}"),
            }
        }
        other => panic!("expected struct, got {other:?}"),
    }
}

#[test]
fn setfield_preserves_exact_integer_destination_storage() {
    let large = 9_007_199_254_740_993_u64;
    let mut root = StructValue::new();
    root.fields.insert(
        "wide".to_string(),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![large, large + 1]), vec![1, 2]).unwrap(),
        ),
    );
    let index = CellArray::new_with_shape(vec![Value::Int(IntValue::U8(2))], vec![1, 1]).unwrap();
    let updated = run_setfield(
        Value::Struct(root),
        vec![
            Value::from("wide"),
            Value::Cell(index),
            Value::Int(IntValue::U64(u64::MAX)),
        ],
    )
    .expect("exact integer setfield");

    let Value::Struct(root) = updated else {
        panic!("expected struct");
    };
    let Value::Tensor(tensor) = root.fields.get("wide").expect("wide field") else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.numeric_value_at(0), Some(NumericScalar::U64(large)));
    assert_eq!(
        tensor.numeric_value_at(1),
        Some(NumericScalar::U64(u64::MAX))
    );
}
