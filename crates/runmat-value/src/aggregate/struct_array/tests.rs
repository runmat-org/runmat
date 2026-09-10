use super::*;

fn structure(fields: &[(&str, f64)]) -> StructValue {
    let mut value = StructValue::new();
    for (name, number) in fields {
        value.insert(*name, Value::Num(*number));
    }
    value
}

#[test]
fn retains_empty_schema_and_nd_shape() {
    let value = StructArray::empty(vec!["x".into(), "y".into()], vec![0, 2, 3]).unwrap();
    assert_eq!(value.field_names().cloned().collect::<Vec<_>>(), ["x", "y"]);
    assert_eq!(value.shape(), &[0, 2, 3]);
}

#[test]
fn rejects_heterogeneous_field_schema() {
    let error = StructArray::new(
        vec![structure(&[("x", 1.0)]), structure(&[("y", 2.0)])],
        vec![1, 2],
    )
    .unwrap_err();
    assert!(error.contains("ordered field schema"));
}

#[test]
fn selection_normalizes_one_element_to_scalar() {
    let value = StructArray::new(
        vec![structure(&[("x", 1.0)]), structure(&[("x", 2.0)])],
        vec![1, 2],
    )
    .unwrap();
    assert!(matches!(
        value.select_linear(&[1], vec![1, 1]).unwrap(),
        Value::Struct(_)
    ));
}

#[test]
fn singleton_normalization_still_validates_shape() {
    let error = StructArray::normalize(vec!["x".into()], vec![structure(&[("x", 1.0)])], vec![1])
        .unwrap_err();
    assert!(error.contains("at least two dimensions"));
}

#[test]
fn singleton_with_trailing_dimensions_normalizes_to_scalar() {
    assert!(matches!(
        StructArray::normalize(
            vec!["x".into()],
            vec![structure(&[("x", 1.0)])],
            vec![1, 1, 1],
        )
        .unwrap(),
        Value::Struct(_)
    ));
}

#[test]
fn rejects_duplicate_explicit_schema() {
    let error = StructArray::empty(vec!["x".into(), "x".into()], vec![0, 2]).unwrap_err();
    assert!(error.contains("unique"));
}

#[test]
fn field_major_construction_validates_and_preserves_columns() {
    let Value::StructArray(array) = StructArray::normalize_field_major(
        vec!["left".into(), "right".into()],
        vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(10.0),
            Value::Num(20.0),
        ],
        vec![2, 1],
    )
    .unwrap() else {
        panic!("expected structure array")
    };
    assert_eq!(
        array.field_values("left").unwrap(),
        [Value::Num(1.0), Value::Num(2.0)]
    );
    assert_eq!(
        array.field_values("right").unwrap(),
        [Value::Num(10.0), Value::Num(20.0)]
    );
    assert!(StructArray::normalize_field_major(
        vec!["x".into(), "x".into()],
        Vec::new(),
        vec![0, 2]
    )
    .unwrap_err()
    .contains("unique"));
    assert!(StructArray::normalize_field_major(
        vec!["x".into()],
        vec![Value::Num(1.0)],
        vec![1, 2]
    )
    .unwrap_err()
    .contains("requires 2 field values"));
}

#[test]
fn rejects_duplicate_reorder_without_mutating_or_panicking() {
    let mut array = StructArray::new(
        vec![
            structure(&[("a", 1.0), ("b", 2.0)]),
            structure(&[("a", 3.0), ("b", 4.0)]),
        ],
        vec![1, 2],
    )
    .unwrap();
    assert!(array.reorder_fields(&["a".into(), "a".into()]).is_err());
    assert_eq!(array.field_names().cloned().collect::<Vec<_>>(), ["a", "b"]);
}

#[test]
fn owned_value_mapping_preserves_shape_schema_and_order() {
    let array = StructArray::new(
        vec![
            structure(&[("a", 1.0), ("b", 2.0)]),
            structure(&[("a", 3.0), ("b", 4.0)]),
        ],
        vec![2, 1, 1],
    )
    .unwrap();
    let mapped = array
        .try_map_values::<std::convert::Infallible>(|value| match value {
            Value::Num(number) => Ok(Value::Num(number + 10.0)),
            other => Ok(other),
        })
        .unwrap();
    assert_eq!(mapped.shape(), [2, 1, 1]);
    assert_eq!(
        mapped.field_names().cloned().collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(mapped.get_linear(0).unwrap().fields["a"], Value::Num(11.0));
    assert_eq!(mapped.get_linear(1).unwrap().fields["b"], Value::Num(14.0));
}

#[test]
fn permutation_accepts_implicit_trailing_singleton_dimensions() {
    let array = StructArray::new(
        vec![structure(&[("x", 1.0)]), structure(&[("x", 2.0)])],
        vec![1, 2],
    )
    .unwrap();
    let permuted = array.permute(&[1, 0, 2]).unwrap();
    assert_eq!(permuted.shape(), [2, 1, 1]);
    assert_eq!(permuted.get_linear(0).unwrap().fields["x"], Value::Num(1.0));
    assert_eq!(permuted.get_linear(1).unwrap().fields["x"], Value::Num(2.0));
}

#[test]
fn permutation_uses_effective_rank_for_stored_trailing_singletons() {
    let array = StructArray::new(
        vec![structure(&[("x", 1.0)]), structure(&[("x", 2.0)])],
        vec![2, 1, 1],
    )
    .unwrap();
    let permuted = array.permute(&[1, 0]).unwrap();
    assert_eq!(permuted.shape(), [1, 2]);

    let array = StructArray::new(
        vec![structure(&[("x", 1.0)]), structure(&[("x", 2.0)])],
        vec![2, 1, 1],
    )
    .unwrap();
    let padded = array.permute(&[1, 0, 2, 3]).unwrap();
    assert_eq!(padded.shape(), [1, 2, 1, 1]);
}
