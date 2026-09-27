use super::*;

#[test]
fn final_colon_folds_trailing_dimensions_including_empty_extents() {
    for (shape, count) in [(vec![2, 3, 4], 24), (vec![2, 3, 0], 0)] {
        let target =
            Value::Cell(CellArray::new_with_shape(vec![Value::Num(0.0); count], shape).unwrap());
        let path = substruct("{}", cell(vec![Value::String(":".into()); 2]));
        assert_eq!(
            call(target, path, Value::String("Expression".into())),
            Value::Num(count as f64)
        );
    }
}

#[test]
fn character_colon_uses_full_nd_shape() {
    let target = Value::CharArray(
        runmat_value::CharArray::new_with_shape(vec!['a'; 6], vec![1, 2, 3]).unwrap(),
    );
    assert_eq!(
        single_subscript_count(&target, &Value::String(":".into()), 0, 1).unwrap(),
        6
    );
    assert_eq!(
        single_subscript_count(&target, &Value::String(":".into()), 1, 2).unwrap(),
        6
    );
}
