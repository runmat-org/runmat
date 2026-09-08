use super::*;

#[test]
fn row_cell_becomes_scalar_structure() {
    let cells = CellArray::new(vec![1.0.into(), Value::from("Ada")], 1, 2).unwrap();
    let names = CellArray::new(vec![Value::from("id"), Value::from("name")], 1, 2).unwrap();
    let Value::Struct(output) = call(cells, Value::Cell(names), Some(Value::Num(2.0))).unwrap()
    else {
        panic!("expected structure")
    };
    assert_eq!(field(&output, "id"), &Value::Num(1.0));
    assert_eq!(field(&output, "name"), &Value::from("Ada"));
}

#[test]
fn crossed_matrix_fields_follow_column_major_coordinates() {
    let cells =
        CellArray::new(vec![1.0.into(), 10.0.into(), 2.0.into(), 20.0.into()], 2, 2).unwrap();
    let names = CellArray::new(vec![Value::from("x"), Value::from("y")], 2, 1).unwrap();
    let Value::Cell(output) = call(cells, Value::Cell(names), Some(Value::Num(1.0))).unwrap()
    else {
        panic!("expected structure-array container")
    };
    let [Value::Struct(first), Value::Struct(second)] = output.data.as_slice() else {
        panic!("expected two structures")
    };
    assert_eq!(
        (field(first, "x"), field(first, "y")),
        (&Value::Num(1.0), &Value::Num(2.0))
    );
    assert_eq!(
        (field(second, "x"), field(second, "y")),
        (&Value::Num(10.0), &Value::Num(20.0))
    );
    assert_eq!(output.shape, [1, 2]);
}

#[test]
fn higher_dimensions_preserve_remaining_shape() {
    let values = (1..=8).map(|value| Value::Num(value as f64)).collect();
    let cells = CellArray::new_with_shape(values, vec![2, 2, 2]).unwrap();
    let names = CellArray::new(vec![Value::from("a"), Value::from("b")], 2, 1).unwrap();
    let Value::Cell(output) = call(cells, Value::Cell(names), Some(Value::Num(3.0))).unwrap()
    else {
        panic!("expected structure-array container")
    };
    assert_eq!(output.shape, [2, 2, 1]);
    assert_eq!(output.data.len(), 4);
}
