use super::*;

#[test]
fn row_cell_becomes_scalar_structure() {
    let cells = CellArray::new(vec![1.0.into(), Value::from("entry")], 1, 2).unwrap();
    let names = CellArray::new(vec![Value::from("id"), Value::from("name")], 1, 2).unwrap();
    let Value::Struct(output) = call(cells, Value::Cell(names), Some(Value::Num(2.0))).unwrap()
    else {
        panic!("expected structure")
    };
    assert_eq!(field(&output, "id"), &Value::Num(1.0));
    assert_eq!(field(&output, "name"), &Value::from("entry"));
}

#[test]
fn crossed_matrix_fields_follow_column_major_coordinates() {
    let cells =
        CellArray::new(vec![1.0.into(), 10.0.into(), 2.0.into(), 20.0.into()], 2, 2).unwrap();
    let names = CellArray::new(vec![Value::from("x"), Value::from("y")], 2, 1).unwrap();
    let Value::StructArray(output) =
        call(cells, Value::Cell(names), Some(Value::Num(1.0))).unwrap()
    else {
        panic!("expected typed structure array")
    };
    let first = output.get_linear(0).expect("first structure");
    let second = output.get_linear(1).expect("second structure");
    assert_eq!(
        (
            first.fields.get("x").unwrap(),
            first.fields.get("y").unwrap()
        ),
        (&Value::Num(1.0), &Value::Num(2.0))
    );
    assert_eq!(
        (
            second.fields.get("x").unwrap(),
            second.fields.get("y").unwrap()
        ),
        (&Value::Num(10.0), &Value::Num(20.0))
    );
    assert_eq!(output.shape(), [1, 2]);
}

#[test]
fn higher_dimensions_preserve_remaining_shape() {
    let values = (1..=8).map(|value| Value::Num(value as f64)).collect();
    let cells = CellArray::new_with_shape(values, vec![2, 2, 2]).unwrap();
    let names = CellArray::new(vec![Value::from("a"), Value::from("b")], 2, 1).unwrap();
    let Value::StructArray(output) =
        call(cells, Value::Cell(names), Some(Value::Num(3.0))).unwrap()
    else {
        panic!("expected typed structure array")
    };
    assert_eq!(output.shape(), [2, 2, 1]);
    assert_eq!(output.len(), 4);
}
