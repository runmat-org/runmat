use super::run;
use runmat_value::{CellArray, CharArray, LogicalArray, StringArray, StructValue, Value};

fn structure() -> Value {
    let mut structure = StructValue::new();
    structure.fields.insert("name".into(), Value::from("Ada"));
    structure.fields.insert("score".into(), Value::Num(42.0));
    Value::Struct(structure)
}

#[test]
fn character_row_is_one_name() {
    let name = CharArray::new_row("name");
    assert_eq!(
        run(structure(), Value::CharArray(name)).unwrap(),
        Value::Bool(true)
    );
}

#[test]
fn string_array_preserves_shape() {
    let names = StringArray::new(vec!["name".into(), "missing".into()], vec![2, 1]).unwrap();
    let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
    assert_eq!(
        run(structure(), Value::StringArray(names)).unwrap(),
        Value::LogicalArray(expected)
    );
}

#[test]
fn mixed_text_cell_preserves_shape_and_column_major_result_order() {
    let names = CellArray::new(
        vec![
            Value::from("name"),
            Value::from("missing"),
            Value::CharArray(CharArray::new_row("score")),
            Value::from("absent"),
        ],
        2,
        2,
    )
    .unwrap();
    let expected = LogicalArray::new(vec![1, 1, 0, 0], vec![2, 2]).unwrap();
    assert_eq!(
        run(structure(), Value::Cell(names)).unwrap(),
        Value::LogicalArray(expected)
    );
}

#[test]
fn non_structure_collection_result_is_same_shaped_false() {
    let names = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
    let expected = LogicalArray::new(vec![0, 0], vec![1, 2]).unwrap();
    assert_eq!(
        run(Value::Num(1.0), Value::StringArray(names)).unwrap(),
        Value::LogicalArray(expected)
    );
}

#[test]
fn dimensionless_empty_cell_uses_matlab_empty_shape() {
    let names = CellArray::new_with_shape(Vec::new(), Vec::new()).unwrap();
    let expected = LogicalArray::new(Vec::new(), vec![0, 0]).unwrap();
    assert_eq!(
        run(structure(), Value::Cell(names)).unwrap(),
        Value::LogicalArray(expected)
    );
}
