use super::{run, structure};
use runmat_value::{CellArray, CharArray, StringArray, Value};

#[test]
fn scalar_cell_and_string_array_forms_are_flattened() {
    let target = structure(&[
        ("a", Value::Num(1.0)),
        ("b", Value::Num(2.0)),
        ("c", Value::Num(3.0)),
    ]);
    let names = CellArray::new(
        vec![Value::CharArray(CharArray::new_row("a")), Value::from("b")],
        1,
        2,
    )
    .unwrap();
    let result = run(target, vec![Value::Cell(names)]).unwrap();
    assert!(
        matches!(result, Value::Struct(value) if value.fields.len() == 1 && value.fields.contains_key("c"))
    );

    let target = structure(&[("a", Value::Num(1.0)), ("b", Value::Num(2.0))]);
    let names = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
    assert!(
        matches!(run(target, vec![Value::StringArray(names)]).unwrap(), Value::Struct(value) if value.fields.is_empty())
    );
}

#[test]
fn duplicate_names_are_removed_once() {
    let target = structure(&[("keep", Value::Num(1.0)), ("drop", Value::Num(2.0))]);
    let names = CellArray::new(vec![Value::from("drop"), Value::from("drop")], 1, 2).unwrap();
    let result = run(target, vec![Value::Cell(names)]).unwrap();
    assert!(
        matches!(result, Value::Struct(value) if value.fields.len() == 1 && value.fields.contains_key("keep"))
    );
}

#[test]
fn cell_names_follow_column_major_linear_order() {
    let target = structure(&[("a", Value::Num(1.0)), ("b", Value::Num(2.0))]);
    let names = CellArray::new(
        vec![
            Value::from("a"),
            Value::from("missing-second"),
            Value::from("missing-first"),
            Value::from("b"),
        ],
        2,
        2,
    )
    .unwrap();
    let error = run(target, vec![Value::Cell(names)]).unwrap_err();
    assert!(error.message().contains("missing-first"));
}

#[test]
fn empty_collection_is_a_noop_after_target_validation() {
    let target = structure(&[("value", Value::Num(10.0))]);
    let names = CellArray::new(Vec::new(), 0, 0).unwrap();
    assert_eq!(
        run(target.clone(), vec![Value::Cell(names)]).unwrap(),
        target
    );
}
