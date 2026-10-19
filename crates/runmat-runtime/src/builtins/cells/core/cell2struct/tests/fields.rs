use super::*;
use runmat_value::{CharArray, StringArray};

#[test]
fn accepts_character_string_and_column_major_cell_names() {
    let values = || CellArray::new(vec![1.0.into(), 2.0.into()], 2, 1).unwrap();
    let character = Value::CharArray(CharArray::new(vec!['a', ' ', 'b', ' '], 2, 2).unwrap());
    let strings =
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap());
    let cells = Value::Cell(
        CellArray::from_column_major(vec![Value::from("a"), Value::from("b")], vec![2, 1]).unwrap(),
    );
    for names in [character, strings, cells] {
        let Value::Struct(output) = call(values(), names, None).unwrap() else {
            panic!("expected structure")
        };
        assert_eq!(field(&output, "a"), &Value::Num(1.0));
        assert_eq!(field(&output, "b"), &Value::Num(2.0));
    }
}

#[test]
fn rejects_empty_and_nontext_names() {
    let values = || CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
    for names in [
        Value::from(""),
        Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
    ] {
        assert_eq!(
            call(values(), names, None).unwrap_err().identifier(),
            Some("RunMat:cell2struct:InvalidInput")
        );
    }
}
