use super::*;
use runmat_value::{CellArray, CharArray, StringArray, Tensor};

#[test]
fn default_uses_ascii_order() {
    let input = structure(&[
        ("b", Value::Num(1.0)),
        ("B", Value::Num(2.0)),
        ("a", Value::Num(3.0)),
        ("A", Value::Num(4.0)),
    ]);
    let Value::Struct(output) = call(Value::Struct(input), Vec::new()).unwrap() else {
        panic!("expected structure")
    };
    assert_eq!(field_order(&output), ["A", "B", "a", "b"]);
}

#[test]
fn accepts_reference_name_and_permutation_forms() {
    let source = || structure(&[("a", 1.0.into()), ("b", 2.0.into()), ("c", 3.0.into())]);
    let reference = structure(&[("c", 0.0.into()), ("a", 0.0.into()), ("b", 0.0.into())]);
    let names = CellArray::new(
        vec![Value::from("c"), Value::from("a"), Value::from("b")],
        1,
        3,
    )
    .unwrap();
    let strings = StringArray::new(vec!["c".into(), "a".into(), "b".into()], vec![1, 3]).unwrap();
    let positions = Tensor::new(vec![3.0, 1.0, 2.0], vec![1, 3]).unwrap();
    let orders = [
        Value::Struct(reference),
        Value::Cell(names),
        Value::StringArray(strings),
        Value::Tensor(positions),
    ];
    for order in orders {
        let Value::Struct(output) = call(Value::Struct(source()), vec![order]).unwrap() else {
            panic!("expected structure")
        };
        assert_eq!(field_order(&output), ["c", "a", "b"]);
    }
}

#[test]
fn character_matrix_uses_trimmed_rows() {
    let source = structure(&[
        ("cat", 1.0.into()),
        ("ant", 2.0.into()),
        ("bat", 3.0.into()),
    ]);
    let names = CharArray::new(vec!['b', 'a', 't', 'c', 'a', 't', 'a', 'n', 't'], 3, 3).unwrap();
    let Value::Struct(output) = call(Value::Struct(source), vec![Value::CharArray(names)]).unwrap()
    else {
        panic!("expected structure")
    };
    assert_eq!(field_order(&output), ["bat", "cat", "ant"]);
}

#[test]
fn cell_name_order_follows_column_major_linear_order() {
    let source = structure(&[
        ("a", 1.0.into()),
        ("b", 2.0.into()),
        ("c", 3.0.into()),
        ("d", 4.0.into()),
    ]);
    let names = CellArray::from_column_major(
        vec![
            Value::from("d"),
            Value::from("c"),
            Value::from("b"),
            Value::from("a"),
        ],
        vec![2, 2],
    )
    .unwrap();
    let Value::Struct(output) = call(Value::Struct(source), vec![Value::Cell(names)]).unwrap()
    else {
        panic!("expected structure")
    };
    assert_eq!(field_order(&output), ["d", "c", "b", "a"]);
}
