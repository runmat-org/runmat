use futures::executor::block_on;
use runmat_value::{CellArray, CharArray, StringArray, Value};

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(super::execute::run(args))
}

#[test]
fn joins_scalars_and_preserves_trailing_separator() {
    let result = run(vec![
        Value::CharArray(CharArray::new_row("data")),
        Value::CharArray(CharArray::new_row("raw")),
        Value::CharArray(CharArray::new_row(std::path::MAIN_SEPARATOR_STR)),
    ])
    .unwrap();
    let text = String::try_from(&result).unwrap();
    assert!(text.ends_with(std::path::MAIN_SEPARATOR));
    assert!(text.contains("data"));
}

#[test]
fn broadcasts_scalars_over_string_arrays() {
    let names = StringArray::new(vec!["a.m".into(), "b.m".into()], vec![1, 2]).unwrap();
    let Value::StringArray(paths) =
        run(vec![Value::from("src"), Value::StringArray(names)]).unwrap()
    else {
        panic!("string array")
    };
    assert_eq!(paths.shape, vec![1, 2]);
    assert!(paths.data[0].ends_with("a.m"));
}

#[test]
fn broadcasts_scalars_over_cells_and_rejects_shape_mismatch() {
    let cells = CellArray::new(
        vec![
            Value::CharArray(CharArray::new_row("a")),
            Value::CharArray(CharArray::new_row("b")),
        ],
        2,
        1,
    )
    .unwrap();
    let Value::Cell(paths) = run(vec![
        Value::CharArray(CharArray::new_row("src")),
        Value::Cell(cells),
    ])
    .unwrap() else {
        panic!("cell")
    };
    assert_eq!(paths.shape, vec![2, 1]);
    let left = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
    let right = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
    assert!(run(vec![Value::StringArray(left), Value::StringArray(right)]).is_err());
}
