use super::*;
use runmat_value::{CellArray, CharArray, IntValue, StringArray};

fn row(text: &str) -> Value {
    Value::CharArray(
        CharArray::new_with_shape(text.chars().collect(), vec![1, text.chars().count(), 1])
            .expect("effective row"),
    )
}

#[test]
fn strict_scalar_text_accepts_rows_and_scalar_strings_without_numeric_coercion() {
    assert_eq!(string_argument(Some(row("code")), "code").unwrap(), "code");
    assert_eq!(
        string_argument(
            Some(Value::StringArray(
                StringArray::new(vec!["name".into()], vec![1, 1]).unwrap()
            )),
            "name"
        )
        .unwrap(),
        "name"
    );
    for invalid in [
        Value::Num(3.0),
        Value::Int(IntValue::I64(3)),
        Value::Bool(true),
        Value::String(String::new()),
        Value::CharArray(CharArray::new(vec!['x', 'y'], 2, 1).unwrap()),
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
    ] {
        assert_eq!(
            string_argument(Some(invalid), "Python code")
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
    }
}

#[test]
fn output_name_cells_and_script_arguments_require_actual_text() {
    let names = Value::Cell(
        CellArray::new(vec![row("answer"), Value::String("other".into())], 1, 2).unwrap(),
    );
    assert_eq!(
        output_names(Some(names)).unwrap(),
        vec!["answer".to_string(), "other".to_string()]
    );
    for invalid in [
        Value::Num(42.0),
        Value::Int(IntValue::I64(1)),
        Value::Bool(false),
    ] {
        let cell = Value::Cell(CellArray::new(vec![invalid.clone()], 1, 1).unwrap());
        assert_eq!(
            output_names(Some(cell)).unwrap_err().identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
        let pair = vec![
            row("Arguments"),
            Value::Cell(CellArray::new(vec![invalid], 1, 1).unwrap()),
        ];
        assert_eq!(
            split_file_inputs(pair).unwrap_err().identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
    }
    let pair = vec![
        row("Arguments"),
        Value::Cell(CellArray::new(vec![row("--flag")], 1, 1).unwrap()),
    ];
    let (arguments, remaining) = split_file_inputs(pair).unwrap();
    assert_eq!(arguments, vec!["--flag".to_string()]);
    assert!(remaining.is_empty());
}
