use super::execute;
use runmat_builtins::{
    TEMPNAME_ERROR_FOLDER_EMPTY, TEMPNAME_ERROR_FOLDER_TYPE, TEMPNAME_ERROR_TOO_MANY_INPUTS,
};
use runmat_value::{CharArray, StringArray, Value};
use std::convert::TryFrom;
use std::path::{Path, PathBuf};

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(execute::run(args))
}

fn text(value: &Value) -> String {
    String::try_from(value).expect("character row")
}

#[test]
fn default_names_are_distinct_unused_character_rows_below_tempdir() {
    let first = run(Vec::new()).expect("first name");
    let second = run(Vec::new()).expect("second name");
    assert!(matches!(first, Value::CharArray(CharArray { rows: 1, .. })));
    assert_ne!(first, second);
    for value in [&first, &second] {
        let path = PathBuf::from(text(value));
        assert_eq!(
            path.parent(),
            Some(crate::builtins::common::env::temp_dir().as_path())
        );
        assert!(runmat_filesystem::metadata(&path).is_err());
    }
}

#[test]
fn accepts_character_string_and_string_array_scalar_folders() {
    let folder = "relative_tempname";
    let values = [
        Value::CharArray(CharArray::new_row(folder)),
        Value::String(folder.to_owned()),
        Value::StringArray(StringArray::new(vec![folder.to_owned()], vec![1, 1]).expect("array")),
    ];
    for value in values {
        let path = PathBuf::from(text(&run(vec![value]).expect("tempname")));
        assert!(path.is_relative());
        assert_eq!(path.parent(), Some(Path::new(folder)));
    }
}

#[test]
fn rejects_invalid_inputs_before_filesystem_access() {
    let cases = [
        (vec![Value::Num(1.0)], TEMPNAME_ERROR_FOLDER_TYPE.message),
        (
            vec![Value::CharArray(CharArray::new_row(""))],
            TEMPNAME_ERROR_FOLDER_EMPTY.message,
        ),
        (
            vec![Value::Num(1.0), Value::Num(2.0)],
            TEMPNAME_ERROR_TOO_MANY_INPUTS.message,
        ),
    ];
    for (arguments, message) in cases {
        assert_eq!(run(arguments).expect_err("error").message(), message);
    }
}
