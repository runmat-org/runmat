use std::fs;

use runmat_value::{CharArray, StringArray, Tensor, Value};
use tempfile::tempdir;

use super::support::call;

#[test]
fn rejects_too_many_inputs_before_decoding() {
    let error = call(vec![
        Value::Bool(true),
        Value::Bool(true),
        Value::Bool(true),
    ])
    .expect_err("too many inputs");
    assert_eq!(
        error.identifier(),
        runmat_builtins::GENPATH_ERROR_TOO_MANY_INPUTS.identifier
    );
}

#[test]
fn rejects_invalid_scalar_text_shapes() {
    let string_array =
        StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).expect("string array");
    let character = CharArray::new(vec!['a', 'b'], 2, 1).expect("character column");
    for value in [
        Value::StringArray(string_array),
        Value::CharArray(character),
    ] {
        let error = call(vec![value]).expect_err("invalid root");
        assert_eq!(
            error.identifier(),
            runmat_builtins::GENPATH_ERROR_FOLDER_TYPE.identifier
        );
    }
}

#[test]
fn rejects_numeric_columns_and_higher_dimensions() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    for shape in [vec![2, 1], vec![1, 1, 2]] {
        let tensor = Tensor::new(vec![65.0; shape.iter().product()], shape).expect("tensor");
        let error = call(vec![Value::Tensor(tensor)]).expect_err("invalid root");
        assert_eq!(
            error.identifier(),
            runmat_builtins::GENPATH_ERROR_FOLDER_TYPE.identifier
        );
    }
}

#[test]
fn rejects_multiple_outputs_before_root_resolution() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = call(vec![Value::from("missing")]).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:genpath:TooManyOutputs"));
}

#[test]
fn accepts_trailing_singleton_text_and_numeric_roots() {
    let directory = tempdir().unwrap();
    let text = directory.path().to_string_lossy().into_owned();
    let chars: Vec<char> = text.chars().collect();
    let shape = vec![1, chars.len(), 1];
    let character = CharArray::new_with_shape(chars.clone(), shape.clone()).unwrap();
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let numeric = Tensor::new(
        chars.into_iter().map(|c| u32::from(c) as f64).collect(),
        shape,
    )
    .unwrap();
    for value in [Value::CharArray(character), Value::Tensor(numeric)] {
        assert_eq!(
            super::support::text(call(vec![value]).unwrap()),
            super::support::canonical(directory.path())
        );
    }
}

#[test]
fn distinguishes_missing_roots_from_files() {
    let directory = tempdir().expect("directory");
    let missing = directory.path().join("missing");
    let missing_error =
        call(vec![Value::String(missing.to_string_lossy().into())]).expect_err("missing root");
    assert_eq!(
        missing_error.identifier(),
        runmat_builtins::GENPATH_ERROR_FOLDER_NOT_FOUND.identifier
    );

    let file = directory.path().join("file.txt");
    fs::write(&file, b"contents").expect("file");
    let file_error =
        call(vec![Value::String(file.to_string_lossy().into())]).expect_err("file root");
    assert_eq!(
        file_error.identifier(),
        runmat_builtins::GENPATH_ERROR_NOT_FOLDER.identifier
    );
}
