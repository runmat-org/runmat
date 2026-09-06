use runmat_value::{CellArray, CharArray, StringArray, Value};

fn run(value: Value) -> crate::BuiltinResult<Value> {
    super::execute::run(vec![value])
}

#[test]
fn handles_dotfiles_trailing_folders_and_multiple_dots() {
    let Value::OutputList(dotfile) =
        run(Value::CharArray(CharArray::new_row("home/.profile"))).unwrap()
    else {
        panic!("outputs")
    };
    assert_eq!(dotfile[1], Value::CharArray(CharArray::new_row("")));
    assert_eq!(dotfile[2], Value::CharArray(CharArray::new_row(".profile")));
    let folder_path = format!(
        "home{}work{}",
        std::path::MAIN_SEPARATOR,
        std::path::MAIN_SEPARATOR
    );
    let Value::OutputList(folder) =
        run(Value::CharArray(CharArray::new_row(&folder_path))).unwrap()
    else {
        panic!("outputs")
    };
    assert_eq!(folder[1], Value::CharArray(CharArray::new_row("")));
    let Value::OutputList(dots) =
        run(Value::CharArray(CharArray::new_row("archive.part.tar"))).unwrap()
    else {
        panic!("outputs")
    };
    assert_eq!(
        dots[1],
        Value::CharArray(CharArray::new_row("archive.part"))
    );
}

#[test]
fn preserves_string_and_cell_shapes() {
    let strings = StringArray::new(vec!["a.m".into(), "b.txt".into()], vec![1, 2]).unwrap();
    let Value::OutputList(outputs) = run(Value::StringArray(strings)).unwrap() else {
        panic!("outputs")
    };
    let Value::StringArray(names) = &outputs[1] else {
        panic!("strings")
    };
    assert_eq!(names.shape, vec![1, 2]);
    let cells = CellArray::new(
        vec![
            Value::CharArray(CharArray::new_row("a.m")),
            Value::CharArray(CharArray::new_row("b.txt")),
        ],
        2,
        1,
    )
    .unwrap();
    let Value::OutputList(outputs) = run(Value::Cell(cells)).unwrap() else {
        panic!("outputs")
    };
    let Value::Cell(names) = &outputs[1] else {
        panic!("cells")
    };
    assert_eq!(names.shape, vec![2, 1]);
}

#[test]
fn rejects_nontext_without_provider_access() {
    let error = run(Value::Num(1.0)).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:fileparts:InvalidFilename"));
}
