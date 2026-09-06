use std::convert::TryFrom;

use runmat_value::{CellArray, CharArray, StringArray, Value};
use tempfile::tempdir;

use super::super::super::path_mutation::test_support::{canonical, PathGuard};

#[test]
fn prepends_returns_previous_path_and_deduplicates() {
    let _guard = PathGuard::new();
    let first = tempdir().expect("first");
    let second = tempdir().expect("second");
    let first = canonical(first.path());
    let second = canonical(second.path());
    crate::builtins::common::path_state::set_path_string(&format!(
        "{first}{}{second}",
        crate::builtins::common::path_state::PATH_LIST_SEPARATOR
    ));

    let returned = super::run(vec![Value::String(second.clone())]).expect("addpath");
    assert_eq!(
        String::try_from(&returned).expect("text"),
        format!(
            "{first}{}{second}",
            crate::builtins::common::path_state::PATH_LIST_SEPARATOR
        )
    );
    assert_eq!(
        crate::builtins::common::path_state::current_path_segments(),
        vec![second, first]
    );
}

#[test]
fn appends_text_containers_and_character_rows() {
    let _guard = PathGuard::new();
    let first = tempdir().expect("first");
    let second = tempdir().expect("second");
    let first_text = first.path().to_string_lossy().into_owned();
    let second_text = second.path().to_string_lossy().into_owned();
    let strings = StringArray::new(vec![first_text], vec![1, 1]).expect("strings");
    let cells = CellArray::new(
        vec![Value::CharArray(CharArray::new_row(&second_text))],
        1,
        1,
    )
    .expect("cell");
    crate::builtins::common::path_state::set_path_string("");
    super::run(vec![
        Value::StringArray(strings),
        Value::Cell(cells),
        Value::String("-end".into()),
    ])
    .expect("addpath");
    assert_eq!(
        crate::builtins::common::path_state::current_path_segments(),
        vec![canonical(first.path()), canonical(second.path())]
    );
}
