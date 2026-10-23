use std::convert::TryFrom;

use runmat_value::{CellArray, CharArray, StringArray, Value};
use tempfile::tempdir;

use super::super::super::path_mutation::test_support::{canonical, PathGuard};

#[test]
fn removes_entries_and_returns_the_previous_path() {
    let _guard = PathGuard::new();
    let target = tempdir().expect("target");
    let keep = tempdir().expect("keep");
    let target = canonical(target.path());
    let keep = canonical(keep.path());
    let original = format!(
        "{target}{}{keep}",
        crate::builtins::common::path_state::PATH_LIST_SEPARATOR
    );
    crate::builtins::common::path_state::set_path_string(&original)
        .expect("test search path update");
    let returned = super::run(vec![Value::String(target.into())]).expect("rmpath");
    assert_eq!(String::try_from(&returned).expect("text"), original);
    assert_eq!(
        crate::builtins::common::path_state::current_path_segments(),
        vec![keep]
    );
}

#[test]
fn accepts_path_lists_string_arrays_character_rows_and_cells() {
    let _guard = PathGuard::new();
    let first = tempdir().expect("first");
    let second = tempdir().expect("second");
    let third = tempdir().expect("third");
    let paths = [
        canonical(first.path()),
        canonical(second.path()),
        canonical(third.path()),
    ];
    crate::builtins::common::path_state::set_path_string(&super::super::super::path_list::join(
        &paths,
    ))
    .expect("test search path update");
    let list = format!(
        "{}{}{}",
        paths[0],
        crate::builtins::common::path_state::PATH_LIST_SEPARATOR,
        paths[1]
    );
    let array = StringArray::new(vec![list], vec![1, 1]).expect("strings");
    let cell =
        CellArray::new(vec![Value::CharArray(CharArray::new_row(&paths[2]))], 1, 1).expect("cell");
    super::run(vec![Value::StringArray(array), Value::Cell(cell)]).expect("rmpath");
    assert!(crate::builtins::common::path_state::current_path_segments().is_empty());
}
#[test]
fn discarded_output_still_removes_path() {
    let _guard = super::super::super::path_mutation::test_support::PathGuard::new();
    let directory = tempfile::tempdir().unwrap();
    let path = super::super::super::path_mutation::test_support::canonical(directory.path());
    crate::builtins::common::path_state::set_path_string(&path).expect("test search path update");
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    super::run(vec![runmat_value::Value::String(path.into())]).unwrap();
    assert!(crate::builtins::common::path_state::current_path_string().is_empty());
}
