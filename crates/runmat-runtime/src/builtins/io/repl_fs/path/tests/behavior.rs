use std::convert::TryFrom;

use runmat_value::{CharArray, Value};

use super::support::{call, PathGuard};
use crate::builtins::common::path_search::search_directories;
use crate::builtins::common::path_state::PATH_LIST_SEPARATOR;

fn locked() -> (std::sync::MutexGuard<'static, ()>, PathGuard) {
    let lock = runmat_filesystem::provider_override_lock();
    (lock, PathGuard::new())
}

#[test]
fn query_returns_a_character_row() {
    let (_lock, _guard) = locked();
    match call(Vec::new()).expect("path query") {
        Value::CharArray(chars) => assert_eq!(chars.rows, 1),
        other => panic!("expected character row, got {other:?}"),
    }
}

#[test]
fn replacement_accepts_text_and_returns_the_previous_path() {
    let (_lock, guard) = locked();
    let returned = call(vec![Value::String("runmat/path/string".into())]).expect("replace");
    assert_eq!(
        String::try_from(&returned).expect("previous path"),
        guard.previous
    );
    assert_eq!(
        String::try_from(&call(Vec::new()).expect("query")).expect("current path"),
        "runmat/path/string"
    );
}

#[test]
fn two_fragments_are_joined_with_the_platform_separator() {
    let (_lock, _guard) = locked();
    call(vec![Value::from("left"), Value::from("right")]).expect("replace");
    let current = String::try_from(&call(Vec::new()).expect("query")).expect("current path");
    assert_eq!(current, format!("left{PATH_LIST_SEPARATOR}right"));
}

#[test]
fn mutation_updates_runtime_search_directories() {
    let (_lock, _guard) = locked();
    let temporary = tempfile::tempdir().expect("temporary directory");
    let path = temporary.path().to_string_lossy().into_owned();
    call(vec![Value::CharArray(CharArray::new_row(&path))]).expect("replace");
    assert!(search_directories("path test")
        .expect("search directories")
        .iter()
        .any(|entry| entry.to_string_lossy() == path));
}
#[test]
fn discarded_output_still_updates_path() {
    let _lock = runmat_filesystem::provider_override_lock();
    let _guard = super::support::PathGuard::new();
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    super::support::call(vec![runmat_value::Value::from("discarded-output-path")]).unwrap();
    assert_eq!(
        crate::builtins::common::path_state::current_path_string(),
        "discarded-output-path"
    );
}
