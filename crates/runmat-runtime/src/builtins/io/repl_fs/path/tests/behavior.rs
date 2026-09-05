use std::convert::TryFrom;

use runmat_value::{CharArray, Value};

use super::support::{call, PathGuard};
use crate::builtins::common::path_search::search_directories;
use crate::builtins::common::path_state::PATH_LIST_SEPARATOR;
use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

fn locked() -> (std::sync::MutexGuard<'static, ()>, PathGuard) {
    let lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|value| value.into_inner());
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
