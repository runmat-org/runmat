use std::convert::TryFrom;
use std::env;

use runmat_value::{CharArray, Value};
use tempfile::tempdir;

use super::support::{call, canonical, DirGuard};
use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

#[test]
fn query_returns_current_directory_as_character_row() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|value| value.into_inner());
    let _guard = DirGuard::new();
    let expected = env::current_dir().expect("current directory");
    let value = call(Vec::new()).expect("cd query");
    assert_eq!(
        String::try_from(&value).expect("path"),
        expected.to_string_lossy()
    );
    assert!(matches!(value, Value::CharArray(CharArray { rows: 1, .. })));
}

#[test]
fn change_returns_previous_directory() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|value| value.into_inner());
    let _guard = DirGuard::new();
    let original = env::current_dir().expect("current directory");
    let target = tempdir().expect("temporary directory");
    let previous = call(vec![Value::from(
        target.path().to_string_lossy().into_owned(),
    )])
    .expect("cd");
    let previous = std::path::PathBuf::from(String::try_from(&previous).expect("previous"));
    assert_eq!(canonical(&previous), canonical(&original));
    assert_eq!(
        canonical(&env::current_dir().expect("changed directory")),
        canonical(target.path())
    );
}

#[test]
fn relative_character_path_is_resolved_from_current_directory() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|value| value.into_inner());
    let _guard = DirGuard::new();
    let root = tempdir().expect("temporary root");
    let child = root.path().join("child");
    std::fs::create_dir(&child).expect("child directory");
    call(vec![Value::from(
        root.path().to_string_lossy().into_owned(),
    )])
    .expect("cd root");
    call(vec![Value::CharArray(CharArray::new_row("child"))]).expect("cd child");
    assert_eq!(
        canonical(&env::current_dir().expect("current directory")),
        canonical(&child)
    );
}

#[test]
fn scalar_string_array_is_accepted() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|value| value.into_inner());
    let _guard = DirGuard::new();
    let current = env::current_dir().expect("current directory");
    let input =
        runmat_value::StringArray::new(vec![current.to_string_lossy().into_owned()], vec![1])
            .expect("scalar string");
    let previous = call(vec![Value::StringArray(input)]).expect("cd");
    assert_eq!(
        String::try_from(&previous).expect("previous"),
        current.to_string_lossy()
    );
}
