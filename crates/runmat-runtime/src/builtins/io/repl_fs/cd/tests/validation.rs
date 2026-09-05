use runmat_value::{CharArray, StringArray, Value};

use super::support::{call, DirGuard};
use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

fn locked() -> (std::sync::MutexGuard<'static, ()>, DirGuard) {
    let lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|value| value.into_inner());
    let guard = DirGuard::new();
    (lock, guard)
}

#[test]
fn rejects_missing_empty_and_invalid_folders_with_stable_errors() {
    let (_lock, _guard) = locked();
    let parent = tempfile::tempdir().expect("temporary parent");
    let missing_path = parent.path().join("missing");
    let missing =
        call(vec![Value::from(missing_path.to_string_lossy().as_ref())]).expect_err("missing");
    assert_eq!(missing.identifier(), Some("RunMat:cd:ChangeFailed"));
    assert!(missing.message().contains("unable to change directory"));

    let empty = call(vec![Value::from("")]).expect_err("empty");
    assert_eq!(empty.identifier(), Some("RunMat:cd:EmptyFolder"));

    let numeric = call(vec![Value::Num(1.0)]).expect_err("numeric");
    assert_eq!(numeric.identifier(), Some("RunMat:cd:InvalidFolder"));
}

#[test]
fn rejects_nonscalar_string_and_multiline_character_arrays() {
    let (_lock, _guard) = locked();
    let strings = StringArray::new(vec!["a".into(), "b".into()], vec![2]).expect("strings");
    let error = call(vec![Value::StringArray(strings)]).expect_err("array");
    assert_eq!(error.identifier(), Some("RunMat:cd:InvalidFolder"));

    let chars = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).expect("characters");
    let error = call(vec![Value::CharArray(chars)]).expect_err("matrix");
    assert_eq!(error.identifier(), Some("RunMat:cd:InvalidFolder"));
}

#[test]
fn rejects_excess_arguments_before_environment_mutation() {
    let (_lock, guard) = locked();
    let error = call(vec![Value::from(".."), Value::from("..")]).expect_err("arity");
    assert_eq!(error.identifier(), Some("RunMat:cd:TooManyInputs"));
    assert_eq!(
        std::env::current_dir().expect("current directory"),
        guard.original
    );
}
