use std::convert::TryFrom;
use std::env;
use std::path::PathBuf;

use runmat_value::{CharArray, Value};
use tempfile::tempdir;

use super::super::REPL_FS_TEST_LOCK;
use super::*;

fn call(args: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(pwd_builtin(args))
}

struct DirGuard {
    original: PathBuf,
}

impl DirGuard {
    fn new() -> Self {
        Self {
            original: env::current_dir().expect("current directory"),
        }
    }
}

impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = env::set_current_dir(&self.original);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn returns_current_directory_as_character_row() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _guard = DirGuard::new();

    let expected = env::current_dir().expect("current directory");
    let value = call(Vec::new()).expect("pwd");
    let actual = String::try_from(&value).expect("character conversion");
    assert_eq!(actual, expected.to_string_lossy());
    match value {
        Value::CharArray(CharArray { rows, cols, .. }) => {
            assert_eq!(rows, 1);
            assert!(cols >= 1);
        }
        other => panic!("expected character row, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn observes_directory_changes() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _guard = DirGuard::new();
    let temporary = tempdir().expect("temporary directory");
    env::set_current_dir(temporary.path()).expect("change directory");

    let actual = PathBuf::from(String::try_from(&call(Vec::new()).expect("pwd")).expect("path"));
    let expected =
        std::fs::canonicalize(temporary.path()).unwrap_or_else(|_| temporary.path().to_path_buf());
    let actual = std::fs::canonicalize(&actual).unwrap_or(actual);
    assert_eq!(actual, expected);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rejects_arguments_with_stable_error() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _guard = DirGuard::new();

    let error = call(vec![Value::Num(1.0)]).expect_err("pwd input must fail");
    assert_eq!(error.message(), "pwd: too many input arguments");
    assert_eq!(error.identifier(), Some("RunMat:pwd:TooManyInputs"));
}
