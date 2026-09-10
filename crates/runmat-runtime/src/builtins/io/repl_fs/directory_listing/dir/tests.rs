use super::*;
use runmat_filesystem::File;
use runmat_value::{IntegerStorage, Tensor};
use std::path::{Path, PathBuf};
use tempfile::tempdir;

use super::super::super::REPL_FS_TEST_LOCK;

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(dir_builtin(args))
}

fn names(value: Value) -> Vec<String> {
    let elements = match value {
        Value::Struct(value) => vec![value],
        Value::StructArray(array) => array.into_elements(),
        other => panic!("expected structure value, got {other:?}"),
    };
    elements
        .into_iter()
        .map(|entry| match entry.fields.get("name") {
            Some(Value::CharArray(name)) if name.rows == 1 => name.data.iter().collect(),
            other => panic!("expected character-row name, got {other:?}"),
        })
        .collect()
}

struct CurrentDirectoryGuard(PathBuf);

impl CurrentDirectoryGuard {
    fn enter(path: &Path) -> Self {
        let original = std::env::current_dir().expect("current directory");
        std::env::set_current_dir(path).expect("change current directory");
        Self(original)
    }
}

impl Drop for CurrentDirectoryGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.0);
    }
}

#[test]
fn lists_current_directory_and_metadata_fields() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let folder = tempdir().expect("temporary directory");
    let _guard = CurrentDirectoryGuard::enter(folder.path());
    File::create("alpha.txt").expect("file");
    std::fs::create_dir("nested").expect("folder");

    let result = run(Vec::new()).expect("dir");
    let listed = names(result);
    assert!(listed.contains(&".".into()));
    assert!(listed.contains(&"..".into()));
    assert!(listed.contains(&"alpha.txt".into()));
    assert!(listed.contains(&"nested".into()));
}

#[test]
fn wildcard_and_folder_pattern_select_entries() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let folder = tempdir().expect("temporary directory");
    File::create(folder.path().join("one.m")).expect("m file");
    File::create(folder.path().join("two.txt")).expect("text file");
    let root = folder.path().to_string_lossy().into_owned();

    let wildcard = run(vec![Value::from(format!("{root}/*.m"))]).expect("wildcard");
    assert_eq!(names(wildcard), vec!["one.m"]);

    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let separated = run(vec![Value::from(root), Value::from("*.txt")]).expect("two inputs");
    assert_eq!(names(separated), vec!["two.txt"]);
}

#[test]
fn strict_mode_rejects_the_two_input_extension() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = run(vec![Value::from("."), Value::from("*.m")]).expect_err("extension");
    assert_eq!(
        error.identifier(),
        runmat_builtins::DIR_FOLDER_PATTERN_EXTENSION.error_identifier
    );
}

#[test]
fn invalid_resident_input_rejects_before_provider_access() {
    let tensor = Tensor::new_integer(IntegerStorage::U8(vec![65]), vec![1, 1]).expect("tensor");
    let error = run(vec![Value::Tensor(tensor)]).expect_err("invalid input");
    assert_eq!(error.message(), runmat_builtins::DIR_ERROR_NAME.message);
}
