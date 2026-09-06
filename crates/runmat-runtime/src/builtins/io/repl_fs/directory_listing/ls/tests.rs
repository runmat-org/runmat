use super::*;
use runmat_filesystem::{File, MemoryFsProvider};
use runmat_value::{IntegerStorage, Tensor};
use std::sync::Arc;
use tempfile::tempdir;

use super::super::super::REPL_FS_TEST_LOCK;

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(ls_builtin(args))
}

fn rows(value: Value) -> Vec<String> {
    let Value::CharArray(array) = value else {
        panic!("expected character array");
    };
    (0..array.rows)
        .map(|row| {
            (0..array.cols)
                .map(|column| array.data[row * array.cols + column])
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

#[test]
fn runmat_mode_lists_one_name_per_row() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let folder = tempdir().expect("temporary directory");
    File::create(folder.path().join("alpha.txt")).expect("file");
    std::fs::create_dir(folder.path().join("nested")).expect("folder");
    let listed = rows(
        run(vec![Value::from(
            folder.path().to_string_lossy().into_owned(),
        )])
        .expect("ls"),
    );
    assert!(listed.contains(&"alpha.txt".into()));
    assert!(listed.contains(&format!("nested{}", std::path::MAIN_SEPARATOR)));
}

#[test]
fn wildcard_uses_the_filesystem_enumerator() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let folder = tempdir().expect("temporary directory");
    File::create(folder.path().join("one.m")).expect("m file");
    File::create(folder.path().join("two.txt")).expect("text file");
    let pattern = format!("{}/*.m", folder.path().to_string_lossy());
    let listed = rows(run(vec![Value::from(pattern)]).expect("ls wildcard"));
    assert_eq!(listed.len(), 1);
    assert!(listed[0].ends_with("one.m"));
}

#[test]
fn relative_wildcard_output_is_independent_of_provider_path_style() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let filesystem = MemoryFsProvider::with_current_dir("/workspace");
    filesystem
        .write_project_path("/workspace/one.m", b"")
        .expect("m file");
    filesystem
        .write_project_path("/workspace/two.txt", b"")
        .expect("text file");

    let listed = runmat_filesystem::with_provider_override(Arc::new(filesystem), || {
        rows(run(vec![Value::from("*.m")]).expect("provider wildcard"))
    });
    assert_eq!(listed, vec!["one.m"]);
}

#[cfg(not(windows))]
#[test]
fn strict_mode_uses_a_unix_character_vector() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let folder = tempdir().expect("temporary directory");
    File::create(folder.path().join("a.txt")).expect("first file");
    File::create(folder.path().join("b.txt")).expect("second file");
    let Value::CharArray(listing) = run(vec![Value::from(
        folder.path().to_string_lossy().into_owned(),
    )])
    .expect("ls") else {
        panic!("expected character array");
    };
    assert_eq!(listing.rows, 1);
    assert!(listing.data.iter().collect::<String>().contains("a.txt"));
    assert!(listing.data.iter().collect::<String>().contains("b.txt"));
}

#[test]
fn invalid_numeric_input_rejects_without_gathering() {
    let tensor = Tensor::new_integer(IntegerStorage::U8(vec![65]), vec![1, 1]).expect("tensor");
    let error = run(vec![Value::Tensor(tensor)]).expect_err("invalid input");
    assert_eq!(error.message(), runmat_builtins::LS_ERROR_NAME.message);
}
