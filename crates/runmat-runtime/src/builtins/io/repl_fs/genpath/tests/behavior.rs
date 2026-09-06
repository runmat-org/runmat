use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use runmat_value::{CharArray, StringArray, Value};
use tempfile::tempdir;

use super::support::{call, canonical, segments, text};
use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

struct CurrentDirectoryGuard(PathBuf);

impl CurrentDirectoryGuard {
    fn change(path: &Path) -> Self {
        let previous = env::current_dir().expect("current directory");
        env::set_current_dir(path).expect("change current directory");
        Self(previous)
    }
}

impl Drop for CurrentDirectoryGuard {
    fn drop(&mut self) {
        let _ = env::set_current_dir(&self.0);
    }
}

#[test]
fn returns_a_character_row_for_a_folder() {
    let directory = tempdir().expect("directory");
    let result = call(vec![Value::String(
        directory.path().to_string_lossy().into(),
    )])
    .expect("genpath");
    assert!(matches!(
        result,
        Value::CharArray(CharArray { rows: 1, .. })
    ));
}

#[test]
fn uses_current_directory_without_arguments() {
    let _lock = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|value| value.into_inner());
    let directory = tempdir().expect("directory");
    fs::create_dir(directory.path().join("alpha")).expect("alpha");
    fs::create_dir(directory.path().join("beta")).expect("beta");
    fs::create_dir(directory.path().join("alpha").join("nested")).expect("nested");
    let _guard = CurrentDirectoryGuard::change(directory.path());

    assert_eq!(
        segments(call(Vec::new()).expect("genpath")),
        vec![
            canonical(directory.path()),
            canonical(&directory.path().join("alpha")),
            canonical(&directory.path().join("alpha").join("nested")),
            canonical(&directory.path().join("beta")),
        ]
    );
}

#[test]
fn accepts_character_and_string_scalar_roots() {
    let directory = tempdir().expect("directory");
    let path = directory.path().to_string_lossy();
    let character = call(vec![Value::CharArray(CharArray::new_row(&path))]).expect("character");
    let string = call(vec![Value::StringArray(
        StringArray::new(vec![path.into_owned()], vec![1]).expect("string scalar"),
    )])
    .expect("string");
    assert_eq!(text(character), canonical(directory.path()));
    assert_eq!(text(string), canonical(directory.path()));
}

#[test]
fn deduplicates_symbolic_link_targets() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        let directory = tempdir().expect("directory");
        let target = directory.path().join("alpha");
        fs::create_dir(&target).expect("target");
        symlink(&target, directory.path().join("alias")).expect("symlink");
        let result = segments(
            call(vec![Value::String(
                directory.path().to_string_lossy().into(),
            )])
            .expect("genpath"),
        );
        assert_eq!(
            result
                .iter()
                .filter(|path| **path == canonical(&target))
                .count(),
            1
        );
    }
}
