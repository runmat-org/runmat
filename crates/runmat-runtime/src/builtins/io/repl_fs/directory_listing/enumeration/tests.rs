use super::*;
use runmat_filesystem::{File, MemoryFsProvider};
use std::sync::Arc;
use tempfile::tempdir;

#[test]
fn recursive_component_matches_files_at_every_depth() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let folder = tempdir().expect("temporary directory");
    std::fs::create_dir_all(folder.path().join("one/two")).expect("nested folders");
    File::create(folder.path().join("root.m")).expect("root file");
    File::create(folder.path().join("one/child.m")).expect("child file");
    File::create(folder.path().join("one/two/leaf.m")).expect("leaf file");
    File::create(folder.path().join("one/two/skip.txt")).expect("other file");
    let pattern = folder.path().join("**/*.m");

    let entries = futures::executor::block_on(matching(&pattern)).expect("matches");
    let mut names = entries
        .iter()
        .map(|entry| filename(&entry.path).to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["child.m", "leaf.m", "root.m"]);
}

#[test]
fn wildcard_traversal_uses_the_installed_filesystem_provider() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let filesystem = MemoryFsProvider::with_current_dir("/workspace");
    filesystem
        .write_project_path("/workspace/source/root.m", b"")
        .expect("root source");
    filesystem
        .write_project_path("/workspace/source/nested/child.m", b"")
        .expect("nested source");
    filesystem
        .write_project_path("/workspace/source/nested/notes.txt", b"")
        .expect("nonmatching file");

    let entries = runmat_filesystem::with_provider_override(Arc::new(filesystem), || {
        futures::executor::block_on(matching(Path::new("source/**/*.m")))
            .expect("provider-backed matches")
    });
    let mut names = entries
        .iter()
        .map(|entry| filename(&entry.path).to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["child.m", "root.m"]);
    assert!(entries.iter().all(|entry| entry.path.is_relative()));
}
