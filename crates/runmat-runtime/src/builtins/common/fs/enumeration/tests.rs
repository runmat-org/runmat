use super::*;
use runmat_filesystem::{File, MemoryFsProvider};
use std::sync::Arc;
use tempfile::tempdir;
#[path = "permission_tests.rs"]
mod permission_tests;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rooted_provider_entry_is_not_rebased_under_cwd() {
    let provider = MemoryFsProvider::with_current_dir("/working");
    assert_eq!(
        resolve(
            &provider,
            Path::new("/working"),
            Path::new("/project/file.m")
        ),
        Path::new("/project/file.m")
    );
    assert_eq!(
        resolve(
            &provider,
            Path::new("/working"),
            Path::new("relative/file.m")
        ),
        Path::new("/working/relative/file.m")
    );
}

#[test]
fn missing_and_file_prefixes_have_no_native_matches() {
    let _lock = runmat_filesystem::provider_override_lock();
    let folder = tempdir().unwrap();
    std::fs::write(folder.path().join("file"), b"").unwrap();
    for prefix in ["missing", "file"] {
        for suffix in ["*.m", "**/*.m"] {
            assert!(futures::executor::block_on(matching(
                &folder.path().join(prefix).join(suffix)
            ))
            .unwrap()
            .is_empty());
        }
    }
}

#[test]
fn missing_and_file_prefixes_have_no_memory_provider_matches() {
    let _lock = runmat_filesystem::provider_override_lock();
    let filesystem = MemoryFsProvider::with_current_dir("/workspace");
    filesystem
        .write_project_path("/workspace/file", b"")
        .unwrap();
    runmat_filesystem::with_provider_override(Arc::new(filesystem), || {
        for prefix in ["missing", "file"] {
            for suffix in ["*.m", "**/*.m"] {
                assert!(
                    futures::executor::block_on(matching(&Path::new(prefix).join(suffix)))
                        .unwrap()
                        .is_empty()
                );
            }
        }
    });
}

#[test]
fn recursive_component_matches_files_at_every_depth() {
    let _lock = runmat_filesystem::provider_override_lock();
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
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn wildcard_traversal_uses_the_installed_filesystem_provider() {
    let _lock = runmat_filesystem::provider_override_lock();
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
        futures::executor::block_on(runmat_filesystem::set_current_dir_async("/workspace"))
            .expect("select the fixture's directory without changing provider swap semantics");
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
