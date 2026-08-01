use runmat_package_cache_native::concurrency::ProcessLock;
use runmat_package_cache_native::filesystem::{atomic_write, make_tree_readonly, CacheLayout};

#[test]
fn layout_atomic_write_and_readonly_tree_are_bounded() {
    let directory = tempfile::tempdir().unwrap();
    let layout = CacheLayout::new(directory.path().join("cache"));
    layout.create().unwrap();
    let file = layout.staging.join("tree").join("file.m");
    atomic_write(&file, b"first").unwrap();
    assert!(atomic_write(&file, b"second").is_err());
    assert_eq!(std::fs::read(&file).unwrap(), b"first");
    make_tree_readonly(file.parent().unwrap()).unwrap();
    assert!(std::fs::metadata(&file).unwrap().permissions().readonly());
}

#[test]
fn process_lock_is_reacquirable_after_release() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("locks").join("materialize.lock");
    {
        let _lock = ProcessLock::acquire(&path).unwrap();
    }
    let _lock = ProcessLock::acquire(&path).unwrap();
}
