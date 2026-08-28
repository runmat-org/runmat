use runmat_package_cache_native::filesystem::CacheLayout;
use runmat_package_cache_native::materialize::remove_interrupted_staging;

#[test]
fn recovery_removes_only_owned_partial_directories() {
    let directory = tempfile::tempdir().unwrap();
    let layout = CacheLayout::new(directory.path().join("cache"));
    layout.create().unwrap();
    let partial = layout.staging.join("tree-123-1.partial");
    let unrelated = layout.staging.join("user-data");
    std::fs::create_dir(&partial).unwrap();
    std::fs::write(partial.join("incomplete"), b"partial").unwrap();
    std::fs::create_dir(&unrelated).unwrap();

    assert_eq!(
        remove_interrupted_staging(&layout).unwrap(),
        vec!["tree-123-1.partial"]
    );
    assert!(!partial.exists());
    assert!(unrelated.exists());
}
