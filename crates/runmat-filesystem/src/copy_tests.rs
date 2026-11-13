use super::*;

#[test]
fn copy_file_on_retains_exact_provider_after_global_swap() {
    let _lock = provider_override_lock();
    let selected = Arc::new(MemoryFsProvider::with_current_dir("/selected"));
    selected
        .write_project_path("/selected/source.bin", b"selected payload")
        .expect("source");
    let replacement = Arc::new(MemoryFsProvider::with_current_dir("/other"));
    replacement
        .write_project_path("/selected/source.bin", b"wrong payload")
        .expect("same lexical source on another provider");
    let _active = replace_provider(replacement.clone());
    let copied = copy_file_on(
        selected.as_ref(),
        Path::new("/selected/source.bin"),
        Path::new("/selected/target.bin"),
    )
    .expect("copy through retained provider");
    assert_eq!(copied, b"selected payload".len() as u64);
    assert_eq!(
        selected.read_project_path("/selected/target.bin").unwrap(),
        b"selected payload"
    );
    assert!(replacement
        .read_project_path("/selected/target.bin")
        .is_err());
    assert_eq!(
        copy_file_on(
            selected.as_ref(),
            Path::new("relative"),
            Path::new("/selected/target.bin")
        )
        .unwrap_err()
        .kind(),
        ErrorKind::InvalidInput
    );
}
