use super::*;
use futures::executor::block_on;

#[test]
fn path_guards_use_configured_vfs_and_preserve_false_on_metadata_error() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let root = tempfile::tempdir().unwrap();
    let provider = runmat_filesystem::SandboxFsProvider::new(root.path().to_owned()).unwrap();
    let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(provider));
    block_on(runmat_filesystem::create_dir_all_async("validator-domain")).unwrap();
    block_on(runmat_filesystem::write_async(
        "validator-domain/data.txt",
        b"test",
    ))
    .unwrap();
    let file = Value::String("/validator-domain/data.txt".into());
    let folder = Value::String("/validator-domain".into());
    assert!(block_on(value_is_file_async(&file)).unwrap());
    assert!(block_on(value_is_folder_async(&folder)).unwrap());
    assert!(!block_on(value_is_folder_async(&file)).unwrap());
    assert!(!block_on(value_is_file_async(&folder)).unwrap());
    assert!(!block_on(value_is_file_async(&Value::String("/missing.txt".into()))).unwrap());
    assert!(block_on(must_be_file::evaluate(vec![file])).is_ok());
    assert!(block_on(must_be_folder::evaluate(vec![folder])).is_ok());
    let invalid = Value::Cell(
        CellArray::new(
            vec![Value::String("/validator-domain".into()), Value::Num(1.0)],
            1,
            2,
        )
        .unwrap(),
    );
    assert_eq!(
        block_on(value_is_folder_async(&invalid))
            .unwrap_err()
            .identifier(),
        Some("RunMat:argumentValidation:InvalidArgument")
    );
}
