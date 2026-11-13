use super::*;
use futures::executor::block_on;

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn path_guards_use_configured_vfs_and_preserve_false_on_metadata_error() {
    let root = tempfile::tempdir().unwrap();
    let provider = runmat_filesystem::SandboxFsProvider::new(root.path().to_owned()).unwrap();
    assert_configured_path_guards(std::sync::Arc::new(provider));
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn path_guards_use_portable_memory_vfs_and_preserve_false_on_metadata_error() {
    assert_configured_path_guards(std::sync::Arc::new(
        runmat_filesystem::MemoryFsProvider::new(),
    ));
}

fn assert_configured_path_guards(provider: std::sync::Arc<dyn runmat_filesystem::FsProvider>) {
    let _lock = runmat_filesystem::provider_override_lock();
    let _provider = runmat_filesystem::replace_provider(provider);
    block_on(runmat_filesystem::create_dir_all_async("/validator-domain")).unwrap();
    block_on(runmat_filesystem::write_async(
        "/validator-domain/data.txt",
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
    assert!(block_on(must_be_file::evaluate(
        crate::builtins::common::validation::tests::retained_arguments(vec![file])
    ))
    .is_ok());
    assert!(block_on(must_be_folder::evaluate(
        crate::builtins::common::validation::tests::retained_arguments(vec![folder])
    ))
    .is_ok());
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
