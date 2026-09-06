use std::sync::Arc;

use runmat_filesystem::MemoryFsProvider;
use runmat_value::Value;

#[test]
fn copy_and_move_use_the_portable_filesystem_provider() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let filesystem = MemoryFsProvider::with_current_dir("/workspace");
    filesystem
        .write_project_path("/workspace/source.bin", b"typed bytes")
        .expect("seed source");

    runmat_filesystem::with_provider_override(Arc::new(filesystem.clone()), || {
        let copied = futures::executor::block_on(super::copyfile::execute::evaluate(&[
            Value::from("source.bin"),
            Value::from("copy.bin"),
        ]))
        .expect("copy through memory provider");
        assert_eq!(copied.status(), 1.0);

        let moved = futures::executor::block_on(super::movefile::execute::evaluate(&[
            Value::from("copy.bin"),
            Value::from("moved.bin"),
        ]))
        .expect("move through memory provider");
        assert_eq!(moved.status(), 1.0);
    });

    assert_eq!(
        filesystem
            .read_project_path("/workspace/source.bin")
            .expect("source retained"),
        b"typed bytes"
    );
    assert!(filesystem
        .metadata_project_path("/workspace/copy.bin")
        .is_err());
    assert_eq!(
        filesystem
            .read_project_path("/workspace/moved.bin")
            .expect("moved copy retained"),
        b"typed bytes"
    );
}
