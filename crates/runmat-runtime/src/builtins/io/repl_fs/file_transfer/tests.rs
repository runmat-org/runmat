use std::sync::Arc;

use runmat_filesystem::MemoryFsProvider;
use runmat_value::Value;

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn copy_and_move_use_the_portable_filesystem_provider() {
    let _lock = runmat_filesystem::provider_override_lock();
    let filesystem = MemoryFsProvider::with_current_dir("/workspace");
    filesystem
        .write_project_path("/workspace/source.bin", b"typed bytes")
        .expect("seed source");

    runmat_filesystem::with_provider_override(Arc::new(filesystem.clone()), || {
        futures::executor::block_on(runmat_filesystem::set_current_dir_async("/workspace"))
            .expect("select the fixture's directory without changing provider swap semantics");
        let copied = futures::executor::block_on(super::copyfile::execute::evaluate(
            &[Value::from("source.bin"), Value::from("copy.bin")],
            None,
        ))
        .expect("copy through memory provider");
        assert_eq!(copied.status(), 1.0, "{copied:?}");

        let moved = futures::executor::block_on(super::movefile::execute::evaluate(
            crate::call::FunctionArgumentView::from_slice(&[
                Value::from("copy.bin"),
                Value::from("moved.bin"),
            ]),
            None,
        ))
        .expect("move through memory provider");
        assert_eq!(moved.status(), 1.0, "{moved:?}");
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
