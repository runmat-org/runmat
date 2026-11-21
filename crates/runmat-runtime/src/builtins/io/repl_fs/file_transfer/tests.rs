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

#[test]
fn move_succeeds_for_a_pinned_execution_output_and_refuses_a_peer_file() {
    use runmat_filesystem::pinned::{InputManifest, PinnedFsProvider};
    let _lock = runmat_filesystem::provider_override_lock();
    let base = MemoryFsProvider::with_current_dir("/workspace");
    base.write_project_path("/workspace/peer.bin", b"peer")
        .expect("seed a file outside the snapshot");
    let pinned = PinnedFsProvider::new(
        Arc::new(base.clone()),
        InputManifest {
            snapshot_id: "capture".into(),
            entries: Vec::new(),
        },
    )
    .expect("pinned provider");

    runmat_filesystem::with_provider_override(Arc::new(pinned), || {
        runmat_filesystem::write("/workspace/result.bin", b"result").expect("run output");
        let moved = futures::executor::block_on(super::movefile::execute::evaluate(
            crate::call::FunctionArgumentView::from_slice(&[
                Value::from("/workspace/result.bin"),
                Value::from("/workspace/moved.bin"),
            ]),
            None,
        ))
        .expect("move an owned output");
        assert_eq!(moved.status(), 1.0, "{moved:?}");
        let refused = futures::executor::block_on(super::movefile::execute::evaluate(
            crate::call::FunctionArgumentView::from_slice(&[
                Value::from("/workspace/peer.bin"),
                Value::from("/workspace/taken.bin"),
            ]),
            None,
        ));
        assert!(
            refused.map_or(true, |outcome| outcome.status() != 1.0),
            "a file outside the execution must not move"
        );
    });

    assert_eq!(
        base.read_project_path("/workspace/moved.bin")
            .expect("moved"),
        b"result"
    );
    assert!(base.metadata_project_path("/workspace/result.bin").is_err());
    assert_eq!(
        base.read_project_path("/workspace/peer.bin")
            .expect("peer kept"),
        b"peer"
    );
    assert!(base.metadata_project_path("/workspace/taken.bin").is_err());
}
