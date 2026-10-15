use runmat_filesystem::MemoryFsProvider;
use std::sync::Arc;

#[test]
fn captured_copy_owner_survives_global_provider_swap() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(MemoryFsProvider::with_current_dir("/selected"));
    selected
        .write_project_path("/selected/source.bin", b"selected")
        .expect("virtual source");
    let other = Arc::new(MemoryFsProvider::with_current_dir("/other"));
    other
        .write_project_path("/selected/source.bin", b"wrong")
        .expect("lexically identical other-provider source");
    let mut captured = runmat_filesystem::with_provider_override(selected.clone(), || {
        super::super::super::TransferFilesystem::capture(
            super::super::super::super::authority::TransferOperation::Copy,
            None,
        )
        .expect("capture selected owner")
    });
    let _replacement = runmat_filesystem::replace_provider(other.clone());
    let outcome = futures::executor::block_on(super::super::super::single::copy(
        &mut captured,
        "source.bin",
        "target.bin",
        false,
    ));
    assert_eq!(outcome.status(), 1.0);
    assert_eq!(
        selected.read_project_path("/selected/target.bin").unwrap(),
        b"selected"
    );
    assert!(other.read_project_path("/other/target.bin").is_err());
}

#[test]
fn captured_copy_owner_applies_to_wildcards_and_recursive_children() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(MemoryFsProvider::with_current_dir("/selected"));
    selected
        .write_project_path("/selected/tree/child/item.bin", b"selected child")
        .expect("selected tree");
    selected
        .write_project_path("/selected/match-a.bin", b"selected wildcard")
        .expect("selected match");
    selected
        .write_project_path("/selected/matches/.keep", b"")
        .expect("selected destination");
    let other = Arc::new(MemoryFsProvider::with_current_dir("/other"));
    other
        .write_project_path("/selected/tree/child/item.bin", b"wrong child")
        .expect("other tree");
    other
        .write_project_path("/selected/match-a.bin", b"wrong wildcard")
        .expect("other match");
    let mut captured = runmat_filesystem::with_provider_override(selected.clone(), || {
        super::super::super::TransferFilesystem::capture(
            super::super::super::super::authority::TransferOperation::Copy,
            None,
        )
        .expect("capture selected owner")
    });
    let _replacement = runmat_filesystem::replace_provider(other.clone());
    let recursive = futures::executor::block_on(super::super::super::single::copy(
        &mut captured,
        "tree",
        "tree-copy",
        false,
    ));
    assert_eq!(recursive.status(), 1.0, "{recursive:?}");
    let wildcard = futures::executor::block_on(super::super::super::wildcard::copy(
        &mut captured,
        "match-*.bin",
        "matches",
        false,
    ));
    assert_eq!(wildcard.status(), 1.0, "{wildcard:?}");
    assert_eq!(
        selected
            .read_project_path("/selected/tree-copy/child/item.bin")
            .unwrap(),
        b"selected child"
    );
    assert_eq!(
        selected
            .read_project_path("/selected/matches/match-a.bin")
            .unwrap(),
        b"selected wildcard"
    );
    assert!(other
        .read_project_path("/selected/tree-copy/child/item.bin")
        .is_err());
    assert!(other
        .read_project_path("/selected/matches/match-a.bin")
        .is_err());
}
