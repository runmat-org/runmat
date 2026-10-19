use super::*;
use crate::tree::tests::{fixture, grant, Fixture};
use runmat_package_cache::TreeInventoryEntry;

fn ordinary_payload() -> Fixture {
    fixture(vec![
        TreeInventoryEntry::file("bin/program", b"physical fixture".to_vec(), true),
        TreeInventoryEntry::file(
            "sysroot/Foo.framework/Resources/data",
            b"resource".to_vec(),
            false,
        ),
    ])
    .unwrap()
}

#[test]
fn physical_installation_survives_preparation_drop_and_explicit_pointer_replacement() {
    let root = tempfile::tempdir().unwrap();
    let storage = grant();
    let slot = NormalizedRelativePath::new("application").unwrap();
    let request = NativeInstallationRequest {
        store: root.path(),
        slot: slot.clone(),
        overwrite: false,
    };
    let source = ordinary_payload();
    let tree = source.resolve(&storage).unwrap();
    let manifest = tree.manifest.clone();
    let identity = Digest::sha256("first physical generation");
    let physical_entries = storage.admit(&manifest).unwrap();
    assert_eq!(
        physical_entries, 6,
        "four implicit ancestor directories accompany the two files"
    );
    let exact = crate::tree::NativeStorageGrant {
        archive: runmat_package_cache::ArchiveLimits {
            max_entries: physical_entries + INSTALLATION_OVERHEAD_ENTRIES,
            ..storage.archive
        },
        ..storage
    };
    let short = crate::tree::NativeStorageGrant {
        archive: runmat_package_cache::ArchiveLimits {
            max_entries: exact.archive.max_entries - 1,
            ..storage.archive
        },
        ..storage
    };
    assert!(matches!(
        materialize(
            &tree,
            &manifest,
            b"expected manifest",
            identity,
            &short,
            &request
        ),
        Err(AotError::ResourceUnavailable { .. })
    ));
    assert_eq!(
        std::fs::read_dir(root.path()).unwrap().count(),
        0,
        "insufficient combined permission fails before creating a generation"
    );
    let first = materialize(
        &tree,
        &manifest,
        b"expected manifest",
        identity,
        &exact,
        &request,
    )
    .unwrap();
    drop(tree);
    drop(source);
    verify_generation_bytes(&first, &manifest, b"expected manifest").unwrap();
    assert_eq!(
        std::fs::read(first.join("payload/sysroot/Foo.framework/Resources/data")).unwrap(),
        b"resource"
    );
    let pointer: InstallationPointer =
        serde_json::from_slice(&std::fs::read(root.path().join(slot.as_str())).unwrap()).unwrap();
    assert_eq!(pointer.installation, identity);
    assert_eq!(
        std::fs::canonicalize(root.path())
            .unwrap()
            .join(pointer.generation.as_str()),
        first
    );
    let second_source = ordinary_payload();
    let second_tree = second_source.resolve(&storage).unwrap();
    let second = materialize(
        &second_tree,
        &manifest,
        b"next manifest",
        Digest::sha256("second generation"),
        &storage,
        &NativeInstallationRequest {
            overwrite: true,
            ..request
        },
    )
    .unwrap();
    assert_ne!(first, second);
    verify_generation_bytes(&first, &manifest, b"expected manifest").unwrap();
    verify_generation_bytes(&second, &manifest, b"next manifest").unwrap();
}

#[test]
fn ordinary_publication_failure_retains_completed_generation_and_previous_slot() {
    let root = tempfile::tempdir().unwrap();
    let storage = grant();
    let slot = NormalizedRelativePath::new("application").unwrap();
    std::fs::write(root.path().join(slot.as_str()), b"prior publication").unwrap();
    let source = ordinary_payload();
    let tree = source.resolve(&storage).unwrap();
    let manifest = tree.manifest.clone();
    let error = materialize(
        &tree,
        &manifest,
        b"expected manifest",
        Digest::sha256("generation"),
        &storage,
        &NativeInstallationRequest {
            store: root.path(),
            slot: slot.clone(),
            overwrite: false,
        },
    )
    .unwrap_err();
    let AotError::InstallationPublication { generation, source } = error else {
        panic!("installation recovery must be explicit")
    };
    assert!(matches!(*source, AotError::Publication { .. }));
    assert_eq!(
        std::fs::read(root.path().join(slot.as_str())).unwrap(),
        b"prior publication"
    );
    verify_generation_bytes(&generation, &manifest, b"expected manifest").unwrap();
}

#[test]
fn failed_store_synchronization_retains_generation_before_any_pointer_handoff() {
    let root = tempfile::tempdir().unwrap();
    let generation = tempfile::tempdir_in(root.path()).unwrap();
    physical::write_new(&generation.path().join("completed"), b"retained").unwrap();
    let error = retain_generation_with(generation, root.path(), |store| {
        Err(AotError::io(
            "injected store sync failure",
            store,
            std::io::Error::other("injected I/O failure"),
        ))
    })
    .unwrap_err();
    let AotError::InstallationPublication { generation, .. } = error else {
        panic!("structured generation recovery required")
    };
    assert_eq!(
        std::fs::read(generation.join("completed")).unwrap(),
        b"retained"
    );
    assert!(!root.path().join("application").exists());
}
