use super::*;
use crate::ObjectNamespace;

#[test]
fn bounded_cold_reads_reject_before_payload_validation_and_preserve_the_original() {
    let directory = tempfile::tempdir().unwrap();
    let mut store = FilesystemObjectStore::open(directory.path().join("objects"), 1024).unwrap();
    let object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "bounded",
        "test/v1",
        vec![7; 128],
    )
    .unwrap();
    store.write_verified(&object).unwrap();
    let digest = object.descriptor.digest;
    drop(store);
    let store = FilesystemObjectStore::open(directory.path().join("objects"), 1024).unwrap();
    for allowance in [0, 1, 127] {
        assert!(matches!(
            store.read_verified_bounded(digest, allowance),
            Err(ArtifactError::Limit(_))
        ));
    }
    assert_eq!(
        store.read_verified_bounded(digest, 128).unwrap(),
        Some(object.bytes.clone())
    );
    assert_eq!(store.read_verified(digest).unwrap(), Some(object.bytes));
    // Oversized poisoned bytes must reject at the size fence, before hash validation.
    fs::write(store.path(digest), vec![9; 129]).unwrap();
    assert!(matches!(
        store.read_verified_bounded(digest, 128),
        Err(ArtifactError::Limit(_))
    ));
    assert!(matches!(
        store.read_verified_bounded(digest, 129),
        Err(ArtifactError::Identity(_))
    ));
    assert_eq!(
        store
            .read_verified_bounded(Digest::sha256(b"absent"), 128)
            .unwrap(),
        None
    );
}

#[test]
fn provider_without_a_preallocation_bound_never_falls_back_to_an_unbounded_read() {
    struct Unbounded;
    impl CacheImport for Unbounded {
        fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
            panic!("unbounded read must not execute");
        }
    }
    assert!(matches!(
        Unbounded.read_verified_bounded(Digest::sha256(b"any"), 128),
        Err(ArtifactError::Limit(_))
    ));
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Filesystem reads admit the actual object buffer before allocation, within the caller
//   bound. Denial, missing objects and poisoned bytes preserve bounded reads and content checks.
//   This is declared buffer storage, not allocator overhead, measured process RSS or numerical
//   qualification.
#[test]
fn admitted_reads_charge_actual_buffer_before_payload_and_preserve_hostile_bounds() {
    let directory = tempfile::tempdir().unwrap();
    let mut store = FilesystemObjectStore::open(directory.path().join("objects"), 1 << 30).unwrap();
    let object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "admitted",
        "test/v1",
        vec![7; 128],
    )
    .unwrap();
    store.write_verified(&object).unwrap();
    let digest = object.descriptor.digest;
    let mut amounts = Vec::new();
    let value = store
        .read_verified_bounded_with_admission(digest, 1 << 30, &mut |bytes| {
            amounts.push(bytes);
            Ok(())
        })
        .unwrap()
        .unwrap();
    assert_eq!(amounts, [128]);
    assert_eq!(value, object.bytes);
    let mut called = false;
    assert!(store
        .read_verified_bounded_with_admission(digest, 127, &mut |_| {
            called = true;
            Ok(())
        })
        .is_err());
    assert!(
        !called,
        "reject metadata length before requesting payload storage"
    );
    fs::write(store.path(digest), vec![9; 128]).unwrap();
    let error = store
        .read_verified_bounded_with_admission(digest, 1 << 30, &mut |_| {
            Err(ArtifactError::Limit("allocation denied before read".into()))
        })
        .unwrap_err();
    assert!(error.to_string().contains("allocation denied before read"));
    assert!(matches!(
        store.read_verified_bounded_with_admission(digest, 1 << 30, &mut |_| Ok(())),
        Err(ArtifactError::Identity(_))
    ));
    assert!(store
        .read_verified_bounded_with_admission(Digest::sha256(b"absent"), 1 << 30, &mut |_| panic!(
            "missing bytes must not request storage"
        ))
        .unwrap()
        .is_none());
}
