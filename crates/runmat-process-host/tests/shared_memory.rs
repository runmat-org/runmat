use runmat_process_host::shared_memory::{
    SharedMemoryDescriptor, SharedMemoryKind, SharedSnapshotStore,
};

#[test]
fn descriptor_rejects_zero_length_regions() {
    let descriptor = SharedMemoryDescriptor {
        kind: SharedMemoryKind::FileBacked,
        name: "region".into(),
        byte_length: 0,
        nonce: [0; 16],
        sha256: [0; 32],
    };
    assert!(descriptor.validate().is_err());
}

#[test]
fn private_store_transfers_and_removes_verified_snapshots() {
    let owner = SharedSnapshotStore::create().unwrap();
    let peer = SharedSnapshotStore::open_existing(owner.root_path()).unwrap();
    let descriptor = owner.publish(b"typed value bytes").unwrap();
    let path = owner.root_path().join(&descriptor.name);
    assert!(path.is_file());
    assert_eq!(
        peer.consume(&descriptor, 1024).unwrap(),
        b"typed value bytes"
    );
    assert!(!path.exists());
}

#[test]
fn private_store_rejects_tampering_and_cleans_the_file() {
    let owner = SharedSnapshotStore::create().unwrap();
    let peer = SharedSnapshotStore::open_existing(owner.root_path()).unwrap();
    let descriptor = owner.publish(b"original").unwrap();
    let path = owner.root_path().join(&descriptor.name);
    std::fs::write(&path, b"tampered").unwrap();
    assert!(peer.consume(&descriptor, 1024).is_err());
    assert!(!path.exists());
}
