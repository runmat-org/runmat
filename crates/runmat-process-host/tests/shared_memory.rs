use runmat_process_host::shared_memory::{SharedMemoryDescriptor, SharedMemoryKind};

#[test]
fn descriptor_rejects_zero_length_regions() {
    let descriptor = SharedMemoryDescriptor {
        kind: SharedMemoryKind::FileBacked,
        name: "region".into(),
        byte_length: 0,
        nonce: [0; 16],
    };
    assert!(descriptor.validate().is_err());
}
