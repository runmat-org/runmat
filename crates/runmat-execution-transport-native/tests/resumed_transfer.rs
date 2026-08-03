use runmat_execution_transport_native::transfer::{verify_ciphertext, OpaqueObject, ResumeState};

#[test]
fn resumed_transfer_is_contiguous_bounded_and_digest_verified() {
    let bytes = b"opaque ciphertext";
    let mut state = ResumeState::new(bytes.len() as u64).unwrap();
    state.accept(0, 6).unwrap();
    assert!(state.accept(5, 2).is_err());
    state.accept(6, bytes.len() - 6).unwrap();
    assert!(state.is_complete());
    verify_ciphertext(
        &OpaqueObject {
            ciphertext_digest:
                "sha256:c52017f9e6d5288e6ea8d8c228c4f09f5f8a838e58f6e120ae69c5250c929a04"
                    .to_string(),
            ciphertext_size_bytes: bytes.len() as u64,
        },
        bytes,
    )
    .unwrap();
}
