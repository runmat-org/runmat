use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_transport_native::frame::{EncryptedFrameSession, FrameKind, FrameLimits};

fn session(run: &str, direction: &str) -> EncryptedFrameSession {
    EncryptedFrameSession::new(
        run,
        [7; 16],
        direction,
        1,
        RunKeyMaterial::from_entropy([9; 32]).unwrap(),
    )
    .unwrap()
}

#[test]
fn application_ciphertext_is_bound_to_run_direction_sequence_and_epoch() {
    let limits = FrameLimits::default();
    let mut sender = session("run-a", "submitter-to-driver");
    let mut receiver = session("run-a", "submitter-to-driver");
    let frame = sender
        .seal_with_entropy(FrameKind::Control, b"private command", [3; 32], limits)
        .unwrap();
    assert!(!frame
        .payload
        .windows(b"private command".len())
        .any(|window| window == b"private command"));
    assert_eq!(
        receiver.open(&frame, limits).unwrap(),
        b"private command".to_vec()
    );
    assert!(receiver.open(&frame, limits).is_err());

    let mut wrong_run = session("run-b", "submitter-to-driver");
    assert!(wrong_run.open(&frame, limits).is_err());
    let mut wrong_direction = session("run-a", "driver-to-submitter");
    assert!(wrong_direction.open(&frame, limits).is_err());
}

#[test]
fn derivation_salt_reuse_and_cross_frame_swaps_fail_closed() {
    let limits = FrameLimits::default();
    let mut sender = session("run-a", "submitter-to-driver");
    let first = sender
        .seal_with_entropy(FrameKind::Artifact, b"one", [4; 32], limits)
        .unwrap();
    assert!(sender
        .seal_with_entropy(FrameKind::Artifact, b"two", [4; 32], limits)
        .is_err());

    let mut receiver = session("run-a", "submitter-to-driver");
    let mut swapped = first.clone();
    swapped.sequence = 8;
    assert!(receiver.open(&swapped, limits).is_err());
    assert_eq!(receiver.open(&first, limits).unwrap(), b"one");
}
