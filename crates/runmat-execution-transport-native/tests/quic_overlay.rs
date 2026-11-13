use std::net::{Ipv4Addr, SocketAddr};

use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_transport_native::frame::{EncryptedFrameSession, FrameKind, FrameLimits};
use runmat_execution_transport_native::overlay::{
    PinnedQuicEndpoint, QuicOverlayConnection, QuicOverlayListener,
};

#[tokio::test]
async fn pinned_quic_carries_only_application_encrypted_frames() {
    let CertifiedKey { cert, signing_key } =
        generate_simple_self_signed(vec!["runmat.execution".into()]).unwrap();
    let certificate_der = cert.der().to_vec();
    let signing_key_der = signing_key.serialize_der();
    let listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![certificate_der.clone()],
        signing_key_der.clone(),
        FrameLimits::default(),
    )
    .unwrap();
    let authority = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let connection = listener.accept().await.unwrap();
        let frame = connection.receive().await.unwrap();
        let mut decrypt = EncryptedFrameSession::new(
            "run-quic",
            [1; 16],
            "submitter-to-driver",
            1,
            RunKeyMaterial::from_entropy([2; 32]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            decrypt.open(&frame, FrameLimits::default()).unwrap(),
            b"secret"
        );
    });
    let client = QuicOverlayConnection::connect(
        SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)),
        &PinnedQuicEndpoint {
            authority,
            server_name: "runmat.execution".into(),
            certificate_der: certificate_der.clone(),
        },
        FrameLimits::default(),
    )
    .await
    .unwrap();
    let mut encrypt = EncryptedFrameSession::new(
        "run-quic",
        [1; 16],
        "submitter-to-driver",
        1,
        RunKeyMaterial::from_entropy([2; 32]).unwrap(),
    )
    .unwrap();
    let frame = encrypt
        .seal_with_entropy(
            FrameKind::Control,
            b"secret",
            [3; 32],
            FrameLimits::default(),
        )
        .unwrap();
    client.send(&frame).await.unwrap();
    server.await.unwrap();

    let rejecting_listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![certificate_der],
        signing_key_der,
        FrameLimits::default(),
    )
    .unwrap();
    let rejecting_authority = rejecting_listener.local_addr().unwrap();
    let accept = tokio::spawn(async move { rejecting_listener.accept().await });
    let wrong_certificate = generate_simple_self_signed(vec!["runmat.execution".into()])
        .unwrap()
        .cert
        .der()
        .to_vec();
    assert!(QuicOverlayConnection::connect(
        SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)),
        &PinnedQuicEndpoint {
            authority: rejecting_authority,
            server_name: "runmat.execution".into(),
            certificate_der: wrong_certificate,
        },
        FrameLimits::default(),
    )
    .await
    .is_err());
    accept.abort();
}

/// The locked Quinn default idle timeout is 30s. Keepalive remains 10s. Stall
/// this current-thread caller beyond the unchanged idle limit: only the shared
/// network reactor can service the connection during that interval.
#[tokio::test(flavor = "current_thread")]
async fn quic_remains_live_while_its_current_thread_caller_is_blocked() {
    let CertifiedKey { cert, signing_key } =
        generate_simple_self_signed(vec!["runmat.execution".into()]).unwrap();
    let certificate = cert.der().to_vec();
    let listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![certificate.clone()],
        signing_key.serialize_der(),
        FrameLimits::default(),
    )
    .unwrap();
    let endpoint = PinnedQuicEndpoint {
        authority: listener.local_addr().unwrap(),
        server_name: "runmat.execution".into(),
        certificate_der: certificate,
    };
    let (client, server) = tokio::join!(
        QuicOverlayConnection::connect(
            SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)),
            &endpoint,
            FrameLimits::default()
        ),
        listener.accept(),
    );
    let client = client.unwrap();
    let server = server.unwrap();
    // Deliberately block the calling executor; spawn_blocking would not exercise
    // the regression. This does not raise any connection or task deadline.
    std::thread::sleep(std::time::Duration::from_secs(35));
    let mut encrypt = EncryptedFrameSession::new(
        "blocked-caller",
        [29; 16],
        "client-to-server",
        1,
        RunKeyMaterial::from_entropy([31; 32]).unwrap(),
    )
    .unwrap();
    let frame = encrypt
        .seal(FrameKind::Control, b"still live", FrameLimits::default())
        .unwrap();
    client.send(&frame).await.unwrap();
    let delivered = tokio::time::timeout(std::time::Duration::from_secs(5), server.receive())
        .await
        .unwrap()
        .unwrap();
    let mut decrypt = EncryptedFrameSession::new(
        "blocked-caller",
        [29; 16],
        "client-to-server",
        1,
        RunKeyMaterial::from_entropy([31; 32]).unwrap(),
    )
    .unwrap();
    assert_eq!(
        decrypt.open(&delivered, FrameLimits::default()).unwrap(),
        b"still live"
    );
    client.close();
    server.close();
}
