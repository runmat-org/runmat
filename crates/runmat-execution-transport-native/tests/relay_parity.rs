use runmat_execution_transport_native::frame::{FrameKind, FrameLimits, WireFrame};
use runmat_execution_transport_native::overlay::{
    DirectQuicRoute, OpaqueRelayRoute, OverlaySession, WebSocketRelayConnection,
};

#[test]
fn direct_and_relay_routes_carry_the_exact_same_application_frame() {
    let direct = DirectQuicRoute::new("node.example:443").unwrap();
    let relay = OpaqueRelayRoute::new("relay.example").unwrap();
    assert_ne!(direct.0, relay.0);
    let mut session = OverlaySession::new([7; 16]);
    let frame = session.frame(FrameKind::Artifact, vec![1, 2, 3]).unwrap();
    let encoded = frame.encode(FrameLimits::default()).unwrap();
    assert_eq!(
        WireFrame::decode(&encoded, FrameLimits::default()).unwrap(),
        frame
    );
}

#[tokio::test]
async fn websocket_relay_duplex_preserves_the_exact_bounded_frame() {
    use futures_util::{SinkExt as _, StreamExt as _};

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        while let Some(message) = socket.next().await {
            let message = message.unwrap();
            if message.is_binary() {
                socket.send(message).await.unwrap();
                break;
            }
        }
    });
    let relay =
        WebSocketRelayConnection::connect(&format!("ws://{address}"), &[], FrameLimits::default())
            .await
            .unwrap()
            .into_duplex();
    let mut session = OverlaySession::new([9; 16]);
    let expected = session
        .frame(FrameKind::Control, vec![3, 1, 4, 1, 5])
        .unwrap();
    relay.send(expected.clone()).await.unwrap();
    assert_eq!(relay.receive().await.unwrap(), expected);
    server.await.unwrap();
}
