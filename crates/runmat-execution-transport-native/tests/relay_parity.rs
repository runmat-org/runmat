use runmat_execution_transport_native::frame::{FrameKind, FrameLimits, WireFrame};
use runmat_execution_transport_native::overlay::{
    DirectQuicRoute, OpaqueRelayRoute, OverlaySession,
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
