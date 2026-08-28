use runmat_execution_transport_native::frame::ReplayWindow;
use runmat_execution_transport_native::TransportError;

#[test]
fn replay_window_accepts_reordering_once_and_rejects_stale_or_duplicate_frames() {
    let mut window = ReplayWindow::default();
    window.accept(70).unwrap();
    window.accept(68).unwrap();
    assert_eq!(window.accept(68), Err(TransportError::Replay));
    assert_eq!(window.accept(1), Err(TransportError::Replay));
    window.accept(71).unwrap();
}
