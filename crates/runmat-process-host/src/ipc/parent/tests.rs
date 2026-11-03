use super::*;

#[test]
fn bounded_normal_slot_does_not_block_host_replies_and_overflow_fails_closed() {
    let limits = FrameLimits {
        max_message_bytes: 1024,
    };
    let mut pipe = Vec::new();
    for frame in [&b"cancel"[..], &b"renewal"[..], &b"overflow"[..]] {
        crate::ipc::write_payload_blocking(&mut pipe, frame, limits).unwrap();
    }
    let replies = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = replies.clone();
    let observer: ParentFrameObserver = Arc::new(move |frame| {
        if frame == b"renewal" {
            observed.fetch_add(1, Ordering::Release);
            true
        } else {
            false
        }
    });
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
    assert!(forward_frames_with_send(
        &pipe[..],
        limits,
        &AtomicBool::new(true),
        Some(&observer),
        |frame| sender.try_send(frame).is_ok()
    ));
    assert_eq!(replies.load(Ordering::Acquire), 1);
    assert_eq!(receiver.try_recv().unwrap().unwrap(), b"cancel");
    assert!(receiver.try_recv().is_err());
}

#[test]
fn private_observer_consumes_only_owned_controls_and_preserves_other_frame_order() {
    let limits = FrameLimits {
        max_message_bytes: 1024,
    };
    let mut pipe = Vec::new();
    for frame in [&b"first"[..], &b"renewal"[..], &b"last"[..]] {
        crate::ipc::write_payload_blocking(&mut pipe, frame, limits).unwrap();
    }
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let observer: ParentFrameObserver = Arc::new(|frame| frame == b"renewal");
    assert!(forward_frames(
        &pipe[..],
        limits,
        &AtomicBool::new(true),
        &sender,
        Some(&observer)
    ));
    assert_eq!(receiver.try_recv().unwrap().unwrap(), b"first");
    assert_eq!(receiver.try_recv().unwrap().unwrap(), b"last");
    assert!(receiver.try_recv().is_err());
}

#[test]
fn parent_loss_does_not_depend_on_executor_polling_and_disarm_preserves_success() {
    let armed = AtomicBool::new(true);
    assert!(parent_loss(&b""[..], &armed));
    assert!(parent_loss(&b"unexpected request"[..], &armed));
    armed.store(false, Ordering::Release);
    assert!(!parent_loss(&b""[..], &armed));
}

#[test]
fn transient_interruptions_do_not_end_an_attached_worker() {
    struct InterruptedOnce(bool);
    impl Read for InterruptedOnce {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            if !self.0 {
                self.0 = true;
                Err(std::io::ErrorKind::Interrupted.into())
            } else {
                Ok(0)
            }
        }
    }
    assert!(parent_loss(InterruptedOnce(false), &AtomicBool::new(true)));
}

#[test]
fn parent_link_forwards_control_frames_and_reports_loss_only_while_armed() {
    let limits = FrameLimits {
        max_message_bytes: 1024,
    };
    let mut pipe = Vec::new();
    for frame in [&b"cancel"[..], &b"collective"[..]] {
        crate::ipc::write_payload_blocking(&mut pipe, frame, limits).unwrap();
    }
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    // EOF after the frames while armed is parent loss; every earlier frame was forwarded.
    assert!(forward_frames(
        &pipe[..],
        limits,
        &AtomicBool::new(true),
        &sender,
        None
    ));
    assert_eq!(receiver.try_recv().unwrap().unwrap(), b"cancel");
    assert_eq!(receiver.try_recv().unwrap().unwrap(), b"collective");
    assert!(receiver.try_recv().is_err());
    // Disarmed (terminal frame published), EOF is delivered as the final item instead.
    assert!(!forward_frames(
        &pipe[..],
        limits,
        &AtomicBool::new(false),
        &sender,
        None
    ));
    assert!(receiver.try_recv().unwrap().is_ok());
    assert!(receiver.try_recv().unwrap().is_ok());
    assert!(receiver.try_recv().unwrap().is_err());
    // A truncated frame is an input error: parent loss while armed.
    assert!(forward_frames(
        &pipe[..3],
        limits,
        &AtomicBool::new(true),
        &sender,
        None
    ));
}
