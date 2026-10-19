use super::*;
use futures::task::noop_waker;

#[test]
fn queued_open_drop_excludes_host_start_without_creating_window() {
    let (sender, receiver) = mpsc::sync_channel(1);
    let host_lifetime = Arc::new(());
    let port = WindowOpenPort {
        sender,
        shutdown: Arc::new(AtomicBool::new(false)),
        host_lifetime: Arc::downgrade(&host_lifetime),
    };
    assert!(!port.is_closed());
    let mut ticket = Box::pin(port.open(7, WindowConfig::default()));
    let state = Arc::clone(&ticket.state);
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(ticket.as_mut().poll(&mut context).is_pending());
    assert_eq!(ticket.disposition(), WindowOpenDisposition::NotOpened);
    assert_eq!(
        state.load(Ordering::Acquire),
        QUEUED,
        "observation must not retire a queued request"
    );
    drop(ticket);
    assert_eq!(state.load(Ordering::Acquire), RETIRED);
    let request = receiver.try_recv().expect("queued metadata is still owned");
    assert!(request
        .state
        .compare_exchange(QUEUED, DISPATCHED, Ordering::AcqRel, Ordering::Acquire)
        .is_err());
    drop(host_lifetime);
    assert!(
        port.is_closed(),
        "an opener is not ownership of the host receiver"
    );
}

#[test]
fn dispatched_open_cannot_be_reclassified_as_no_effect() {
    let (sender, receiver) = mpsc::sync_channel(1);
    let port = WindowOpenPort {
        sender,
        shutdown: Arc::new(AtomicBool::new(false)),
        host_lifetime: Weak::new(),
    };
    let mut ticket = Box::pin(port.open(7, WindowConfig::default()));
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(ticket.as_mut().poll(&mut context).is_pending());
    let request = receiver.try_recv().expect("queued open");
    request
        .state
        .compare_exchange(QUEUED, DISPATCHED, Ordering::AcqRel, Ordering::Acquire)
        .expect("host wins dispatch");
    assert_eq!(ticket.abandon(), WindowOpenDisposition::Indeterminate);
    assert_eq!(ticket.disposition(), WindowOpenDisposition::Indeterminate);
    request.state.store(OPENED, Ordering::Release);
    assert_eq!(ticket.abandon(), WindowOpenDisposition::Opened);
    assert_eq!(ticket.disposition(), WindowOpenDisposition::Opened);
}

#[test]
fn finite_open_ingress_and_invalid_handle_fail_before_host_effects() {
    let (sender, receiver) = mpsc::sync_channel(1);
    let port = WindowOpenPort {
        sender,
        shutdown: Arc::new(AtomicBool::new(false)),
        host_lifetime: Weak::new(),
    };
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);

    let mut invalid = Box::pin(port.open(0, WindowConfig::default()));
    assert!(matches!(
        invalid.as_mut().poll(&mut context),
        Poll::Ready(Err(WindowOpenError::InvalidHandle))
    ));
    assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

    let mut first = Box::pin(port.open(7, WindowConfig::default()));
    assert!(first.as_mut().poll(&mut context).is_pending());
    let mut second = Box::pin(port.open(8, WindowConfig::default()));
    assert!(matches!(
        second.as_mut().poll(&mut context),
        Poll::Ready(Err(WindowOpenError::Full))
    ));
    drop(first);
    let admitted = receiver.try_recv().expect("first request retained");
    assert_eq!(admitted.figure_handle, 7);
    assert_eq!(admitted.state.load(Ordering::Acquire), RETIRED);
}
