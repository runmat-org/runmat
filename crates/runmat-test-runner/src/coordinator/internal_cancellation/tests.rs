use super::*;
use crate::host::NeverCancelled;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Wake, Waker};

#[derive(Default)]
struct Wakes(AtomicUsize);
impl Wake for Wakes {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn concurrent_pending_groups_all_wake_without_cancelling_other_runs() {
    let internal = InternalCancellation::default();
    let external = NeverCancelled;
    let groups = (0..3)
        .map(|_| CombinedCancellation::new(&external, internal.clone()))
        .collect::<Vec<_>>();
    let mut pending = groups
        .iter()
        .map(CancellationPort::cancelled)
        .collect::<Vec<_>>();
    let wakes = (0..3)
        .map(|_| Arc::new(Wakes::default()))
        .collect::<Vec<_>>();
    for (future, wake) in pending.iter_mut().zip(&wakes) {
        let waker = Waker::from(wake.clone());
        assert!(future
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
    }
    // Abandon one group without unregistering either surviving listener.
    drop(pending.pop().unwrap());
    let unrelated = InternalCancellation::default();
    internal.cancel("fatal assertion");
    internal.cancel("later reason must not overwrite");
    assert_eq!(wakes[2].0.load(Ordering::SeqCst), 0);
    for (future, wake) in pending.iter_mut().zip(&wakes) {
        assert_eq!(wake.0.load(Ordering::SeqCst), 1);
        let waker = Waker::from(wake.clone());
        assert_eq!(
            future.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Ready("fatal assertion".into())
        );
    }
    assert!(!unrelated.is_cancelled());
    assert_eq!(unrelated.reason(), None);
    assert!(!external.is_cancelled());
    assert_eq!(internal.reason().as_deref(), Some("fatal assertion"));
}

struct Reenter {
    internal: InternalCancellation,
    calls: AtomicUsize,
}
impl Wake for Reenter {
    fn wake(self: Arc<Self>) {
        assert!(self.internal.is_cancelled());
        assert_eq!(self.internal.reason().as_deref(), Some("first reason"));
        self.internal.cancel("reentrant reason");
        self.calls.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn group_waker_can_reenter_reason_and_cancel_without_lock_recursion() {
    let internal = InternalCancellation::default();
    let external = NeverCancelled;
    let combined = CombinedCancellation::new(&external, internal.clone());
    let mut pending = combined.cancelled();
    let wake = Arc::new(Reenter {
        internal: internal.clone(),
        calls: AtomicUsize::new(0),
    });
    let waker = Waker::from(wake.clone());
    assert!(pending
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    internal.cancel("first reason");
    assert_eq!(wake.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        pending.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Ready("first reason".into())
    );
}

#[test]
fn signal_before_group_registration_remains_observable() {
    let internal = InternalCancellation::default();
    internal.cancel("already cancelled");
    let external = NeverCancelled;
    let combined = CombinedCancellation::new(&external, internal);
    let mut pending = combined.cancelled();
    let waker = Waker::from(Arc::new(Wakes::default()));
    assert_eq!(
        pending.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Ready("already cancelled".into())
    );
}

struct External(InternalCancellation);
impl CancellationPort for External {
    fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
    fn reason(&self) -> Option<String> {
        self.0.reason()
    }
    fn cancelled<'a>(&'a self) -> PortFuture<'a, String> {
        Box::pin(self.0.cancelled())
    }
}

#[test]
fn independent_external_group_cancellation_does_not_signal_siblings() {
    let internal = InternalCancellation::default();
    let first = External(InternalCancellation::default());
    let second = External(InternalCancellation::default());
    let group_one = CombinedCancellation::new(&first, internal.clone());
    let group_two = CombinedCancellation::new(&second, internal.clone());
    let mut one = group_one.cancelled();
    let mut two = group_two.cancelled();
    let first_wakes = Arc::new(Wakes::default());
    let second_wakes = Arc::new(Wakes::default());
    let first_waker = Waker::from(first_wakes.clone());
    let second_waker = Waker::from(second_wakes.clone());
    assert!(one
        .as_mut()
        .poll(&mut Context::from_waker(&first_waker))
        .is_pending());
    assert!(two
        .as_mut()
        .poll(&mut Context::from_waker(&second_waker))
        .is_pending());
    first.0.cancel("external request");
    assert_eq!(first_wakes.0.load(Ordering::SeqCst), 1);
    assert_eq!(second_wakes.0.load(Ordering::SeqCst), 0);
    assert_eq!(
        one.as_mut().poll(&mut Context::from_waker(&first_waker)),
        Poll::Ready("external request".into())
    );
    assert!(!internal.is_cancelled());
    assert!(!group_two.is_cancelled());
    internal.cancel("run abort");
    assert_eq!(second_wakes.0.load(Ordering::SeqCst), 1);
    assert_eq!(
        two.as_mut().poll(&mut Context::from_waker(&second_waker)),
        Poll::Ready("run abort".into())
    );
}
