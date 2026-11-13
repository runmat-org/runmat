use super::*;
use std::sync::atomic::AtomicUsize;
use std::sync::Barrier;
use std::task::Wake;

#[path = "raw_waker_tests.rs"]
mod raw_waker_tests;

#[derive(Default)]
struct WakeCount(AtomicUsize);

impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn poll(listener: &mut Cancelled, waker: &Waker) -> Poll<CancellationReason> {
    Pin::new(listener).poll(&mut Context::from_waker(waker))
}

#[test]
fn generation_identity_is_distinct_from_cancellation_state() {
    let first = CancellationSource::new();
    let second = CancellationSource::new();
    let token = first.token();
    assert!(token.same_generation(&token.clone()));
    assert!(!token.same_generation(&second.token()));
    let combined = token.or(&second.token());
    assert!(combined.same_generation(&token.or(&second.token())));
    assert!(!combined.same_generation(&second.token().or(&token)));
    first.cancel(CancellationReason::User);
    assert!(token.same_generation(&first.token()));
}

#[test]
fn cancellation_wakes_every_listener_and_keeps_first_reason() {
    let source = CancellationSource::new();
    let token = source.token();
    let counters = (0..8)
        .map(|_| Arc::new(WakeCount::default()))
        .collect::<Vec<_>>();
    let mut listeners = counters
        .iter()
        .map(|counter| {
            let mut listener = token.cancelled();
            assert!(poll(&mut listener, &Waker::from(counter.clone())).is_pending());
            listener
        })
        .collect::<Vec<_>>();
    assert!(!token.is_cancelled());
    assert!(source.cancel(CancellationReason::User));
    assert!(!source.cancel(CancellationReason::Shutdown));
    for (listener, counter) in listeners.iter_mut().zip(&counters) {
        assert_eq!(counter.0.load(Ordering::Relaxed), 1);
        assert_eq!(
            poll(listener, &Waker::from(counter.clone())),
            Poll::Ready(CancellationReason::User)
        );
    }
    assert!(token.states()[0].listeners().slots.is_empty());
}

#[test]
fn drop_and_repoll_release_only_their_listener() {
    let source = CancellationSource::new();
    let token = source.token();
    drop(token.cancelled());
    assert!(token.states()[0].listeners().slots.is_empty());
    let first = Arc::new(WakeCount::default());
    let second = Arc::new(WakeCount::default());
    let mut dropped = token.cancelled();
    assert!(poll(&mut dropped, &Waker::from(first.clone())).is_pending());
    drop(dropped);
    assert_eq!(Arc::strong_count(&first), 1);
    let mut live = token.cancelled();
    assert!(poll(&mut live, &Waker::from(first.clone())).is_pending());
    assert!(poll(&mut live, &Waker::from(second.clone())).is_pending());
    assert_eq!(Arc::strong_count(&first), 1);
    source.cancel(CancellationReason::Deadline);
    assert_eq!(first.0.load(Ordering::Relaxed), 0);
    assert_eq!(second.0.load(Ordering::Relaxed), 1);
}

#[test]
fn cancellation_and_registration_race_has_no_lost_wakeup() {
    for _ in 0..64 {
        let source = CancellationSource::new();
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = barrier.clone();
        let worker_source = source.clone();
        let worker = std::thread::spawn(move || {
            worker_barrier.wait();
            worker_source.cancel(CancellationReason::User);
        });
        let counter = Arc::new(WakeCount::default());
        let waker = Waker::from(counter.clone());
        let mut listener = source.token().cancelled();
        barrier.wait();
        let first = poll(&mut listener, &waker);
        worker.join().unwrap();
        if first.is_pending() {
            assert_eq!(counter.0.load(Ordering::Relaxed), 1);
        }
        assert_eq!(
            poll(&mut listener, &waker),
            Poll::Ready(CancellationReason::User)
        );
    }
}

struct ReenterOnWake {
    source: CancellationSource,
    other: Mutex<Option<Cancelled>>,
    count: AtomicUsize,
}

impl Wake for ReenterOnWake {
    fn wake(self: Arc<Self>) {
        assert_eq!(self.source.token().reason(), Some(CancellationReason::User));
        assert!(!self.source.cancel(CancellationReason::Shutdown));
        drop(self.other.lock().unwrap().take());
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn wake_can_reenter_and_drop_another_listener() {
    let source = CancellationSource::new();
    let counter = Arc::new(WakeCount::default());
    let mut other = source.token().cancelled();
    assert!(poll(&mut other, &Waker::from(counter.clone())).is_pending());
    let reentrant = Arc::new(ReenterOnWake {
        source: source.clone(),
        other: Mutex::new(Some(other)),
        count: AtomicUsize::new(0),
    });
    let mut listener = source.token().cancelled();
    assert!(poll(&mut listener, &Waker::from(reentrant.clone())).is_pending());
    source.cancel(CancellationReason::User);
    assert_eq!(reentrant.count.load(Ordering::Relaxed), 1);
    assert!(reentrant.other.lock().unwrap().is_none());
}

#[test]
fn fresh_generation_does_not_revive_or_inherit_previous_cancellation() {
    let old = CancellationSource::new();
    let old_token = old.token();
    let fresh = CancellationSource::new();
    old.cancel(CancellationReason::ParentScope);
    assert!(old_token.is_cancelled());
    assert!(!fresh.token().is_cancelled());
    fresh.cancel(CancellationReason::User);
    assert_eq!(old_token.reason(), Some(CancellationReason::ParentScope));
}

#[test]
fn composed_token_wakes_for_either_parent_and_releases_all_registrations() {
    for cancelled_parent in 0..2 {
        let sources = [CancellationSource::new(), CancellationSource::new()];
        let token = sources[0].token().or(&sources[1].token());
        let counter = Arc::new(WakeCount::default());
        let waker = Waker::from(counter.clone());
        let mut listener = token.cancelled();
        assert!(poll(&mut listener, &waker).is_pending());
        sources[cancelled_parent].cancel(CancellationReason::User);
        assert_eq!(counter.0.load(Ordering::Relaxed), 1);
        assert_eq!(
            poll(&mut listener, &waker),
            Poll::Ready(CancellationReason::User)
        );
        assert!(token
            .states()
            .iter()
            .all(|state| { state.listeners().slots.iter().all(Option::is_none) }));
        // A retained completed listener must not receive unrelated later wakes.
        sources[1 - cancelled_parent].cancel(CancellationReason::Shutdown);
        assert_eq!(counter.0.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn composition_flattens_deduplicates_and_uses_stable_reason_precedence() {
    let first = CancellationSource::new();
    let second = CancellationSource::new();
    let token = first.token().or(&second.token());
    let mut nested = token.clone();
    for _ in 0..32 {
        nested = nested.or(&first.token()).or(&token);
    }
    assert_eq!(nested.states().len(), 2);
    second.cancel(CancellationReason::Deadline);
    first.cancel(CancellationReason::User);
    assert_eq!(nested.reason(), Some(CancellationReason::User));
    let fresh = CancellationSource::new();
    assert!(!fresh.token().is_cancelled());
    assert!(nested.is_cancelled());
}

#[test]
fn dropping_composed_listener_unregisters_each_parent() {
    let first = CancellationSource::new();
    let second = CancellationSource::new();
    let token = first.token().or(&second.token());
    let counter = Arc::new(WakeCount::default());
    let mut listener = token.cancelled();
    assert!(poll(&mut listener, &Waker::from(counter.clone())).is_pending());
    drop(listener);
    assert_eq!(Arc::strong_count(&counter), 1);
    first.cancel(CancellationReason::User);
    second.cancel(CancellationReason::Shutdown);
    assert_eq!(counter.0.load(Ordering::Relaxed), 0);
}
