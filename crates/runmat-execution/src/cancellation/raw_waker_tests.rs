//! Raw executor hooks may synchronously reenter the cancellation authority.
//! No timing, sleeps, or scheduler races are needed to exercise these windows.
use super::*;
use std::mem::ManuallyDrop;
use std::task::{RawWaker, RawWakerVTable};

struct Hooks {
    source: CancellationSource,
    cancel_on_clone: AtomicBool,
    cancel_on_drop: AtomicBool,
    other: Mutex<Option<Cancelled>>,
    clones: AtomicUsize,
    drops: AtomicUsize,
    wakes: AtomicUsize,
}

impl Hooks {
    fn new(source: &CancellationSource) -> Arc<Self> {
        Arc::new(Self {
            source: source.clone(),
            cancel_on_clone: AtomicBool::new(false),
            cancel_on_drop: AtomicBool::new(false),
            other: Mutex::new(None),
            clones: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
            wakes: AtomicUsize::new(0),
        })
    }

    fn assert_reentrant(&self) {
        // Fail immediately instead of hanging if a regression invokes a hook
        // while holding the registry mutex.
        assert!(self.source.state.listeners.try_lock().is_ok());
        let _ = self.source.token().reason();
    }

    fn cloned(&self) {
        self.assert_reentrant();
        self.clones.fetch_add(1, Ordering::SeqCst);
        if self.cancel_on_clone.swap(false, Ordering::SeqCst) {
            self.source.cancel(CancellationReason::User);
        }
    }

    fn dropped(&self) {
        self.assert_reentrant();
        self.drops.fetch_add(1, Ordering::SeqCst);
        let other = self.other.lock().unwrap().take();
        drop(other);
        if self.cancel_on_drop.swap(false, Ordering::SeqCst) {
            self.source.cancel(CancellationReason::Deadline);
        }
    }

    fn woken(&self) {
        self.assert_reentrant();
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe fn clone_raw(data: *const ()) -> RawWaker {
    // SAFETY: every raw pointer owns one Arc created by raw(); cloning must
    // borrow that ownership and return exactly one new strong reference.
    let hooks = ManuallyDrop::new(unsafe { Arc::<Hooks>::from_raw(data.cast()) });
    hooks.cloned();
    raw(Arc::clone(&hooks))
}

unsafe fn wake_raw(data: *const ()) {
    // SAFETY: wake consumes this raw waker's one strong reference.
    let hooks = unsafe { Arc::<Hooks>::from_raw(data.cast()) };
    hooks.woken();
}

unsafe fn wake_by_ref_raw(data: *const ()) {
    // SAFETY: wake_by_ref borrows rather than consumes the strong reference.
    let hooks = ManuallyDrop::new(unsafe { Arc::<Hooks>::from_raw(data.cast()) });
    hooks.woken();
}

unsafe fn drop_raw(data: *const ()) {
    // SAFETY: drop consumes this raw waker's one strong reference.
    let hooks = unsafe { Arc::<Hooks>::from_raw(data.cast()) };
    hooks.dropped();
}

static VTABLE: RawWakerVTable = RawWakerVTable::new(clone_raw, wake_raw, wake_by_ref_raw, drop_raw);

fn raw(hooks: Arc<Hooks>) -> RawWaker {
    RawWaker::new(Arc::into_raw(hooks).cast(), &VTABLE)
}

fn waker(hooks: &Arc<Hooks>) -> Waker {
    // SAFETY: VTABLE observes the Arc reference ownership rules above.
    unsafe { Waker::from_raw(raw(hooks.clone())) }
}

#[test]
fn clone_cancels_between_fast_check_and_registration_without_lost_signal() {
    let source = CancellationSource::new();
    let hooks = Hooks::new(&source);
    hooks.cancel_on_clone.store(true, Ordering::SeqCst);
    let waker = waker(&hooks);
    let mut listener = source.token().cancelled();
    assert_eq!(
        poll(&mut listener, &waker),
        Poll::Ready(CancellationReason::User)
    );
    assert_eq!(hooks.clones.load(Ordering::SeqCst), 1);
    assert_eq!(
        hooks.wakes.load(Ordering::SeqCst),
        0,
        "no listener was registered yet"
    );
    assert!(source.state.listeners().slots.is_empty());
    drop(listener);
    drop(waker);
    assert_eq!(Arc::strong_count(&hooks), 1);
}

#[test]
fn replaced_waker_drop_can_unregister_sibling_and_cancel_replacement() {
    let source = CancellationSource::new();
    let hooks = Hooks::new(&source);
    let first = waker(&hooks);
    let replacement_count = Arc::new(WakeCount::default());
    let replacement = Waker::from(replacement_count.clone());
    let sibling_count = Arc::new(WakeCount::default());
    let mut sibling = source.token().cancelled();
    assert!(poll(&mut sibling, &Waker::from(sibling_count.clone())).is_pending());
    *hooks.other.lock().unwrap() = Some(sibling);
    let mut listener = source.token().cancelled();
    assert!(poll(&mut listener, &first).is_pending());
    hooks.cancel_on_drop.store(true, Ordering::SeqCst);
    assert_eq!(
        poll(&mut listener, &replacement),
        Poll::Ready(CancellationReason::Deadline)
    );
    assert_eq!(hooks.drops.load(Ordering::SeqCst), 1);
    assert_eq!(replacement_count.0.load(Ordering::Relaxed), 1);
    assert_eq!(sibling_count.0.load(Ordering::Relaxed), 0);
    assert!(hooks.other.lock().unwrap().is_none());
    drop(listener);
    drop(first);
    assert_eq!(Arc::strong_count(&hooks), 1);
}

#[test]
fn unregister_waker_drop_can_reenter_and_cancel_without_waking_removed_listener() {
    let source = CancellationSource::new();
    let hooks = Hooks::new(&source);
    let waker = waker(&hooks);
    let mut listener = source.token().cancelled();
    assert!(poll(&mut listener, &waker).is_pending());
    hooks.cancel_on_drop.store(true, Ordering::SeqCst);
    drop(listener);
    assert_eq!(source.token().reason(), Some(CancellationReason::Deadline));
    assert_eq!(hooks.drops.load(Ordering::SeqCst), 1);
    assert_eq!(hooks.wakes.load(Ordering::SeqCst), 0);
    drop(waker);
    assert_eq!(Arc::strong_count(&hooks), 1);
}
