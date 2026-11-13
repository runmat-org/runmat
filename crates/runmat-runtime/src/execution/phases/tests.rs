use super::*;
use std::{
    cell::Cell,
    marker::PhantomPinned,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

struct Probe {
    address: Cell<Option<usize>>,
    ready: Rc<Cell<bool>>,
    polls: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
    error: bool,
    value: usize,
    _pin: PhantomPinned,
}
impl Future for Probe {
    type Output = Result<usize, &'static str>;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_ref().get_ref();
        let address = this as *const Self as usize;
        assert_eq!(
            this.address.replace(Some(address)).unwrap_or(address),
            address
        );
        this.polls.set(this.polls.get() + 1);
        if !this.ready.get() {
            Poll::Pending
        } else if this.error {
            Poll::Ready(Err("original phase error"))
        } else {
            Poll::Ready(Ok(this.value))
        }
    }
}
impl Drop for Probe {
    fn drop(&mut self) {
        if let Some(address) = self.address.get() {
            assert_eq!(address, self as *const Self as usize);
        }
        self.drops.set(self.drops.get() + 1);
    }
}
struct Wakes(AtomicUsize);
impl futures::task::ArcWake for Wakes {
    fn wake_by_ref(this: &Arc<Self>) {
        this.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn phase_transition_preserves_pin_retirement_errors_and_abandonment() {
    for mode in 0..4 {
        let first_drops = Rc::new(Cell::new(0));
        let next_drops = Rc::new(Cell::new(0));
        let next_polls = Rc::new(Cell::new(0));
        let calls = Rc::new(Cell::new(0));
        let ready = Rc::new(Cell::new(false));
        let first = Probe {
            address: Cell::new(None),
            ready: Rc::new(Cell::new(true)),
            polls: Rc::new(Cell::new(0)),
            drops: Rc::clone(&first_drops),
            error: mode == 1,
            value: 7,
            _pin: PhantomPinned,
        };
        let (observed_calls, drops, polls, next_ready) = (
            Rc::clone(&calls),
            Rc::clone(&next_drops),
            Rc::clone(&next_polls),
            Rc::clone(&ready),
        );
        let mut work = Box::pin(after_result(first, move |value| {
            observed_calls.set(observed_calls.get() + 1);
            Probe {
                address: Cell::new(None),
                ready: next_ready,
                polls,
                drops,
                error: false,
                value: value * 6,
                _pin: PhantomPinned,
            }
        }));
        let wakes = Arc::new(Wakes(AtomicUsize::new(0)));
        let waker = futures::task::waker(Arc::clone(&wakes));
        let mut context = Context::from_waker(&waker);
        if mode != 0 {
            if mode == 1 {
                assert_eq!(
                    work.as_mut().poll(&mut context),
                    Poll::Ready(Err("original phase error"))
                );
            } else {
                assert_eq!(work.as_mut().poll(&mut context), Poll::Pending);
                assert_eq!(first_drops.get(), 1);
                assert_eq!(next_polls.get(), 1);
                assert_eq!(wakes.0.load(Ordering::Relaxed), 0);
                assert_eq!(work.as_mut().poll(&mut context), Poll::Pending);
                if mode == 3 {
                    ready.set(true);
                    assert_eq!(work.as_mut().poll(&mut context), Poll::Ready(Ok(42)));
                }
            }
        }
        drop(work);
        assert_eq!(first_drops.get(), 1);
        assert_eq!(calls.get(), usize::from(mode >= 2));
        assert_eq!(next_drops.get(), calls.get());
    }
}

#[test]
fn transition_unwind_retains_predecessor_roots_and_drops_each_owner_once() {
    struct First {
        drops: Rc<Cell<usize>>,
        panic_on_drop: bool,
        _pin: PhantomPinned,
    }
    impl Future for First {
        type Output = Result<(), ()>;
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Ready(Ok(()))
        }
    }
    impl Drop for First {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
            assert!(!self.panic_on_drop, "predecessor destructor failure");
        }
    }
    struct SuccessorOwner(Rc<Cell<usize>>);
    impl Drop for SuccessorOwner {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    for factory_panics in [false, true] {
        let first_drops = Rc::new(Cell::new(0));
        let successor_drops = Rc::new(Cell::new(0));
        let observed_first = Rc::clone(&first_drops);
        let successor_owner = SuccessorOwner(Rc::clone(&successor_drops));
        let mut work = Box::pin(after_result(
            First {
                drops: Rc::clone(&first_drops),
                panic_on_drop: !factory_panics,
                _pin: PhantomPinned,
            },
            move |()| {
                // The factory can still consult roots held by the predecessor.
                assert_eq!(observed_first.get(), 0);
                assert!(!factory_panics, "successor construction failure");
                async move {
                    drop(successor_owner);
                    Ok::<(), ()>(())
                }
            },
        ));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            work.as_mut().poll(&mut context)
        }))
        .is_err());
        // Pin::set retains the replacement if predecessor destruction panics;
        // a panicking factory instead destroys its captured owner immediately.
        assert_eq!(successor_drops.get(), usize::from(factory_panics));
        // A caught factory/destructor panic cannot repoll a completed child.
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            work.as_mut().poll(&mut context)
        }))
        .is_err());
        drop(work);
        assert_eq!(first_drops.get(), 1);
        assert_eq!(successor_drops.get(), 1);
    }
}
