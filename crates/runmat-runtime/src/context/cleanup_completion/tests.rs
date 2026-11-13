use super::*;
use std::cell::Cell;
use std::marker::PhantomPinned;
use std::pin::Pin;
use std::task::Context;

struct CapturedOwner {
    value: Rc<u32>,
    dropped: Rc<Cell<bool>>,
    pending: Cell<bool>,
    _pin: PhantomPinned,
}

impl Future for CapturedOwner {
    type Output = Rc<u32>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.pending.replace(false) {
            context.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(Rc::clone(&self.value))
        }
    }
}

impl Drop for CapturedOwner {
    fn drop(&mut self) {
        self.dropped.set(true);
    }
}

#[test]
fn completed_pinned_future_releases_captures_before_publication_and_retains_output() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let value = Rc::new(42);
    let output = Rc::downgrade(&value);
    let dropped = Rc::new(Cell::new(false));
    let future = CapturedOwner {
        value,
        dropped: Rc::clone(&dropped),
        pending: Cell::new(true),
        _pin: PhantomPinned,
    };
    let cleanup = runtime.function_cleanup_scope().expect("cleanup scope");
    let (value, completion) =
        futures::executor::block_on(cleanup.run_with_post_poll(&runtime, future, |_| {
            assert!(dropped.get());
            assert_eq!(output.strong_count(), 1);
            Ok(())
        }))
        .expect("completed future");
    completion.expect("cleanup completion");
    assert_eq!(*value, 42);
}
