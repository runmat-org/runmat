use super::*;
use runmat_execution::profiling::{
    CommandProfilingRoot, CommandProfilingSnapshot, DeferredWorkCounts,
};
use std::cell::RefCell;
use std::rc::Rc;

struct PendingRemove {
    root: Rc<RefCell<Option<CommandProfilingRoot>>>,
    observed: Rc<RefCell<Option<CommandProfilingSnapshot>>>,
}

impl Future for PendingRemove {
    type Output = std::io::Result<()>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingRemove {
    fn drop(&mut self) {
        let root = self.root.borrow_mut().take().expect("one command root");
        *self.observed.borrow_mut() = Some(root.seal());
    }
}

#[wasm_bindgen_test::wasm_bindgen_test]
fn abandoned_rollback_drops_inner_capture_before_its_linear_token() {
    let root = Rc::new(RefCell::new(Some(CommandProfilingRoot::new())));
    let observed = Rc::new(RefCell::new(None));
    let token = root
        .borrow()
        .as_ref()
        .unwrap()
        .receipt()
        .begin_deferred_work()
        .expect("cleanup admitted while command open");
    drop(DetachedPartialCleanup {
        inner: Some(Box::pin(PendingRemove {
            root,
            observed: observed.clone(),
        })),
        token: Some(token),
    });
    assert_eq!(
        observed.borrow().unwrap().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            pending: 1,
            ..DeferredWorkCounts::default()
        }),
        "the captured removal future is destroyed before the token settles",
    );

    let root = CommandProfilingRoot::new();
    let token = root
        .receipt()
        .begin_deferred_work()
        .expect("second cleanup admitted");
    drop(DetachedPartialCleanup {
        inner: None,
        token: Some(token),
    });
    assert_eq!(
        root.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            missing: 1,
            ..DeferredWorkCounts::default()
        }),
    );
}
