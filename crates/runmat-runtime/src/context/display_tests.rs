use super::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use runmat_value::{get_display_format, FormatMode, Value};
use std::cell::Cell;
use std::future::Future;
use std::rc::Rc;
use std::task::{Context, Poll};

struct ObserveDrop {
    expected: FormatMode,
    count: Rc<Cell<usize>>,
}

impl Drop for ObserveDrop {
    fn drop(&mut self) {
        assert_eq!(get_display_format(), self.expected);
        self.count.set(self.count.get() + 1);
    }
}

#[test]
fn pending_sessions_keep_format_effects_through_nested_entry_ready_and_cancel() {
    let caller = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    caller.display_format_owner().set(FormatMode::Long);
    let _caller = caller.enter();
    let left = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let right = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let make = |runtime: RuntimeContext,
                option: &'static str,
                expected: FormatMode,
                release: futures::channel::oneshot::Receiver<()>,
                drops: Rc<Cell<usize>>| {
        let selected = runtime.clone();
        runtime.scope(async move {
            crate::call_builtin_async_with_outputs_in_context(
                &selected,
                "format",
                &[Value::String(option.into())],
                0,
            )
            .await
            .expect("format policy update");
            let _drop = ObserveDrop {
                expected,
                count: drops,
            };
            assert_eq!(get_display_format(), expected);
            {
                let _nested = selected.enter();
                assert_eq!(get_display_format(), expected);
            }
            release.await.expect("resume session");
            assert_eq!(get_display_format(), expected);
        })
    };
    let (left_tx, left_rx) = futures::channel::oneshot::channel();
    let (right_tx, right_rx) = futures::channel::oneshot::channel();
    let left_drops = Rc::new(Cell::new(0));
    let right_drops = Rc::new(Cell::new(0));
    let mut left_future = Box::pin(make(
        left.clone(),
        "hex",
        FormatMode::Hex,
        left_rx,
        Rc::clone(&left_drops),
    ));
    let mut right_future = Box::pin(make(
        right.clone(),
        "shortE",
        FormatMode::ShortE,
        right_rx,
        Rc::clone(&right_drops),
    ));
    let waker = futures::task::noop_waker();
    let mut cx = Context::from_waker(&waker);
    assert!(left_future.as_mut().poll(&mut cx).is_pending());
    assert_eq!(get_display_format(), FormatMode::Long);
    assert!(right_future.as_mut().poll(&mut cx).is_pending());
    assert_eq!(get_display_format(), FormatMode::Long);
    left_tx.send(()).expect("release left");
    assert!(matches!(
        left_future.as_mut().poll(&mut cx),
        Poll::Ready(())
    ));
    assert_eq!(get_display_format(), FormatMode::Long);
    drop(right_future);
    drop(right_tx);
    assert_eq!(left_drops.get(), 1);
    assert_eq!(right_drops.get(), 1);
    assert_eq!(get_display_format(), FormatMode::Long);
    assert_eq!(left.display_format_owner().get(), FormatMode::Hex);
    assert_eq!(right.display_format_owner().get(), FormatMode::ShortE);
    let lab = left
        .fork_parallel_lab(left.service_ports().clone())
        .expect("idle session fork");
    assert_eq!(lab.display_format_owner().get(), FormatMode::Hex);
    lab.display_format_owner().set(FormatMode::Rational);
    assert_eq!(left.display_format_owner().get(), FormatMode::Hex);
}
