use super::super::{record_recent_figure, reset_recent_figures, take_recent_figures, FigureHandle};
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use futures::task::noop_waker;
use std::future::Future;
use std::rc::Rc;
use std::task::{Context, Poll};

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

fn handles() -> Vec<u32> {
    let mut values = take_recent_figures()
        .into_iter()
        .map(FigureHandle::as_u32)
        .collect::<Vec<_>>();
    values.sort_unstable();
    values
}

#[test]
fn nested_selected_calls_share_the_command_receipt_but_other_sessions_do_not() {
    let command = runtime().with_fresh_plot_receipts();
    let nested = command.clone();
    let other = runtime().with_fresh_plot_receipts();
    let _command = command.enter();
    reset_recent_figures();
    record_recent_figure(11_u32.into());
    {
        let _other = other.enter();
        reset_recent_figures();
        record_recent_figure(33_u32.into());
        assert_eq!(handles(), [33]);
    }
    {
        let _nested = nested.enter();
        record_recent_figure(22_u32.into());
    }
    assert_eq!(handles(), [11, 22]);
}

#[test]
fn old_selected_callback_cannot_enter_a_fresh_command_receipt() {
    let first = runtime().with_fresh_plot_receipts();
    let old_callback_context = first.clone();
    {
        let _first = first.enter();
        record_recent_figure(41_u32.into());
        assert_eq!(handles(), [41]);
    }
    let next = first.with_fresh_plot_receipts();
    let _next = next.enter();
    reset_recent_figures();
    record_recent_figure(43_u32.into());
    {
        let _old_callback = old_callback_context.enter();
        record_recent_figure(42_u32.into());
        assert_eq!(handles(), [42]);
    }
    assert_eq!(handles(), [43]);
}

#[test]
fn canceled_command_does_not_carry_touches_into_a_later_command() {
    let first = runtime().with_fresh_plot_receipts();
    let mut pending = Box::pin(first.clone().scope(async {
        record_recent_figure(51_u32.into());
        futures::future::pending::<()>().await;
    }));
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
    drop(pending);
    let next = first.with_fresh_plot_receipts();
    let _next = next.enter();
    reset_recent_figures();
    record_recent_figure(52_u32.into());
    assert_eq!(handles(), [52]);
}

#[test]
fn bare_plotting_keeps_its_legacy_thread_local_receipt() {
    reset_recent_figures();
    record_recent_figure(61_u32.into());
    assert_eq!(handles(), [61]);
    assert!(handles().is_empty());
}
