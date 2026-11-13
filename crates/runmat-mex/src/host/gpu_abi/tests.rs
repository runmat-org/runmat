use super::*;
use crate::MxApiMode;
use runmat_accelerate_api::ThreadProviderGuard;
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Wake, Waker};

mod provider;
mod exports;
use provider::{Fault, Provider};

struct NoopWake;
impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}
fn waker() -> Waker {
    Waker::from(Arc::new(NoopWake))
}

#[test]
fn native_context_abi_exit_uses_entered_provider_after_ambient_switch() {
    let entered = Provider::new(Fault::None, Fault::None, Fault::None);
    let other = Provider::new(Fault::None, Fault::None, Fault::None);
    let state = MexCallState::new(MxApiMode::InterleavedComplex);
    let host = (&state as *const MexCallState).cast_mut().cast();
    let _entered = ThreadProviderGuard::set(Some(entered));
    // SAFETY: the state remains in place; the returned opaque context is consumed once.
    let context = unsafe { context_enter(host) };
    assert!(!context.is_null());
    let _other = ThreadProviderGuard::set(Some(other));
    assert_eq!(unsafe { context_leave(host, context) }, 0);
    assert!(state.lock().unwrap().error.is_none());
    assert_eq!(entered.events(), vec!["enter", "synchronize", "leave"]);
    assert!(other.events().is_empty());
}

#[test]
fn native_context_abi_combines_sync_and_exit_errors_and_preserves_status() {
    let entered = Provider::new(Fault::None, Fault::Error, Fault::Error);
    let state = MexCallState::new(MxApiMode::InterleavedComplex);
    let host = (&state as *const MexCallState).cast_mut().cast();
    let _selected = ThreadProviderGuard::set(Some(entered));
    let context = unsafe { context_enter(host) };
    assert!(!context.is_null());
    assert_eq!(unsafe { context_leave(host, context) }, 1);
    assert_eq!(
        state.lock().unwrap().error.as_ref().unwrap().message,
        "CUDA synchronization failed: sync failure; context exit failed: exit failure"
    );
    assert_eq!(entered.events(), vec!["enter", "synchronize", "leave"]);
    assert_eq!(unsafe { context_leave(host, std::ptr::null_mut()) }, 0);
    assert_eq!(entered.events(), vec!["enter", "synchronize", "leave"]);
}

#[test]
fn native_context_explicit_exit_reports_each_failure_without_skipping_leave() {
    for (sync, leave, expected) in [
        (
            Fault::Error,
            Fault::None,
            "CUDA synchronization failed: sync failure",
        ),
        (
            Fault::None,
            Fault::Error,
            "CUDA context exit failed: exit failure",
        ),
    ] {
        let provider = Provider::new(Fault::None, sync, leave);
        let context = NativeInvocationContext::enter(provider).unwrap();
        assert_eq!(context.leave().unwrap_err(), expected);
        assert_eq!(provider.events(), vec!["enter", "synchronize", "leave"]);
    }
}

#[test]
fn native_context_cancellation_completion_releases_in_owner_order() {
    let provider = Provider::new(Fault::None, Fault::None, Fault::None);
    let source = runmat_execution::cancellation::CancellationSource::new();
    let token = source.token();
    let mut future = Box::pin(async move {
        let _context = NativeInvocationContext::enter(provider).unwrap();
        token.cancelled().await
    });
    let waker = waker();
    let mut task = Context::from_waker(&waker);
    assert!(future.as_mut().poll(&mut task).is_pending());
    assert_eq!(provider.events(), vec!["enter"]);
    source.cancel(runmat_execution::state::CancellationReason::User);
    assert_eq!(
        future.as_mut().poll(&mut task),
        std::task::Poll::Ready(runmat_execution::state::CancellationReason::User)
    );
    drop(future);
    assert_eq!(provider.events(), vec!["enter", "synchronize", "leave"]);
}

#[test]
fn native_context_polled_abandonment_releases_once_and_unpolled_future_enters_nothing() {
    let provider = Provider::new(Fault::None, Fault::None, Fault::None);
    let pending = || async move {
        let _context = NativeInvocationContext::enter(provider).unwrap();
        std::future::pending::<()>().await;
    };
    drop(pending());
    assert!(provider.events().is_empty());
    let mut future = Box::pin(pending());
    let waker = waker();
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    drop(future);
    assert_eq!(provider.events(), vec!["enter", "synchronize", "leave"]);
}

#[cfg(panic = "unwind")]
#[test]
fn native_context_abi_contains_provider_entry_sync_and_exit_panics() {
    for (enter, sync, leave) in [
        (Fault::Panic, Fault::None, Fault::None),
        (Fault::None, Fault::Panic, Fault::Panic),
    ] {
        let provider = Provider::new(enter, sync, leave);
        let state = MexCallState::new(MxApiMode::InterleavedComplex);
        let host = (&state as *const MexCallState).cast_mut().cast();
        let _selected = ThreadProviderGuard::set(Some(provider));
        let context = unsafe { context_enter(host) };
        if context.is_null() {
            assert_eq!(provider.events(), vec!["enter"]);
            assert_eq!(
                state.lock().unwrap().error.as_ref().unwrap().message,
                "CUDA context entry panicked"
            );
        } else {
            assert_eq!(unsafe { context_leave(host, context) }, 1);
            let message = state
                .lock()
                .unwrap()
                .error
                .as_ref()
                .unwrap()
                .message
                .clone();
            assert!(
                message.contains("synchronization panicked")
                    && message.contains("context exit panicked")
            );
            assert_eq!(provider.events(), vec!["enter", "synchronize", "leave"]);
        }
    }
}

#[cfg(panic = "unwind")]
#[test]
fn native_context_drop_contains_cleanup_panics_during_existing_unwind() {
    let provider = Provider::new(Fault::None, Fault::Panic, Fault::Panic);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _context = NativeInvocationContext::enter(provider).unwrap();
        panic!("invocation unwind");
    }));
    assert!(result.is_err());
    assert_eq!(provider.events(), vec!["enter", "synchronize", "leave"]);
}
