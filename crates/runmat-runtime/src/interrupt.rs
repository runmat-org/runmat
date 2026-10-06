use runmat_execution::{CancellationReason, CancellationToken};
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::sync::{atomic::AtomicBool, Arc};

runmat_thread_local! {
    static INTERRUPT_HANDLE: RefCell<Option<CancellationToken>> = const { RefCell::new(None) };
}

pub struct InterruptGuard {
    previous: Option<CancellationToken>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl InterruptGuard {
    fn install(token: Option<CancellationToken>) -> Self {
        if let Some(state) = active_state() {
            let previous = state.cancellation.replace(token.unwrap_or_default());
            Self {
                previous: Some(previous),
                state: Some(state),
            }
        } else {
            let previous = INTERRUPT_HANDLE.with(|slot| slot.replace(token));
            Self {
                previous,
                state: None,
            }
        }
    }
}

impl Drop for InterruptGuard {
    fn drop(&mut self) {
        if let Some(state) = &self.state {
            if let Some(previous) = self.previous.take() {
                state.cancellation.replace(previous);
            }
        } else {
            INTERRUPT_HANDLE.with(|slot| {
                slot.replace(self.previous.take());
            });
        }
    }
}

pub fn replace_interrupt(handle: Option<Arc<AtomicBool>>) -> InterruptGuard {
    InterruptGuard::install(handle.map(CancellationToken::from_shared))
}

/// Extend the active tree; keep session/ancestor sources and earliest deadline.
pub fn scope_invocation(
    session: Arc<AtomicBool>,
    caller: Option<Arc<AtomicBool>>,
    deadline: Option<u64>,
) -> InterruptGuard {
    scope_invocation_with_parent(session, caller, deadline, None)
}

pub fn scope_invocation_with_parent(
    session: Arc<AtomicBool>,
    caller: Option<Arc<AtomicBool>>,
    deadline: Option<u64>,
    parent: Option<CancellationToken>,
) -> InterruptGuard {
    let mut token = current_cancellation()
        .unwrap_or_else(|| CancellationToken::from_shared(session.clone()))
        .with_source(session);
    if let Some(parent) = parent {
        token = token.with_parent(&parent);
    }
    if let Some(caller) = caller {
        token = token.with_source(caller);
    }
    InterruptGuard::install(Some(token.with_deadline(deadline)))
}

pub fn stop_reason() -> Option<CancellationReason> {
    current_cancellation().and_then(|token| token.stop_reason())
}

pub fn is_cancelled() -> bool {
    stop_reason().is_some()
}

pub fn current_cancellation() -> Option<CancellationToken> {
    if let Some(state) = active_state() {
        return Some(state.cancellation.borrow().clone());
    }
    INTERRUPT_HANDLE.with(|slot| slot.borrow().clone())
}

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_scope_keeps_all_original_sources_and_restores_parent() {
        let session = Arc::new(AtomicBool::new(false));
        let caller = Arc::new(AtomicBool::new(false));
        let _base = replace_interrupt(Some(session.clone()));
        {
            let _outer = scope_invocation(session.clone(), Some(caller.clone()), Some(0));
            assert_eq!(stop_reason(), Some(CancellationReason::Deadline));
            {
                let _inner = scope_invocation(session.clone(), None, Some(u64::MAX));
                assert_eq!(stop_reason(), Some(CancellationReason::Deadline));
                caller.store(true, std::sync::atomic::Ordering::Release);
                assert_eq!(stop_reason(), Some(CancellationReason::User));
            }
            assert_eq!(stop_reason(), Some(CancellationReason::User));
        }
        assert_eq!(stop_reason(), None);
        session.store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(stop_reason(), Some(CancellationReason::User));
    }
}
