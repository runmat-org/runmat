use runmat_execution::cancellation::{CancellationSource, CancellationToken};
use runmat_execution::CancellationReason;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::sync::{atomic::AtomicBool, Arc};

runmat_thread_local! {
    static INTERRUPT_HANDLE: RefCell<Option<CancellationToken>> = const { RefCell::new(None) };
}

pub struct InterruptGuard {
    previous: Option<CancellationToken>,
    runtime: Option<crate::context::RuntimeContext>,
}

impl InterruptGuard {
    pub fn install(handle: Option<CancellationToken>) -> Self {
        if let Some(runtime) = crate::context::legacy::active() {
            let replacement = handle.unwrap_or_else(|| CancellationSource::new().token());
            let previous = runtime.replace_cancellation_token(replacement);
            Self {
                previous: Some(previous),
                runtime: Some(runtime),
            }
        } else {
            let previous = INTERRUPT_HANDLE.with(|slot| slot.replace(handle));
            Self {
                previous,
                runtime: None,
            }
        }
    }
}

impl Drop for InterruptGuard {
    fn drop(&mut self) {
        if let Some(runtime) = &self.runtime {
            if let Some(previous) = self.previous.take() {
                runtime.replace_cancellation_token(previous);
            }
        } else {
            INTERRUPT_HANDLE.with(|slot| {
                slot.replace(self.previous.take());
            });
        }
    }
}

pub fn replace_interrupt(handle: Option<CancellationToken>) -> InterruptGuard {
    InterruptGuard::install(handle)
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
    let mut token = current_interrupt()
        .unwrap_or_else(|| CancellationToken::from_shared(session.clone()))
        .with_source(session);
    if let Some(parent) = parent {
        token = token.or(&parent);
    }
    if let Some(caller) = caller {
        token = token.with_source(caller);
    }
    InterruptGuard::install(Some(token.with_deadline(deadline)))
}

/// Why the active execution must stop: a cancelled generation or raised
/// interrupt (`User`), otherwise an elapsed inherited deadline.
pub fn stop_reason() -> Option<CancellationReason> {
    current_interrupt().and_then(|token| token.stop_reason())
}

pub fn is_cancelled() -> bool {
    if let Some(runtime) = crate::context::legacy::active() {
        return runtime.is_cancelled();
    }
    INTERRUPT_HANDLE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(CancellationToken::is_cancelled)
            .unwrap_or(false)
    })
}

pub fn current_interrupt() -> Option<CancellationToken> {
    if let Some(runtime) = crate::context::legacy::active() {
        return Some(runtime.cancellation());
    }
    INTERRUPT_HANDLE.with(|slot| slot.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_scope_keeps_all_original_sources_and_restores_parent() {
        let session = Arc::new(AtomicBool::new(false));
        let caller = Arc::new(AtomicBool::new(false));
        let _base = replace_interrupt(Some(CancellationToken::from_shared(session.clone())));
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
