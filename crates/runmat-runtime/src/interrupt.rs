use runmat_execution::cancellation::{CancellationSource, CancellationToken};
use runmat_execution::CancellationReason;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;

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

/// Why the active execution must stop: a cancelled generation or raised
/// interrupt (`User`), otherwise an elapsed inherited deadline.
pub fn stop_reason() -> Option<CancellationReason> {
    let _ = crate::analysis::fea_run_admission::FeaRunScope::check_current();
    current_interrupt().and_then(|token| token.stop_reason())
}

pub fn is_cancelled() -> bool {
    if crate::analysis::fea_run_admission::FeaRunScope::check_current().is_err() {
        return true;
    }
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
