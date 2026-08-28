use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

runmat_thread_local! {
    static INTERRUPT_HANDLE: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

pub struct InterruptGuard {
    previous: Option<Arc<AtomicBool>>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl InterruptGuard {
    pub fn install(handle: Option<Arc<AtomicBool>>) -> Self {
        if let Some(state) = active_state() {
            let replacement = handle.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
            let previous = state.cancellation.replace(replacement);
            Self {
                previous: Some(previous),
                state: Some(state),
            }
        } else {
            let previous = INTERRUPT_HANDLE.with(|slot| slot.replace(handle));
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
    InterruptGuard::install(handle)
}

pub fn is_cancelled() -> bool {
    if let Some(state) = active_state() {
        return state.is_cancelled();
    }
    INTERRUPT_HANDLE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|flag| flag.load(Ordering::Relaxed))
            .unwrap_or(false)
    })
}

pub fn current_interrupt() -> Option<Arc<AtomicBool>> {
    if let Some(state) = active_state() {
        return Some(Arc::clone(&state.cancellation.borrow()));
    }
    INTERRUPT_HANDLE.with(|slot| slot.borrow().clone())
}

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}
