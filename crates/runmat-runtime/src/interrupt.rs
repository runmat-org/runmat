use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

thread_local! {
    static INTERRUPT_HANDLE: RefCell<Option<Arc<AtomicBool>>> = RefCell::new(None);
}

pub struct InterruptGuard {
    previous: Option<Arc<AtomicBool>>,
}

impl InterruptGuard {
    pub fn install(handle: Option<Arc<AtomicBool>>) -> Self {
        let previous = INTERRUPT_HANDLE.with(|slot| slot.replace(handle));
        Self { previous }
    }
}

impl Drop for InterruptGuard {
    fn drop(&mut self) {
        INTERRUPT_HANDLE.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

pub fn replace_interrupt(handle: Option<Arc<AtomicBool>>) -> InterruptGuard {
    InterruptGuard::install(handle)
}

pub fn is_cancelled() -> bool {
    INTERRUPT_HANDLE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|flag| flag.load(Ordering::Relaxed))
            .unwrap_or(false)
    })
}
