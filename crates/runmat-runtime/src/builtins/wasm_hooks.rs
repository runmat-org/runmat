#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;

#[cfg(target_arch = "wasm32")]
std::thread_local! {
    static REGISTRATIONS: RefCell<Vec<Box<dyn FnOnce()>>> = RefCell::new(Vec::new());
}

#[cfg(target_arch = "wasm32")]
pub fn register(f: Box<dyn FnOnce()>) {
    REGISTRATIONS.with(|queue| queue.borrow_mut().push(f));
}

#[cfg(target_arch = "wasm32")]
pub fn flush() {
    REGISTRATIONS.with(|queue| {
        for f in queue.borrow_mut().drain(..) {
            f();
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub fn register(_f: Box<dyn FnOnce()>) {}

#[cfg(not(target_arch = "wasm32"))]
pub fn flush() {}

