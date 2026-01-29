use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::sync::Arc;

runmat_thread_local! {
    static CURRENT_SOURCE: RefCell<Option<Arc<str>>> = const { RefCell::new(None) };
}

pub struct SourceContextGuard {
    prev: Option<Arc<str>>,
}

impl Drop for SourceContextGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        CURRENT_SOURCE.with(|slot| {
            *slot.borrow_mut() = prev;
        });
    }
}

/// Replace the current source text for this thread.
///
/// This is used for UX features like "show the original expression" in legends and for
/// diagnostics that need to slice the source by byte-span.
pub fn replace_current_source(source: Option<&str>) -> SourceContextGuard {
    let next = source.map(Arc::<str>::from);
    let prev = CURRENT_SOURCE.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), next));
    SourceContextGuard { prev }
}

pub fn current_source() -> Option<Arc<str>> {
    CURRENT_SOURCE.with(|slot| slot.borrow().clone())
}
