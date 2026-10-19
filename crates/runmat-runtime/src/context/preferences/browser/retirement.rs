//! Retain an IndexedDB observer subject until its actual terminal event.

use runmat_execution::profiling::{CommandDeferredWork, DeferredWorkTerminal};

pub(super) struct Observer<T> {
    subject: Option<T>,
    deferred: Option<CommandDeferredWork>,
}

impl<T> Observer<T> {
    pub(super) fn new(subject: T, deferred: Option<CommandDeferredWork>) -> Self {
        Self {
            subject: Some(subject),
            deferred,
        }
    }

    pub(super) fn subject(&self) -> &T {
        self.subject.as_ref().expect("observer subject is live")
    }

    pub(super) fn finish(mut self, terminal_observed: bool) {
        drop(self.subject.take());
        if terminal_observed {
            if let Some(deferred) = self.deferred.take() {
                deferred.complete(DeferredWorkTerminal::Completed);
            }
        }
    }
}

impl<T> Drop for Observer<T> {
    fn drop(&mut self) {
        drop(self.subject.take());
        drop(self.deferred.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::profiling::{CommandProfilingRoot, DeferredWorkCounts};
    use std::cell::Cell;
    use std::rc::Rc;
    use wasm_bindgen_test::wasm_bindgen_test;

    struct Subject(Rc<Cell<bool>>);

    impl Drop for Subject {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    #[wasm_bindgen_test]
    fn observer_requires_a_real_terminal_event() {
        let missing = CommandProfilingRoot::new();
        let released = Rc::new(Cell::new(false));
        Observer::new(
            Subject(Rc::clone(&released)),
            missing.receipt().begin_deferred_work(),
        )
        .finish(false);
        assert!(released.get());
        assert_eq!(
            missing.seal().deferred_work,
            Some(DeferredWorkCounts {
                admitted: 1,
                missing: 1,
                ..DeferredWorkCounts::default()
            })
        );

        let completed = CommandProfilingRoot::new();
        let released = Rc::new(Cell::new(false));
        Observer::new(
            Subject(Rc::clone(&released)),
            completed.receipt().begin_deferred_work(),
        )
        .finish(true);
        assert!(released.get());
        assert_eq!(
            completed.seal().deferred_work,
            Some(DeferredWorkCounts {
                admitted: 1,
                completed: 1,
                ..DeferredWorkCounts::default()
            })
        );
    }
}
