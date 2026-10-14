//! Own a queued blocking preference worker and its deferred-work lease.
//!
//! Tokio may discard a queued closure without polling it. Explicitly retire
//! the work capture before the lease in that case and on unwind.

use runmat_execution::profiling::{CommandDeferredWork, DeferredWorkTerminal};

pub(super) struct BlockingWork<F> {
    work: Option<F>,
    deferred: Option<CommandDeferredWork>,
}

impl<F> BlockingWork<F> {
    pub(super) fn new(work: F, deferred: Option<CommandDeferredWork>) -> Self {
        Self {
            work: Some(work),
            deferred,
        }
    }

    pub(super) fn run<R>(mut self) -> R
    where
        F: FnOnce() -> R,
    {
        let work = self.work.take().expect("blocking work is consumed once");
        let result = work();
        if let Some(deferred) = self.deferred.take() {
            deferred.complete(DeferredWorkTerminal::Completed);
        }
        result
    }
}

impl<F> Drop for BlockingWork<F> {
    fn drop(&mut self) {
        drop(self.work.take());
        drop(self.deferred.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::profiling::{CommandProfilingRoot, DeferredWorkCounts};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    struct Capture(Arc<AtomicBool>);

    impl Drop for Capture {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn queued_unrun_work_retires_capture_then_records_missing() {
        let command = CommandProfilingRoot::new();
        let retired = Arc::new(AtomicBool::new(false));
        let capture = Capture(Arc::clone(&retired));
        let work = BlockingWork::new(
            move || drop(capture),
            command.receipt().begin_deferred_work(),
        );
        drop(work);
        assert!(retired.load(Ordering::SeqCst));
        assert_eq!(
            command.seal().deferred_work,
            Some(DeferredWorkCounts {
                admitted: 1,
                missing: 1,
                ..DeferredWorkCounts::default()
            })
        );
    }

    #[test]
    fn completed_attempt_transfers_returned_owner_to_caller() {
        let command = CommandProfilingRoot::new();
        let retired = Arc::new(AtomicBool::new(false));
        let capture = Capture(Arc::clone(&retired));
        let returned =
            BlockingWork::new(move || capture, command.receipt().begin_deferred_work()).run();
        assert!(!retired.load(Ordering::SeqCst));
        assert_eq!(
            command.seal().deferred_work,
            Some(DeferredWorkCounts {
                admitted: 1,
                completed: 1,
                ..DeferredWorkCounts::default()
            })
        );
        drop(returned);
        assert!(retired.load(Ordering::SeqCst));
    }

    #[test]
    fn unwound_worker_records_missing() {
        let command = CommandProfilingRoot::new();
        let retired = Arc::new(AtomicBool::new(false));
        let capture = Capture(Arc::clone(&retired));
        let work = BlockingWork::new(
            move || {
                let _capture = capture;
                panic!("worker panic");
            },
            command.receipt().begin_deferred_work(),
        );
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work.run())).is_err());
        assert!(retired.load(Ordering::SeqCst));
        assert_eq!(
            command.seal().deferred_work,
            Some(DeferredWorkCounts {
                admitted: 1,
                missing: 1,
                ..DeferredWorkCounts::default()
            })
        );
    }
}
