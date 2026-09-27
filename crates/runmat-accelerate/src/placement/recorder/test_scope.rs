//! Thread-local recorder isolation for synchronous unit-test execution.
//! Trace contexts must be completed/dropped inside the scope that created them.
use super::*;
use std::{cell::RefCell, rc::Rc, sync::Arc};

thread_local! {
    static CURRENT: RefCell<Option<Arc<Mutex<RecorderState>>>> = const { RefCell::new(None) };
}

pub(super) fn current() -> Option<Arc<Mutex<RecorderState>>> {
    CURRENT.with(|current| current.borrow().clone())
}

pub(crate) struct ScopedRecorder {
    previous: Option<Arc<Mutex<RecorderState>>>,
    _thread_affinity: std::marker::PhantomData<Rc<()>>,
}

impl ScopedRecorder {
    pub(crate) fn new() -> Self {
        let recorder = Arc::new(Mutex::new(RecorderState::default()));
        Self {
            previous: CURRENT.with(|current| current.replace(Some(recorder))),
            _thread_affinity: std::marker::PhantomData,
        }
    }
}

impl Drop for ScopedRecorder {
    fn drop(&mut self) {
        CURRENT.with(|current| current.replace(self.previous.take()));
    }
}

#[test]
fn scoped_recorder_preserves_its_active_trace_across_foreign_reset_and_nested_scope() {
    let _scope = ScopedRecorder::new();
    let trace = begin_trace("owned", None);
    let correlation = trace.correlation();
    std::thread::spawn(|| {
        reset();
        begin_trace("foreign", None).complete(None, None);
        reset();
    })
    .join()
    .unwrap();
    {
        let _nested = ScopedRecorder::new();
        begin_trace("nested", None).complete(None, None);
        assert_eq!(report().traces[0].operation, "nested");
    }
    trace.fallback(None, "provider_operation_unsupported", None);
    let report = report();
    assert_eq!(report.traces.len(), 1);
    let trace = &report.traces[0];
    assert_eq!(trace.correlation, correlation);
    assert_eq!(trace.operation, "owned");
    assert!(trace.complete);
    assert_eq!(trace.events.len(), 1);
    assert_eq!(trace.events[0].kind, PlacementEventKind::Fallback);
    assert_eq!(
        trace.events[0].reason.as_deref(),
        Some("provider_operation_unsupported")
    );
}
