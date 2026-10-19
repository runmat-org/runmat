use super::*;
use runmat_value::{ExactHostAllocationReservation, ExactHostNumericAllocationError, Value};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug)]
struct TestPermit {
    live: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct TestClaim {
    bytes: usize,
    live: Arc<AtomicUsize>,
}

impl Drop for TestClaim {
    fn drop(&mut self) {
        self.live.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

impl ExactHostAllocationReservation for TestClaim {}

impl ExactHostAllocationPermit for TestPermit {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        self.live.fetch_add(bytes, Ordering::SeqCst);
        Ok(Arc::new(TestClaim {
            bytes,
            live: Arc::clone(&self.live),
        }))
    }
}

struct Handles(Vec<GcHandle>);
impl Trace for Handles {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for handle in &self.0 {
            tracer.mark(*handle);
        }
    }
}

struct ChangingTrace {
    handle: GcHandle,
    calls: Cell<usize>,
}
impl Trace for ChangingTrace {
    fn trace(&self, tracer: &mut dyn Tracer) {
        let call = self.calls.get();
        self.calls.set(call + 1);
        tracer.mark(self.handle);
        if call != 0 {
            tracer.mark(self.handle);
        }
    }
}

#[test]
fn prepared_lease_is_atomic_deduplicated_and_shared() {
    crate::gc_test_context(|| {
        let source_root = crate::gc_allocate_rooted(Value::Num(7.0)).expect("source root");
        let handle = source_root.handle();
        let source = Handles(vec![handle, handle]);
        let plan = PreparedRootPlan::for_traced(&source).expect("root plan");
        assert!(plan.quote().retained_claims >= 1);
        let live = Arc::new(AtomicUsize::new(0));
        let permit = TestPermit {
            live: Arc::clone(&live),
        };
        let lease = plan.prepare(&source, &permit).expect("prepared roots");
        assert_eq!(lease.handles(), &[handle]);
        assert_eq!(crate::GC.root_ptrs.lock().get(&handle), Some(&2));
        let alias = lease.clone();
        drop(lease);
        assert_eq!(crate::GC.root_ptrs.lock().get(&handle), Some(&2));
        drop(alias);
        assert_eq!(crate::GC.root_ptrs.lock().get(&handle), Some(&1));
        drop(source_root);
        assert!(!crate::GC.root_ptrs.lock().contains_key(&handle));
        assert_eq!(live.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn changed_second_trace_fails_before_registry_publication() {
    crate::gc_test_context(|| {
        let source_root = crate::gc_allocate_rooted(Value::Num(11.0)).expect("source root");
        let handle = source_root.handle();
        let source = ChangingTrace {
            handle,
            calls: Cell::new(0),
        };
        let plan = PreparedRootPlan::for_traced(&source).expect("root plan");
        let live = Arc::new(AtomicUsize::new(0));
        let permit = TestPermit {
            live: Arc::clone(&live),
        };
        assert_eq!(
            plan.prepare(&source, &permit).unwrap_err(),
            RootAdmissionError::TraceChanged
        );
        assert_eq!(crate::GC.root_ptrs.lock().get(&handle), Some(&1));
        assert_eq!(live.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn source_failure_preserves_its_type_and_rolls_back_before_root_publication() {
    #[derive(Debug, PartialEq, Eq)]
    enum SourceFailure {
        ScratchDenied,
    }

    crate::gc_test_context(|| {
        let source_root = crate::gc_allocate_rooted(Value::Num(17.0)).expect("source root");
        let handle = source_root.handle();
        let failed_plan = PreparedRootPlan::try_for_traced(|tracer| {
            tracer.mark(handle);
            Err(SourceFailure::ScratchDenied)
        });
        assert_eq!(
            failed_plan.unwrap_err(),
            RootTraceFailure::Source(SourceFailure::ScratchDenied)
        );
        let source = Handles(vec![handle]);
        let plan = PreparedRootPlan::for_traced(&source).expect("root plan");
        let live = Arc::new(AtomicUsize::new(0));
        let permit = TestPermit {
            live: Arc::clone(&live),
        };
        let failed_lease = plan.try_prepare(
            |tracer| {
                tracer.mark(handle);
                Err(SourceFailure::ScratchDenied)
            },
            &permit,
        );
        assert_eq!(
            failed_lease.unwrap_err(),
            RootTraceFailure::Source(SourceFailure::ScratchDenied)
        );
        assert_eq!(crate::GC.root_ptrs.lock().get(&handle), Some(&1));
        assert_eq!(live.load(Ordering::SeqCst), 0);
        // The failed traversal did not change the registry generation either.
        let lease = plan
            .prepare(&source, &permit)
            .expect("same plan remains live");
        drop(lease);
        assert_eq!(crate::GC.root_ptrs.lock().get(&handle), Some(&1));
        assert_eq!(live.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn intervening_root_mutation_stales_the_prepared_generation() {
    crate::gc_test_context(|| {
        let source_root = crate::gc_allocate_rooted(Value::Num(13.0)).expect("source root");
        let handle = source_root.handle();
        let source = Handles(vec![handle]);
        let plan = PreparedRootPlan::for_traced(&source).expect("root plan");
        let empty = Handles(Vec::new());
        let empty_plan = PreparedRootPlan::for_traced(&empty).expect("empty root plan");
        crate::gc_add_root(handle).expect("intervening compatibility root");
        assert!(empty_plan
            .prepare_unbounded(&empty)
            .expect("empty plan has no registry dependency")
            .handles()
            .is_empty());
        let live = Arc::new(AtomicUsize::new(0));
        let permit = TestPermit {
            live: Arc::clone(&live),
        };
        assert_eq!(
            plan.prepare(&source, &permit).unwrap_err(),
            RootAdmissionError::StalePlan
        );
        assert_eq!(crate::GC.root_ptrs.lock().get(&handle), Some(&2));
        crate::gc_remove_root(handle).expect("remove intervening root");
        assert_eq!(live.load(Ordering::SeqCst), 0);
    });
}
