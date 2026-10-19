use super::*;

#[test]
fn retained_allocation_survives_original_error_in_a_nested_source_chain() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct Lease(Arc<AtomicUsize>);
    impl Drop for Lease {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    let original = runtime_error("full original message")
        .with_identifier("RunMat:Original")
        .with_span(miette::SourceSpan::new(5.into(), 3))
        .with_call_frames(vec![CallFrame {
            function: "origin".into(),
            source_id: Some(7),
            span: Some((5, 8)),
            source_owner: None,
            diagnostic_source: None,
        }])
        .with_source(std::io::Error::other("typed original cause"))
        .with_retained_diagnostic_allocation(RetainedDiagnosticAllocation::Owned(Box::new(Lease(
            drops.clone(),
        ))))
        .build();
    let nested = runtime_error("outer").with_source(original).build();
    let inner = StdError::source(&nested)
        .unwrap()
        .downcast_ref::<RuntimeError>()
        .expect("original runtime error remains downcastable");
    assert_eq!(inner.identifier(), Some("RunMat:Original"));
    assert_eq!(inner.message(), "full original message");
    assert_eq!(inner.span.as_ref().map(miette::SourceSpan::offset), Some(5));
    assert_eq!(inner.context.call_frames[0].function, "origin");
    assert_eq!(inner.context.call_frames[0].source_id, Some(7));
    assert!(StdError::source(inner).unwrap().is::<std::io::Error>());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(nested);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn call_frame_source_owners_compare_issuance_not_empty_payload() {
    let issued = CallFrameSourceOwner::issue();
    assert_eq!(issued, issued.clone());
    assert_ne!(issued, CallFrameSourceOwner::issue());
}

#[test]
fn shared_retention_does_not_change_source_or_release_before_last_error() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct Lease(Arc<AtomicUsize>);
    impl Drop for Lease {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    let lease: Arc<dyn DiagnosticAllocationLease> = Arc::new(Lease(drops.clone()));
    let error = runtime_error("kept")
        .with_retained_diagnostic_allocation(RetainedDiagnosticAllocation::Shared(lease.clone()))
        .build();
    assert!(StdError::source(&error).is_none());
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(lease);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn rethrow_with_external_original_keeps_its_charge_only_on_the_original_owner() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct Lease(Arc<AtomicUsize>);
    impl Drop for Lease {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    let original = RuntimeErrorOwner::new(
        runtime_error("original full diagnostic")
            .with_identifier("RunMat:Original")
            .with_builtin("readmatrix")
            .with_source(std::io::Error::other("typed provider cause"))
            .with_retained_diagnostic_allocation(RetainedDiagnosticAllocation::Owned(Box::new(
                Lease(drops.clone()),
            )))
            .build(),
    );
    let outward = RuntimeError::from_shared_original(original.clone());
    assert_eq!(outward.message(), original.message());
    assert_eq!(outward.identifier(), original.identifier());
    assert_eq!(outward.context.builtin, original.context.builtin);
    assert!(outward.retained_diagnostic_allocation.is_none());
    let shared = StdError::source(&outward)
        .unwrap()
        .downcast_ref::<SharedRuntimeErrorSource>()
        .expect("outward error owns the exact original, not a formatted source");
    assert!(std::ptr::eq(shared.original(), original.as_ref()));
    assert!(StdError::source(shared.original())
        .unwrap()
        .is::<std::io::Error>());
    drop(original);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(outward);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn rethrow_with_unique_original_moves_its_charge_without_a_metadata_copy() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct Lease(Arc<AtomicUsize>);
    impl Drop for Lease {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    let original = RuntimeErrorOwner::new(
        runtime_error("unique full diagnostic")
            .with_identifier("RunMat:Unique")
            .with_retained_diagnostic_allocation(RetainedDiagnosticAllocation::Owned(Box::new(
                Lease(drops.clone()),
            )))
            .build(),
    );
    let original_message = original.message.as_ptr();
    let outward = RuntimeError::from_shared_original(original);
    assert_eq!(outward.message.as_ptr(), original_message);
    assert_eq!(outward.identifier(), Some("RunMat:Unique"));
    assert!(outward.source.is_none());
    assert!(outward.retained_diagnostic_allocation.is_some());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(outward);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_sources_remain_send_sync_and_owned_through_formatting_and_nested_errors() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RuntimeError>();
    #[derive(Debug, thiserror::Error)]
    #[error("retained native owner")]
    struct Retained(Arc<AtomicUsize>);
    impl Drop for Retained {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let error = runtime_error("inner")
        .with_source(Retained(drops.clone()))
        .build();
    let error = runtime_error("outer")
        .with_source(error)
        .with_identifier("RunMat:test")
        .build();
    assert!(error.format_diagnostic().contains("RunMat:test"));
    let inner = StdError::source(&error)
        .unwrap()
        .downcast_ref::<RuntimeError>()
        .unwrap();
    assert!(StdError::source(inner).unwrap().is::<Retained>());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[cfg(target_arch = "wasm32")]
#[test]
fn browser_sources_retain_non_send_owners_through_diagnostic_borrow_and_drop() {
    use std::{cell::Cell, rc::Rc};
    #[derive(Debug, thiserror::Error)]
    #[error("retained browser owner")]
    struct Retained(Rc<Cell<usize>>);
    impl Drop for Retained {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let drops = Rc::new(Cell::new(0));
    let error = runtime_error("inner")
        .with_source(Retained(drops.clone()))
        .build();
    let error = runtime_error("outer").with_source(error).build();
    assert!(error.format_diagnostic().contains("outer"));
    let inner = StdError::source(&error)
        .unwrap()
        .downcast_ref::<RuntimeError>()
        .unwrap();
    assert!(StdError::source(inner).unwrap().is::<Retained>());
    assert_eq!(drops.get(), 0);
    drop(error);
    assert_eq!(drops.get(), 1);
}
