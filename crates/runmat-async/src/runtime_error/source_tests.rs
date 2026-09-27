use super::*;

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
