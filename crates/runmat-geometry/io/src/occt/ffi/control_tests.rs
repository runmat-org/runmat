use super::*;
use runmat_execution::CancellationToken;
use std::sync::{atomic::AtomicBool, Arc};

#[test]
fn native_registry_keeps_original_source_and_deadline() {
    let caller = Arc::new(AtomicBool::new(false));
    let token = CancellationToken::default().with_source(caller.clone());
    let registration = OcctCancelTokenRegistration::new(Some(token));
    assert_ne!(registration.id(), 0);
    assert!(!occt_import_cancelled(registration.id()));
    caller.store(true, Ordering::Release);
    assert!(occt_import_cancelled(registration.id()));
    let id = registration.id();
    drop(registration);
    assert!(!occt_import_cancelled(id));
    let deadline =
        OcctCancelTokenRegistration::new(Some(CancellationToken::default().with_deadline(Some(0))));
    assert!(occt_import_cancelled(deadline.id()));
}

#[test]
fn poison_recovery_preserves_original_registry_without_disabling_control() {
    let registry = Arc::new(Mutex::new(CancelTokenMap::new()));
    let token = CancellationToken::default().with_deadline(Some(0));
    let poisoned = registry.clone();
    assert!(std::thread::spawn(move || {
        let mut guard = poisoned.lock().unwrap();
        guard.insert(123, token);
        panic!("intentional registry-owner unwind");
    })
    .join()
    .is_err());
    assert!(registry.is_poisoned());
    assert!(lock_cancel_tokens(&registry)
        .get(&123)
        .unwrap()
        .is_cancelled());
    assert!(!registry.is_poisoned());
}
