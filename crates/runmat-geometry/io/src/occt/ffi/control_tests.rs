use super::*;
use std::sync::{atomic::AtomicBool, Arc};

#[test]
fn native_registry_follows_its_caller_probe() {
    let caller = Arc::new(AtomicBool::new(false));
    let flag = caller.clone();
    let registration =
        OcctCancelTokenRegistration::new(Some(Arc::new(move || flag.load(Ordering::Acquire))));
    assert_ne!(registration.id(), 0);
    assert!(!occt_import_cancelled(registration.id()));
    caller.store(true, Ordering::Release);
    assert!(occt_import_cancelled(registration.id()));
}

#[test]
fn poison_recovery_preserves_original_registry_without_disabling_control() {
    let registry = Arc::new(Mutex::new(CancelTokenMap::new()));
    let poisoned = registry.clone();
    assert!(std::thread::spawn(move || {
        let mut guard = poisoned.lock().unwrap();
        guard.insert(123, Arc::new(|| true));
        panic!("intentional registry-owner unwind");
    })
    .join()
    .is_err());
    assert!(registry.is_poisoned());
    let probe = lock_cancel_tokens(&registry).get(&123).cloned().unwrap();
    assert!(probe());
    assert!(!registry.is_poisoned());
}
