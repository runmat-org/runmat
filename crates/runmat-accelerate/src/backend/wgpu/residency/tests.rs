use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[test]
fn symbolic_extent_is_the_same_nonzero_size_used_by_exact_pool_keys() {
    assert_eq!(BufferResidency::requested_bytes(40, 4), Some(160));
    assert_eq!(BufferResidency::requested_bytes(40, 8), Some(320));
    assert_eq!(BufferResidency::requested_bytes(0, 4), Some(4));
    assert_eq!(BufferResidency::requested_bytes(0, 8), Some(8));
    assert_eq!(BufferResidency::requested_bytes(1, 0), None);
    #[cfg(target_pointer_width = "64")]
    assert_eq!(BufferResidency::requested_bytes(usize::MAX, 8), None);
}

#[test]
fn pool_retains_exact_usage_and_bytes_and_returns_rejected_payload() {
    let pools = Mutex::new(HashMap::new());
    let generic = ResidencyKey::new(BufferUsageClass::Generic, 16);
    let matrix = ResidencyKey::new(BufferUsageClass::MatmulOut, 16);
    assert_eq!(release_into(&pools, 1, generic, 10), Ok(()));
    assert_eq!(
        release_into(&pools, 1, generic, 20),
        Err((PoolRejectionReason::Full, 20))
    );
    assert_eq!(release_into(&pools, 1, matrix, 30), Ok(()));
    assert_eq!(
        release_into(&pools, 0, matrix, 40),
        Err((PoolRejectionReason::Full, 40))
    );
    assert_eq!(
        release_into(
            &pools,
            1,
            ResidencyKey::new(BufferUsageClass::Generic, 0),
            50
        ),
        Err((PoolRejectionReason::Empty, 50))
    );
    let pools = pools.lock().unwrap();
    assert_eq!(pools.len(), 2);
    assert_eq!(pools[&generic], VecDeque::from([10]));
    assert_eq!(pools[&matrix], VecDeque::from([30]));
}

#[cfg(panic = "unwind")]
#[test]
fn poisoned_pool_returns_buffer_without_claiming_device_failure() {
    let pools = Mutex::new(HashMap::new());
    let key = ResidencyKey::new(BufferUsageClass::Generic, 16);
    assert_eq!(release_into(&pools, 2, key, 10), Ok(()));
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = pools.lock().unwrap();
        panic!("poison optional reuse pool");
    }))
    .is_err());
    assert_eq!(
        release_into(&pools, 2, key, 20),
        Err((PoolRejectionReason::Poisoned, 20))
    );
    assert_eq!(
        pools.lock().unwrap_err().into_inner()[&key],
        VecDeque::from([10])
    );
}

struct ReentrantDrop(Box<dyn Fn() + Send + Sync>);
impl Drop for ReentrantDrop {
    fn drop(&mut self) {
        (self.0)();
    }
}

#[test]
fn rejected_and_drained_buffers_drop_outside_pool_lock() {
    let pools = Arc::new(Mutex::new(HashMap::new()));
    let drops = Arc::new(AtomicUsize::new(0));
    let resource = || {
        let pools = Arc::downgrade(&pools);
        let drops = drops.clone();
        ReentrantDrop(Box::new(move || {
            let pools = pools.upgrade().unwrap();
            assert!(
                pools.try_lock().is_ok(),
                "resource destructor reentered pool lock"
            );
            drops.fetch_add(1, Ordering::Relaxed);
        }))
    };
    let key = ResidencyKey::new(BufferUsageClass::Generic, 16);
    assert!(release_into(&pools, 1, key, resource()).is_ok());
    let rejected = release_into(&pools, 1, key, resource());
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    let Err((reason, buffer)) = rejected else {
        panic!("full pool must return buffer");
    };
    assert_eq!(reason, PoolRejectionReason::Full);
    drop(buffer);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    let drained = pools.lock().unwrap().remove(&key);
    drop(drained);
    assert_eq!(drops.load(Ordering::Relaxed), 2);
}
