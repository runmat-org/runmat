//! Exact GC-root retirement notifications after admission and scanner locks.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::{GcHandle, RootId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RootRetirement {
    Registration(RootId),
    Explicit(GcHandle),
}

type Observer = dyn Fn(RootRetirement) + Send + Sync;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static OBSERVERS: Mutex<Vec<(u64, Weak<Observer>)>> = Mutex::new(Vec::new());
const MAX_OBSERVERS: usize = 4096;

/// Non-cloneable host lease. Only a weak callback is retained by the GC;
/// cross-thread root retirement may wake a host whose execution lane is idle.
pub struct RootRetirementObserver {
    id: u64,
    _callback: Arc<Observer>,
}

impl RootRetirementObserver {
    pub fn register(callback: Arc<dyn Fn(RootRetirement) + Send + Sync>) -> Option<Self> {
        let mut observers = OBSERVERS.lock();
        observers.retain(|(_, observer)| observer.strong_count() > 0);
        if observers.len() >= MAX_OBSERVERS {
            return None;
        }
        let id = NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .ok()?;
        observers.push((id, Arc::downgrade(&callback)));
        Some(Self {
            id,
            _callback: callback,
        })
    }
}

impl Drop for RootRetirementObserver {
    fn drop(&mut self) {
        OBSERVERS.lock().retain(|(id, _)| *id != self.id);
    }
}

pub(crate) fn notify(event: RootRetirement) {
    // Capture the finite observer identity frontier for this event. An
    // observer registered reentrantly by a callback belongs to a later event.
    let upper = NEXT_ID.load(Ordering::Acquire).saturating_sub(1);
    let mut cursor = 0;
    loop {
        // No GC, observer-list, or root lock is held while a callback runs or
        // while the callback's last strong owner is dropped.
        let next = {
            let observers = OBSERVERS.lock();
            let next = observers.partition_point(|(id, _)| *id <= cursor);
            observers
                .get(next)
                .filter(|(id, _)| *id <= upper)
                .map(|(id, observer)| (*id, observer.upgrade()))
        };
        let Some((id, callback)) = next else {
            break;
        };
        cursor = id;
        if let Some(callback) = callback {
            callback(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn notification_follows_admission_unlock_and_releases_observer_lock() {
        crate::gc_test_context(|| {
            let root = crate::gc_allocate_rooted(crate::Value::Num(1.0)).expect("explicit root");
            let target = (root.handle().addr(), root.handle().epoch());
            let observed = Arc::new(AtomicBool::new(false));
            let nested_observed = Arc::new(AtomicBool::new(false));
            let nested_observer = Arc::new(Mutex::new(None));
            let seen = Arc::clone(&observed);
            let nested_seen = Arc::clone(&nested_observed);
            let nested_owner = Arc::clone(&nested_observer);
            let _observer = RootRetirementObserver::register(Arc::new(move |event| {
                let RootRetirement::Explicit(handle) = event else {
                    return;
                };
                if (handle.addr(), handle.epoch()) != target {
                    return;
                }
                assert!(crate::ROOT_ADMISSION.try_write().is_some());
                if nested_owner.lock().is_none() {
                    let nested_seen = Arc::clone(&nested_seen);
                    let nested = RootRetirementObserver::register(Arc::new(move |_| {
                        nested_seen.store(true, Ordering::Relaxed);
                    }))
                    .expect("nested observer identity");
                    *nested_owner.lock() = Some(nested);
                }
                seen.store(true, Ordering::Relaxed);
            }))
            .expect("observer identity");
            root.unroot().expect("retire explicit root");
            assert!(observed.load(Ordering::Relaxed));
            assert!(!nested_observed.load(Ordering::Relaxed));
            let second = crate::gc_allocate_rooted(crate::Value::Num(2.0)).expect("second root");
            second.unroot().expect("retire second explicit root");
            assert!(nested_observed.load(Ordering::Relaxed));
        });
    }
}
