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
    // No GC, observer-list, or root lock is held while a Waker is invoked.
    let callbacks = OBSERVERS
        .lock()
        .iter()
        .filter_map(|(_, observer)| observer.upgrade())
        .collect::<Vec<_>>();
    for callback in callbacks {
        callback(event);
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
            let seen = Arc::clone(&observed);
            let _observer = RootRetirementObserver::register(Arc::new(move |event| {
                let RootRetirement::Explicit(handle) = event else {
                    return;
                };
                if (handle.addr(), handle.epoch()) != target {
                    return;
                }
                assert!(crate::ROOT_ADMISSION.try_write().is_some());
                let _nested = RootRetirementObserver::register(Arc::new(|_| {}))
                    .expect("nested observer identity");
                seen.store(true, Ordering::Relaxed);
            }))
            .expect("observer identity");
            root.unroot().expect("retire explicit root");
            assert!(observed.load(Ordering::Relaxed));
        });
    }
}
