//! One test allocation map, mirroring the required existing-provider authority.
use crate::allocation::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
type Callback = Box<dyn FnOnce() + Send>;
#[cfg(target_arch = "wasm32")]
type Callback = Box<dyn FnOnce()>;

pub(super) enum Entry {
    Unregistered,
    Live(WeakAllocationLease),
    Retiring,
    Finished(AllocationRetirementOutcome),
}

pub(super) struct TestOwner {
    device: u32,
    pub(super) entries: Mutex<HashMap<AllocationIdentity, Entry>>,
    pub(super) retire_calls: AtomicUsize,
    pub(super) panic_during_retire: AtomicBool,
    outcome: AllocationRetirementOutcome,
    callback: Mutex<Option<Callback>>,
}

impl TestOwner {
    pub(super) fn new(id: AllocationIdentity, outcome: AllocationRetirementOutcome) -> Arc<Self> {
        Arc::new(Self {
            device: id.device_id(),
            entries: Mutex::new(HashMap::from([(id, Entry::Unregistered)])),
            retire_calls: AtomicUsize::new(0),
            panic_during_retire: AtomicBool::new(false),
            outcome,
            callback: Mutex::new(None),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn set_callback(&self, callback: impl FnOnce() + Send + 'static) {
        *self.callback.lock().unwrap() = Some(Box::new(callback));
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn set_callback(&self, callback: impl FnOnce() + 'static) {
        *self.callback.lock().unwrap() = Some(Box::new(callback));
    }

    pub(super) fn weak_lease(&self, id: AllocationIdentity) -> WeakAllocationLease {
        let entries = self.entries.lock().unwrap();
        let Some(Entry::Live(weak)) = entries.get(&id) else {
            panic!("missing live lease")
        };
        weak.clone()
    }

    pub(super) fn outcome(&self, id: AllocationIdentity) -> AllocationRetirementOutcome {
        let entries = self.entries.lock().unwrap();
        let Some(Entry::Finished(outcome)) = entries.get(&id) else {
            panic!("missing retirement record")
        };
        outcome.clone()
    }
}

impl AllocationOwner for TestOwner {
    fn register_lease(
        &self,
        id: AllocationIdentity,
        lease: WeakAllocationLease,
    ) -> Result<(), AllocationRegistrationError> {
        if id.device_id() != self.device {
            return Err(AllocationRegistrationError::WrongOwner);
        }
        let mut entries = self.entries.lock().unwrap();
        if !entries.contains_key(&id) {
            return Err(
                if entries
                    .keys()
                    .any(|known| known.allocation_id() == id.allocation_id())
                {
                    AllocationRegistrationError::StaleGeneration
                } else {
                    AllocationRegistrationError::UnknownAllocation
                },
            );
        }
        match entries.get(&id).unwrap() {
            Entry::Unregistered => {}
            Entry::Live(_) => {
                return Err(AllocationRegistrationError::AlreadyRegistered);
            }
            Entry::Retiring | Entry::Finished(_) => {
                return Err(AllocationRegistrationError::NotLive)
            }
        }
        assert!(
            lease.acquire().is_none(),
            "registration must complete before acquisition"
        );
        entries.insert(id, Entry::Live(lease));
        Ok(())
    }

    fn retire(&self, id: AllocationIdentity) -> AllocationRetirementOutcome {
        {
            let mut entries = self.entries.lock().unwrap();
            assert!(matches!(entries.get(&id), Some(Entry::Live(_))));
            entries.insert(id, Entry::Retiring);
        }
        self.retire_calls.fetch_add(1, Ordering::SeqCst);
        let callback = self.callback.lock().unwrap().take();
        if let Some(callback) = callback {
            callback();
        }
        assert!(
            !self.panic_during_retire.load(Ordering::SeqCst),
            "retirement fault"
        );
        self.entries
            .lock()
            .unwrap()
            .insert(id, Entry::Finished(self.outcome.clone()));
        self.outcome.clone()
    }

    fn record_retirement_panic(&self, id: AllocationIdentity) {
        self.entries.lock().unwrap().insert(
            id,
            Entry::Finished(AllocationRetirementOutcome::Uncertain {
                detail: "unexpected backend unwind".into(),
            }),
        );
    }
}
