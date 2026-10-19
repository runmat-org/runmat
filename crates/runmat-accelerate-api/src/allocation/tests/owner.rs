//! One test allocation map, mirroring the required existing-provider authority.
use crate::allocation::shared::Shared;
use crate::allocation::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

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

pub(super) struct Record {
    pub(super) phase: Entry,
    pub(super) payload: Option<Box<[u8]>>,
}

#[derive(Clone, Copy)]
pub(super) enum RegistrationFault {
    None,
    RejectBeforeAssociation,
    #[cfg(panic = "unwind")]
    PanicBeforeAssociation,
}

pub(super) struct TestOwner {
    device: u32,
    pub(super) entries: Mutex<HashMap<AllocationIdentity, Record>>,
    pub(super) retire_calls: AtomicUsize,
    pub(super) panic_during_retire: AtomicBool,
    pub(super) registration_fault: Mutex<RegistrationFault>,
    pub(super) observed_unpublished_registration: AtomicBool,
    outcome: AllocationRetirementOutcome,
    callback: Mutex<Option<Callback>>,
}

impl TestOwner {
    pub(super) fn new(
        id: AllocationIdentity,
        outcome: AllocationRetirementOutcome,
    ) -> Shared<Self> {
        Shared::new(Self {
            device: id.device_id(),
            entries: Mutex::new(HashMap::from([(
                id,
                Record {
                    phase: Entry::Unregistered,
                    payload: Some(vec![19, 23].into_boxed_slice()),
                },
            )])),
            retire_calls: AtomicUsize::new(0),
            panic_during_retire: AtomicBool::new(false),
            registration_fault: Mutex::new(RegistrationFault::None),
            observed_unpublished_registration: AtomicBool::new(false),
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
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Some(Entry::Live(weak)) = entries.get(&id).map(|record| &record.phase) else {
            panic!("missing live lease")
        };
        weak.clone()
    }

    pub(super) fn outcome(&self, id: AllocationIdentity) -> AllocationRetirementOutcome {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Some(Entry::Finished(outcome)) = entries.get(&id).map(|record| &record.phase) else {
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
    ) -> Result<GpuPhysicalDescriptor, AllocationRegistrationError> {
        if id.device_id() != self.device {
            return Err(AllocationRegistrationError::WrongOwner);
        }
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner());
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
        match &entries.get(&id).unwrap().phase {
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
        self.observed_unpublished_registration
            .store(true, Ordering::SeqCst);
        // Faults deliberately precede association: rejected/unwound admission
        // leaves the payload available for a later valid registration.
        let fault = *self.registration_fault.lock().unwrap();
        match fault {
            RegistrationFault::RejectBeforeAssociation => {
                return Err(AllocationRegistrationError::NotLive)
            }
            #[cfg(panic = "unwind")]
            RegistrationFault::PanicBeforeAssociation => {
                panic!("pre-association registration fault")
            }
            RegistrationFault::None => {}
        }
        let physical = GpuPhysicalDescriptor::new(
            crate::NumericElementType::U8,
            crate::GpuTensorStorage::Real,
            2,
            2,
            2,
        )
        .unwrap();
        entries.get_mut(&id).unwrap().phase = Entry::Live(lease);
        Ok(physical)
    }

    fn validate_live(&self, id: AllocationIdentity) -> Result<(), AllocationAcquisitionError> {
        if id.device_id() != self.device {
            return Err(AllocationAcquisitionError::WrongOwner);
        }
        let entries = self
            .entries
            .lock()
            .map_err(|_| AllocationAcquisitionError::NotLive)?;
        if entries
            .get(&id)
            .is_some_and(|record| matches!(record.phase, Entry::Live(_)))
        {
            Ok(())
        } else {
            Err(AllocationAcquisitionError::NotLive)
        }
    }

    fn validate_view(
        &self,
        id: AllocationIdentity,
        view: &GpuTensorView,
    ) -> Result<(), AllocationAcquisitionError> {
        self.validate_live(id)?;
        if view.is_logical() {
            return Err(AllocationAcquisitionError::InvalidView(
                GpuViewError::InvalidLogicalClass,
            ));
        }
        Ok(())
    }

    fn retire(&self, id: AllocationIdentity) -> AllocationRetirementOutcome {
        {
            let mut entries = self
                .entries
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            assert!(matches!(
                entries.get(&id).map(|record| &record.phase),
                Some(Entry::Live(_))
            ));
            entries.get_mut(&id).unwrap().phase = Entry::Retiring;
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
        let payload = {
            let mut entries = self
                .entries
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let record = entries.get_mut(&id).unwrap();
            record.phase = Entry::Finished(self.outcome.clone());
            if self.outcome == AllocationRetirementOutcome::Retired {
                record.payload.take()
            } else {
                None
            }
        };
        drop(payload);
        self.outcome.clone()
    }

    fn record_retirement_panic(&self, id: AllocationIdentity) {
        self.entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get_mut(&id)
            .unwrap()
            .phase = Entry::Finished(AllocationRetirementOutcome::Uncertain {
            detail: "unexpected backend unwind".into(),
        });
    }
}
