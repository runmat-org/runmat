use super::super::super::owner_shared::OwnerShared;
use super::*;
use runmat_accelerate_api::allocation::{
    AllocationAcquisitionError, AllocationIdentity, AllocationLease, AllocationOwner,
    AllocationRegistrationError, AllocationRetirementOutcome, GpuPhysicalDescriptor,
    WeakAllocationLease,
};
use runmat_accelerate_api::{GpuTensorStorage, GpuTensorView, NumericElementType};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct CascadeProbe {
    calls: AtomicUsize,
    active: AtomicUsize,
    peak: AtomicUsize,
}

struct CascadeOwner {
    identity: AllocationIdentity,
    lease: Mutex<Option<WeakAllocationLease>>,
    payload: Mutex<Option<Box<[u8]>>>,
    next: Mutex<Option<GpuTensorHandle>>,
    retired: AtomicBool,
    probe: Arc<CascadeProbe>,
}

impl AllocationOwner for CascadeOwner {
    fn register_lease(
        &self,
        identity: AllocationIdentity,
        lease: WeakAllocationLease,
    ) -> std::result::Result<GpuPhysicalDescriptor, AllocationRegistrationError> {
        if identity != self.identity {
            return Err(AllocationRegistrationError::WrongOwner);
        }
        if self.retired.load(Ordering::Acquire) || self.payload.lock().unwrap().is_none() {
            return Err(AllocationRegistrationError::NotLive);
        }
        let physical =
            GpuPhysicalDescriptor::new(NumericElementType::U8, GpuTensorStorage::Real, 1, 1, 1)
                .unwrap();
        let mut slot = self.lease.lock().unwrap();
        if slot.is_some() {
            return Err(AllocationRegistrationError::AlreadyRegistered);
        }
        *slot = Some(lease);
        Ok(physical)
    }

    fn validate_live(
        &self,
        identity: AllocationIdentity,
    ) -> std::result::Result<(), AllocationAcquisitionError> {
        if identity != self.identity {
            return Err(AllocationAcquisitionError::WrongOwner);
        }
        if self.retired.load(Ordering::Acquire) || self.payload.lock().unwrap().is_none() {
            return Err(AllocationAcquisitionError::NotLive);
        }
        Ok(())
    }

    fn validate_view(
        &self,
        identity: AllocationIdentity,
        _view: &GpuTensorView,
    ) -> std::result::Result<(), AllocationAcquisitionError> {
        self.validate_live(identity)
    }

    fn retire(&self, identity: AllocationIdentity) -> AllocationRetirementOutcome {
        assert_eq!(identity, self.identity);
        assert!(!self.retired.swap(true, Ordering::AcqRel));
        let active = self.probe.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.probe.peak.fetch_max(active, Ordering::SeqCst);
        self.payload.lock().unwrap().take();
        if let Some(next) = self.next.lock().unwrap().take() {
            drain_detached(vec![next]).expect("callback release queue must drain");
        }
        self.probe.calls.fetch_add(1, Ordering::SeqCst);
        self.probe.active.fetch_sub(1, Ordering::SeqCst);
        AllocationRetirementOutcome::Retired
    }

    fn record_retirement_panic(&self, identity: AllocationIdentity) {
        assert_eq!(identity, self.identity);
        self.retired.store(true, Ordering::Release);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn captured_allocation_retirement_callback_cascade_drains_iteratively() {
    const CHAIN: usize = 4096;
    let probe = Arc::new(CascadeProbe {
        calls: AtomicUsize::new(0),
        active: AtomicUsize::new(0),
        peak: AtomicUsize::new(0),
    });
    let mut owners = Vec::with_capacity(CHAIN);
    let mut next = None;
    for id in (1..=CHAIN as u64).rev() {
        let identity = AllocationIdentity::new(1, id, 1);
        let owner = OwnerShared::new(CascadeOwner {
            identity,
            lease: Mutex::new(None),
            payload: Mutex::new(Some(vec![id as u8].into_boxed_slice())),
            next: Mutex::new(next),
            retired: AtomicBool::new(false),
            probe: probe.clone(),
        });
        let lease = AllocationLease::register(identity, owner.clone()).unwrap();
        let handle =
            GpuTensorHandle::from_allocation(lease, GpuTensorView::contiguous(vec![1])).unwrap();
        owners.push(owner);
        next = Some(handle);
    }
    drain_detached(vec![next.unwrap()]).unwrap();
    assert_eq!(probe.calls.load(Ordering::SeqCst), CHAIN);
    assert_eq!(probe.active.load(Ordering::SeqCst), 0);
    assert_eq!(probe.peak.load(Ordering::SeqCst), 1);
    assert!(owners
        .iter()
        .all(|owner| owner.retired.load(Ordering::Acquire)));
}
