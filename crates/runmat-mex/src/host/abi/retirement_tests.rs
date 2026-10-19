//! Real allocation retirement reenters the host only after arena detachment.
use super::*;
use runmat_accelerate_api::allocation::{
    AllocationAcquisitionError, AllocationIdentity, AllocationLease, AllocationOwner,
    AllocationRegistrationError, AllocationRetirementOutcome, WeakAllocationLease,
};
use runmat_accelerate_api::{
    GpuPhysicalDescriptor, GpuTensorHandle, GpuTensorStorage, GpuTensorView, NumericElementType,
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Weak,
};

struct ReentrantOwner {
    identity: AllocationIdentity,
    payload: Mutex<(Option<Vec<f64>>, Option<WeakAllocationLease>)>,
    host: Weak<MexCallState>,
    calls: AtomicUsize,
    unlocked: AtomicBool,
}
impl ReentrantOwner {
    fn physical() -> GpuPhysicalDescriptor {
        GpuPhysicalDescriptor::new(NumericElementType::F64, GpuTensorStorage::Real, 1, 8, 8)
            .unwrap()
    }
}
impl AllocationOwner for ReentrantOwner {
    fn register_lease(
        &self,
        id: AllocationIdentity,
        lease: WeakAllocationLease,
    ) -> Result<GpuPhysicalDescriptor, AllocationRegistrationError> {
        if id != self.identity {
            return Err(AllocationRegistrationError::WrongOwner);
        }
        let mut record = self.payload.lock().unwrap();
        if record.0.is_none() {
            return Err(AllocationRegistrationError::NotLive);
        }
        if record.1.is_some() {
            return Err(AllocationRegistrationError::AlreadyRegistered);
        }
        record.1 = Some(lease);
        Ok(Self::physical())
    }
    fn validate_live(&self, id: AllocationIdentity) -> Result<(), AllocationAcquisitionError> {
        if id != self.identity {
            return Err(AllocationAcquisitionError::WrongOwner);
        }
        if self.payload.lock().unwrap().0.is_none() {
            return Err(AllocationAcquisitionError::NotLive);
        }
        Ok(())
    }
    fn validate_view(
        &self,
        id: AllocationIdentity,
        view: &GpuTensorView,
    ) -> Result<(), AllocationAcquisitionError> {
        self.validate_live(id)?;
        if view.is_logical() {
            return Err(AllocationAcquisitionError::Unsupported);
        }
        view.validate(&Self::physical())
            .map_err(AllocationAcquisitionError::InvalidView)
    }
    fn retire(&self, id: AllocationIdentity) -> AllocationRetirementOutcome {
        assert_eq!(id, self.identity);
        let payload = self.payload.lock().unwrap().0.take();
        assert_eq!(payload.as_deref(), Some([7.0].as_slice()));
        drop(payload);
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(host) = self.host.upgrade() {
            let available = host.inner.try_lock().is_ok();
            self.unlocked.store(available, Ordering::SeqCst);
            if available {
                host.begin_call();
            }
        }
        AllocationRetirementOutcome::Retired
    }
    fn record_retirement_panic(&self, _: AllocationIdentity) {
        self.unlocked.store(false, Ordering::SeqCst);
    }
}

fn allocation(host: &Arc<MexCallState>) -> (GpuTensorHandle, Arc<ReentrantOwner>) {
    let identity = AllocationIdentity::new(runmat_accelerate_api::next_device_id(), 1, 1);
    let owner = Arc::new(ReentrantOwner {
        identity,
        payload: Mutex::new((Some(vec![7.0]), None)),
        host: Arc::downgrade(host),
        calls: AtomicUsize::new(0),
        unlocked: AtomicBool::new(false),
    });
    let lease = AllocationLease::register(identity, owner.clone()).unwrap();
    let handle =
        GpuTensorHandle::from_allocation(lease, GpuTensorView::contiguous(vec![1, 1])).unwrap();
    (handle, owner)
}
fn gpu_array(host: &Arc<MexCallState>) -> (*mut MxArray, Arc<ReentrantOwner>) {
    let (handle, owner) = allocation(host);
    let array =
        MxArray::gpu_owned(MxClassId::Double, crate::mxarray::MxGpuLease::owned(handle)).unwrap();
    let pointer = host.lock().unwrap().mx.allocate(array);
    (pointer, owner)
}
fn retired_once(owner: &ReentrantOwner) {
    assert_eq!(owner.calls.load(Ordering::SeqCst), 1);
    assert!(
        owner.unlocked.load(Ordering::SeqCst),
        "physical retirement must run outside the host state lock"
    );
    assert!(owner.payload.lock().unwrap().0.is_none());
}

#[test]
fn finish_destroy_and_data_release_retire_outside_host_lock() {
    for route in 0..3 {
        let host = Arc::new(MexCallState::new(MxApiMode::InterleavedComplex));
        host.begin_call();
        let (array, owner) = gpu_array(&host);
        let api = host.host_api();
        match route {
            0 => host.finish_call(),
            1 => assert_eq!(unsafe { destroy_array(api.host, array) }, 0),
            _ => {
                assert_eq!(unsafe { retain_data_array(api.host, array) }, 0);
                assert_eq!(unsafe { release_data_array(api.host, array, 1) }, 0);
            }
        }
        retired_once(&owner);
        assert!(host.lock().unwrap().mx.arena().is_empty());
        drop(host);
        assert_eq!(owner.calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn nested_cell_retirement_and_overwrite_run_outside_host_lock() {
    for overwrite in [false, true] {
        let host = Arc::new(MexCallState::new(MxApiMode::InterleavedComplex));
        let (child, owner) = gpu_array(&host);
        let parent = host
            .lock()
            .unwrap()
            .mx
            .allocate(MxArray::cell(vec![None], vec![1, 1]).unwrap());
        let api = host.host_api();
        assert_eq!(unsafe { set_cell(api.host, parent, 0, child) }, 0);
        if overwrite {
            assert_eq!(
                unsafe { set_cell(api.host, parent, 0, std::ptr::null_mut()) },
                0
            );
        } else {
            host.finish_call();
        }
        retired_once(&owner);
    }
}

#[test]
fn caught_failure_attachment_retires_owned_cause_outside_host_lock() {
    #[derive(Debug)]
    struct Cause {
        _allocation: GpuTensorHandle,
    }
    impl std::fmt::Display for Cause {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("upload failure")
        }
    }
    impl std::error::Error for Cause {}
    let host = Arc::new(MexCallState::new(MxApiMode::InterleavedComplex));
    let (handle, owner) = allocation(&host);
    host.lock()
        .unwrap()
        .fail_gpu(crate::MexGpuError::Provider(anyhow::Error::new(Cause {
            _allocation: handle,
        })));
    let api = host.host_api();
    assert!(!unsafe { take_error(api.host) }.is_null());
    host.finish_call();
    retired_once(&owner);
}

#[test]
fn persistent_nested_child_remains_owned_until_explicit_destruction() {
    let host = Arc::new(MexCallState::new(MxApiMode::InterleavedComplex));
    let (child, owner) = gpu_array(&host);
    let parent = {
        let mut state = host.lock().unwrap();
        state.mx.make_persistent(child).unwrap();
        state
            .mx
            .allocate(MxArray::cell(vec![None], vec![1, 1]).unwrap())
    };
    let api = host.host_api();
    assert_eq!(unsafe { set_cell(api.host, parent, 0, child) }, 0);
    host.finish_call();
    assert_eq!(owner.calls.load(Ordering::SeqCst), 0);
    assert!(host.lock().unwrap().mx.arena().get(child).is_ok());
    assert!(host.lock().unwrap().mx.arena().get(parent).is_err());
    assert_eq!(unsafe { destroy_array(api.host, child) }, 0);
    retired_once(&owner);
}
