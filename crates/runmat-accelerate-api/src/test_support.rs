//! Opt-in fixture owners publish actual owned zeroed byte allocations, never
//! fabricated operational handles. Shipping defaults do not enable this module.
use crate::allocation::*;
use crate::*;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};

static NEXT: AtomicU64 = AtomicU64::new(1);

/// Device identity is minted privately. Fixture providers delegate their
/// device_id to this factory; callers cannot impersonate a shipping provider.
pub struct TestAllocationOwner {
    device: u32,
}
impl Default for TestAllocationOwner {
    fn default() -> Self {
        Self::new()
    }
}
impl TestAllocationOwner {
    pub fn new() -> Self {
        Self {
            device: crate::next_device_id(),
        }
    }
    pub fn device_id(&self) -> u32 {
        self.device
    }
    pub fn allocate_zeroed(
        &self,
        shape: Vec<usize>,
        element: NumericElementType,
        storage: GpuTensorStorage,
    ) -> anyhow::Result<TestAllocation> {
        native_buffer_byte_length(&shape, element, storage)?;
        let (handle, owner) = owned(self.device, shape, element, storage, false);
        Ok(TestAllocation {
            handle,
            probe: RetirementProbe { owner },
        })
    }
    /// Zero-filled floating storage is an actual trusted logical producer.
    pub fn allocate_logical_zeroed(
        &self,
        shape: Vec<usize>,
        precision: ProviderPrecision,
    ) -> anyhow::Result<TestAllocation> {
        let element = match precision {
            ProviderPrecision::F32 => NumericElementType::F32,
            ProviderPrecision::F64 => NumericElementType::F64,
        };
        native_buffer_byte_length(&shape, element, GpuTensorStorage::Real)?;
        let (handle, owner) = owned(self.device, shape, element, GpuTensorStorage::Real, true);
        Ok(TestAllocation {
            handle,
            probe: RetirementProbe { owner },
        })
    }
}
pub struct TestAllocation {
    pub handle: GpuTensorHandle,
    pub probe: RetirementProbe,
}
#[derive(Clone)]
pub struct RetirementProbe {
    owner: Arc<FixtureOwner>,
}
impl RetirementProbe {
    pub fn retirements(&self) -> usize {
        self.owner.retired.load(Ordering::SeqCst)
    }
}
pub(crate) struct FixtureOwner {
    identity: AllocationIdentity,
    physical: GpuPhysicalDescriptor,
    logical: bool,
    record: Mutex<(Option<Vec<u8>>, Option<WeakAllocationLease>)>,
    pub retired: AtomicUsize,
}
impl AllocationOwner for FixtureOwner {
    fn register_lease(
        &self,
        identity: AllocationIdentity,
        lease: WeakAllocationLease,
    ) -> Result<GpuPhysicalDescriptor, AllocationRegistrationError> {
        if identity != self.identity {
            return Err(AllocationRegistrationError::WrongOwner);
        }
        let mut record = self.record.lock().unwrap();
        if record.0.is_none() {
            return Err(AllocationRegistrationError::NotLive);
        }
        if record.1.is_some() {
            return Err(AllocationRegistrationError::AlreadyRegistered);
        }
        record.1 = Some(lease);
        Ok(self.physical)
    }
    fn validate_live(
        &self,
        identity: AllocationIdentity,
    ) -> Result<(), AllocationAcquisitionError> {
        if identity != self.identity {
            return Err(AllocationAcquisitionError::WrongOwner);
        }
        if self.record.lock().unwrap().0.is_none() {
            return Err(AllocationAcquisitionError::NotLive);
        }
        Ok(())
    }
    fn validate_view(
        &self,
        identity: AllocationIdentity,
        view: &GpuTensorView,
    ) -> Result<(), AllocationAcquisitionError> {
        self.validate_live(identity)?;
        if view.is_logical() && !self.logical {
            return Err(AllocationAcquisitionError::InvalidView(
                GpuViewError::InvalidLogicalClass,
            ));
        }
        Ok(())
    }
    fn retire(&self, identity: AllocationIdentity) -> AllocationRetirementOutcome {
        assert_eq!(identity, self.identity);
        let payload = self.record.lock().unwrap().0.take();
        assert!(payload.is_some());
        drop(payload);
        self.retired.fetch_add(1, Ordering::SeqCst);
        AllocationRetirementOutcome::Retired
    }
    fn record_retirement_panic(&self, _: AllocationIdentity) {}
}

pub(crate) fn owned(
    device: u32,
    shape: Vec<usize>,
    element: NumericElementType,
    storage: GpuTensorStorage,
    logical: bool,
) -> (GpuTensorHandle, Arc<FixtureOwner>) {
    let count = if shape.contains(&0) {
        0
    } else {
        shape.iter().product::<usize>()
    };
    let bytes = crate::native_buffer_byte_length(&shape, element, storage).unwrap();
    let owner = Arc::new(FixtureOwner {
        identity: AllocationIdentity::new(device, NEXT.fetch_add(1, Ordering::SeqCst), 1),
        physical: GpuPhysicalDescriptor::new(
            element,
            storage,
            count as u64,
            bytes as u64,
            bytes as u64,
        )
        .unwrap(),
        logical,
        record: Mutex::new((Some(vec![0; bytes]), None)),
        retired: AtomicUsize::new(0),
    });
    let lease = AllocationLease::register(owner.identity, owner.clone()).unwrap();
    let handle = GpuTensorHandle::from_allocation(
        lease,
        GpuTensorView::contiguous(shape).with_logical(logical),
    )
    .unwrap();
    (handle, owner)
}
#[cfg(test)]
pub(crate) fn handle(
    device: u32,
    shape: Vec<usize>,
    element: NumericElementType,
    storage: GpuTensorStorage,
) -> GpuTensorHandle {
    owned(device, shape, element, storage, false).0
}
