//! Opt-in fixture owners publish actual owned typed allocations, never
//! fabricated operational handles. Shipping defaults do not enable this module.
use crate::allocation::*;
use crate::*;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Mutex,
};

#[cfg(not(target_arch = "wasm32"))]
type FixtureOwnerRef = std::sync::Arc<FixtureOwner>;
#[cfg(target_arch = "wasm32")]
type FixtureOwnerRef = std::rc::Rc<FixtureOwner>;

static NEXT: AtomicU64 = AtomicU64::new(1);
mod payload;
#[cfg(test)]
mod tests;

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
    /// Move aligned native storage into the same owner that retires its lease.
    /// This is fixture data, not a simulated device pointer or shipping upload.
    pub fn allocate_numeric(&self, host: HostNumericTensorOwned) -> anyhow::Result<TestAllocation> {
        host.validate()?;
        let (handle, owner) = owned_data(self.device, host, false)?;
        Ok(TestAllocation {
            handle,
            probe: RetirementProbe { owner },
        })
    }

    pub fn allocate_logical(
        &self,
        data: Vec<u8>,
        shape: Vec<usize>,
    ) -> anyhow::Result<TestAllocation> {
        HostLogicalTensorView {
            data: &data,
            shape: &shape,
        }
        .validate()?;
        let host = HostNumericTensorOwned {
            data: HostNumericDataOwned::U8(data),
            shape,
            storage: GpuTensorStorage::Real,
        };
        let (handle, owner) = owned_data(self.device, host, true)?;
        Ok(TestAllocation {
            handle,
            probe: RetirementProbe { owner },
        })
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
    owner: FixtureOwnerRef,
}
impl RetirementProbe {
    /// Make this fixture allocation unavailable, as when its device is lost.
    /// Existing owners still retain the actual payload until final retirement;
    /// this neither frees storage nor records a retirement attempt. The
    /// transition is terminal and does not affect other fixture allocations.
    pub fn mark_unavailable(&self) {
        self.owner.available.store(false, Ordering::Release);
    }

    /// Inject a retained-payload retirement failure. Keep this probe alive to
    /// retain the failure record and payload until the fixture is torn down.
    pub fn fail_retirement(&self, detail: impl Into<String>) {
        let mut failure = self.owner.retirement_failure.lock().unwrap();
        assert_eq!(self.retirements(), 0, "retirement already attempted");
        *failure = Some(detail.into());
    }

    /// Number of retirement attempts, including injected failures.
    pub fn retirements(&self) -> usize {
        self.owner.retired.load(Ordering::SeqCst)
    }

    /// Borrow the actual owner payload without copying it. The callback runs
    /// under the allocation record lock and must not reenter this owner.
    pub fn with_data<T>(
        &self,
        handle: &GpuTensorHandle,
        callback: impl FnOnce(HostNumericDataView<'_>) -> T,
    ) -> anyhow::Result<T> {
        self.check_handle(handle)?;
        let record = self.owner.record.lock().unwrap();
        let data = record
            .0
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("fixture allocation was retired"))?;
        Ok(callback(data.as_view()))
    }

    /// Borrow the stable, correctly aligned address of the actual typed payload.
    /// Native pointer use must retain `handle` (normally via the shared owning
    /// native export wrapper) and obey exclusive-write/context synchronization
    /// requirements. The callback must not reenter the allocation owner.
    pub fn with_native_address<T>(
        &self,
        handle: &GpuTensorHandle,
        callback: impl FnOnce(u64, usize) -> T,
    ) -> anyhow::Result<T> {
        self.check_handle(handle)?;
        let mut record = self.owner.record.lock().unwrap();
        let data = record
            .0
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("fixture allocation was retired"))?;
        let bytes = usize::try_from(self.owner.physical.payload_bytes())?;
        Ok(callback(payload::address(data), bytes))
    }

    fn check_handle(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        if handle.allocation_identity() != self.owner.identity {
            anyhow::bail!("fixture probe does not own this allocation");
        }
        handle.allocation_lease().validate_live()?;
        Ok(())
    }
}
pub(crate) struct FixtureOwner {
    identity: AllocationIdentity,
    physical: GpuPhysicalDescriptor,
    logical: bool,
    available: AtomicBool,
    record: Mutex<(Option<HostNumericDataOwned>, Option<WeakAllocationLease>)>,
    pub retired: AtomicUsize,
    retirement_failure: Mutex<Option<String>>,
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
        if !self.available.load(Ordering::Acquire) || record.0.is_none() {
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
        if !self.available.load(Ordering::Acquire)
            || self.retired.load(Ordering::SeqCst) != 0
            || self.record.lock().unwrap().0.is_none()
        {
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
        let failure = self.retirement_failure.lock().unwrap();
        assert_eq!(self.retired.fetch_add(1, Ordering::SeqCst), 0);
        if let Some(detail) = failure.as_ref() {
            return AllocationRetirementOutcome::RetainedFailure {
                detail: detail.clone(),
            };
        }
        let payload = self.record.lock().unwrap().0.take();
        assert!(payload.is_some());
        drop(payload);
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
) -> (GpuTensorHandle, FixtureOwnerRef) {
    let bytes = crate::native_buffer_byte_length(&shape, element, storage).unwrap();
    let host = HostNumericTensorOwned {
        data: payload::zeroed(element, bytes / element.element_size()),
        shape,
        storage,
    };
    owned_data(device, host, logical).unwrap()
}

fn owned_data(
    device: u32,
    host: HostNumericTensorOwned,
    logical: bool,
) -> anyhow::Result<(GpuTensorHandle, FixtureOwnerRef)> {
    host.validate()?;
    let element = host.data.element_type();
    let bytes = crate::native_buffer_byte_length(&host.shape, element, host.storage)?;
    let count = if host.shape.contains(&0) {
        0
    } else {
        host.shape.iter().product::<usize>()
    };
    let owner = FixtureOwnerRef::new(FixtureOwner {
        identity: AllocationIdentity::new(device, NEXT.fetch_add(1, Ordering::SeqCst), 1),
        physical: GpuPhysicalDescriptor::new(
            element,
            host.storage,
            count as u64,
            bytes as u64,
            bytes as u64,
        )?,
        logical,
        available: AtomicBool::new(true),
        record: Mutex::new((Some(host.data), None)),
        retired: AtomicUsize::new(0),
        retirement_failure: Mutex::new(None),
    });
    let lease = AllocationLease::register(owner.identity, owner.clone())?;
    let handle = GpuTensorHandle::from_allocation(
        lease,
        GpuTensorView::contiguous(host.shape).with_logical(logical),
    )?;
    Ok((handle, owner))
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
