use super::*;
use runmat_accelerate_api::allocation::{
    AllocationAcquisitionError as Acquire, AllocationOwner,
    AllocationRegistrationError as Register, GpuTensorLayout, GpuTensorView,
    GpuTensorWireDescriptor, GpuViewError,
};
use runmat_accelerate_api::GpuTensorHandle;

impl NumericAllocationState {
    pub(in crate::simple_provider) fn acquire(
        &self,
        wire: &GpuTensorWireDescriptor,
    ) -> Result<GpuTensorHandle, Acquire> {
        let weak = {
            let records = self.allocations.lock().map_err(|_| Acquire::NotLive)?;
            let record = records
                .get(&wire.allocation.allocation_id())
                .ok_or(Acquire::NotLive)?;
            if record.identity != wire.allocation {
                return Err(Acquire::WrongOwner);
            }
            let Phase::Live(weak) = &record.phase else {
                return Err(Acquire::NotLive);
            };
            weak.clone()
        };
        weak.acquire_tensor(wire)
    }
}

impl AllocationOwner for NumericAllocationState {
    fn register_lease(
        &self,
        identity: AllocationIdentity,
        lease: WeakAllocationLease,
    ) -> Result<GpuPhysicalDescriptor, Register> {
        let mut records = self.allocations.lock().map_err(|_| Register::NotLive)?;
        let record = records
            .get_mut(&identity.allocation_id())
            .ok_or(Register::UnknownAllocation)?;
        if record.identity.device_id() != identity.device_id() {
            return Err(Register::WrongOwner);
        }
        if record.identity.generation() != identity.generation() {
            return Err(Register::StaleGeneration);
        }
        match record.phase {
            Phase::Pending => {}
            Phase::Live(_) => return Err(Register::AlreadyRegistered),
            _ => return Err(Register::NotLive),
        }
        if record.payload.is_none() {
            return Err(Register::NotLive);
        }
        let physical = record.physical;
        record.phase = Phase::Live(lease);
        Ok(physical)
    }

    fn validate_live(&self, identity: AllocationIdentity) -> Result<(), Acquire> {
        let records = self.allocations.lock().map_err(|_| Acquire::NotLive)?;
        let record = records
            .get(&identity.allocation_id())
            .ok_or(Acquire::NotLive)?;
        if record.identity.device_id() != identity.device_id() {
            return Err(Acquire::WrongOwner);
        }
        if record.identity.generation() != identity.generation() {
            return Err(Acquire::StaleGeneration);
        }
        if !matches!(record.phase, Phase::Live(_)) || record.payload.is_none() {
            return Err(Acquire::NotLive);
        }
        Ok(())
    }

    fn validate_view(
        &self,
        identity: AllocationIdentity,
        view: &GpuTensorView,
    ) -> Result<(), Acquire> {
        self.validate_live(identity)?;
        if !matches!(view.layout(), GpuTensorLayout::Contiguous) {
            return Err(Acquire::Unsupported);
        }
        let records = self.allocations.lock().map_err(|_| Acquire::NotLive)?;
        let record = records
            .get(&identity.allocation_id())
            .ok_or(Acquire::NotLive)?;
        let count = if view.shape().contains(&0) {
            0
        } else {
            view.shape()
                .iter()
                .try_fold(1u64, |n, d| n.checked_mul(*d as u64))
                .ok_or(Acquire::InvalidView(GpuViewError::SizeOverflow))?
        };
        // Existing kernels consume the full typed payload. Reshape is a view,
        // but prefix slicing and lazy addressing are not implemented here.
        if count != record.physical.element_capacity() {
            return Err(Acquire::Unsupported);
        }
        if view.is_logical() && !record.producer_logical {
            return Err(Acquire::InvalidView(GpuViewError::InvalidLogicalClass));
        }
        Ok(())
    }

    fn retire(&self, identity: AllocationIdentity) -> AllocationRetirementOutcome {
        // Recover solely for exact teardown. No new publication or live access
        // is admitted through a poisoned owner.
        let detached = {
            let mut records = self
                .allocations
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let Some(record) = records.get_mut(&identity.allocation_id()) else {
                return AllocationRetirementOutcome::RetainedFailure {
                    detail: "in-process allocation record is absent".into(),
                };
            };
            if record.identity == identity {
                if let Phase::Failed(outcome) = &record.phase {
                    return outcome.clone();
                }
            }
            if record.identity != identity || !matches!(record.phase, Phase::Live(_)) {
                return AllocationRetirementOutcome::RetainedFailure {
                    detail: "in-process allocation is not eligible for retirement".into(),
                };
            }
            record.phase = Phase::Retiring;
            records.remove(&identity.allocation_id())
        };
        drop(detached);
        AllocationRetirementOutcome::Retired
    }

    fn record_retirement_panic(&self, identity: AllocationIdentity) {
        let mut records = self
            .allocations
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(record) = records.get_mut(&identity.allocation_id()) {
            if record.identity == identity {
                record.phase = Phase::Failed(AllocationRetirementOutcome::Uncertain {
                    detail: "in-process retirement unwound".into(),
                });
            }
        }
    }
}
