//! Admission and retirement through the existing CUDA allocation records.
use super::*;
use runmat_accelerate_api::allocation::{
    AllocationAcquisitionError as Acquire, AllocationOwner,
    AllocationRegistrationError as Register, AllocationRetirementOutcome, GpuTensorLayout,
    GpuTensorWireDescriptor,
};

impl AllocationState {
    pub(in crate::backend::cuda) fn acquire(
        &self,
        wire: &GpuTensorWireDescriptor,
    ) -> Result<GpuTensorHandle, Acquire> {
        self.validate_live(wire.allocation)?;
        let weak = self
            .records()
            .get(&wire.allocation.allocation_id())
            .and_then(|record| record.lease.clone())
            .ok_or(Acquire::NotLive)?;
        // Owner validation re-enters the map; never keep its lock across upgrade.
        weak.acquire_tensor(wire)
    }

    pub(in crate::backend::cuda) fn validate_native_write(
        &self,
        handle: &GpuTensorHandle,
    ) -> Result<()> {
        self.allocation(handle)?;
        let records = self.records();
        let record = records
            .get(&handle.buffer_id())
            .ok_or_else(|| anyhow!("CUDA allocation is not live"))?;
        anyhow::ensure!(
            !record.class.is_logical() || handle.view().is_logical(),
            "a numeric view cannot expose mutable logical allocation storage"
        );
        Ok(())
    }
}

impl AllocationOwner for AllocationState {
    fn register_lease(
        &self,
        id: AllocationIdentity,
        lease: WeakAllocationLease,
    ) -> Result<GpuPhysicalDescriptor, Register> {
        if id.device_id() != self.device_id {
            return Err(Register::WrongOwner);
        }
        if id.generation() != 1 {
            return Err(Register::StaleGeneration);
        }
        self.ensure_available().map_err(|_| Register::NotLive)?;
        let mut records = self.records();
        let record = records
            .get_mut(&id.allocation_id())
            .ok_or(Register::UnknownAllocation)?;
        if !matches!(record.phase, Phase::Live) || record.allocation.is_none() {
            return Err(Register::NotLive);
        }
        if record.lease.is_some() {
            return Err(Register::AlreadyRegistered);
        }
        let physical = record.physical;
        record.lease = Some(lease);
        Ok(physical)
    }

    fn validate_live(&self, id: AllocationIdentity) -> Result<(), Acquire> {
        if id.device_id() != self.device_id {
            return Err(Acquire::WrongOwner);
        }
        if id.generation() != 1 {
            return Err(Acquire::StaleGeneration);
        }
        self.ensure_available().map_err(|_| Acquire::NotLive)?;
        let records = self.records();
        let record = records.get(&id.allocation_id()).ok_or(Acquire::NotLive)?;
        if !matches!(record.phase, Phase::Live) || record.allocation.is_none() {
            return Err(Acquire::NotLive);
        }
        Ok(())
    }

    fn validate_view(&self, id: AllocationIdentity, view: &GpuTensorView) -> Result<(), Acquire> {
        self.validate_live(id)?;
        // CUDA has no lazy-layout producer or addressing kernel. The existing
        // runtime transpose fallback remains responsible for materialization.
        if !matches!(view.layout(), GpuTensorLayout::Contiguous) {
            return Err(Acquire::Unsupported);
        }
        let records = self.records();
        let record = records.get(&id.allocation_id()).ok_or(Acquire::NotLive)?;
        if view.is_logical() && !record.class.is_logical() {
            return Err(Acquire::InvalidView(
                runmat_accelerate_api::allocation::GpuViewError::InvalidLogicalClass,
            ));
        }
        Ok(())
    }

    fn retire(&self, id: AllocationIdentity) -> AllocationRetirementOutcome {
        if id.device_id() != self.device_id || id.generation() != 1 {
            return AllocationRetirementOutcome::Uncertain {
                detail: "CUDA retirement identity mismatch".into(),
            };
        }
        match self.retire_record(id.allocation_id()) {
            Ok(()) => AllocationRetirementOutcome::Retired,
            Err(error) => {
                let records = self.records();
                match records.get(&id.allocation_id()).map(|record| &record.phase) {
                    Some(Phase::EntryFailed(detail)) => {
                        AllocationRetirementOutcome::RetryableFailure {
                            detail: detail.clone(),
                        }
                    }
                    Some(Phase::Uncertain(detail)) => AllocationRetirementOutcome::Uncertain {
                        detail: detail.clone(),
                    },
                    Some(Phase::FreedExitFailed(_)) => AllocationRetirementOutcome::DeviceLost {
                        detail: error.to_string(),
                    },
                    _ => AllocationRetirementOutcome::RetainedFailure {
                        detail: error.to_string(),
                    },
                }
            }
        }
    }

    fn record_retirement_panic(&self, id: AllocationIdentity) {
        // Preserve a known released pointer if only context exit unwound.
        let mut records = self.records();
        if let Some(record) = records.get_mut(&id.allocation_id()) {
            if !matches!(record.phase, Phase::FreedExitFailed(_)) {
                record.phase = Phase::Uncertain("CUDA retirement unwound".into());
            }
        }
    }
}
