//! Lease admission uses the existing allocation table, never an ID-to-owner map.
use super::*;
use runmat_accelerate_api::allocation::{
    AllocationAcquisitionError, AllocationIdentity, AllocationOwner, AllocationRegistrationError,
    AllocationRetirementOutcome, GpuPhysicalDescriptor, GpuTensorView, GpuTensorWireDescriptor,
    WeakAllocationLease,
};

impl AllocationState {
    pub(super) fn entry_for_handle(&self, handle: &GpuTensorHandle) -> Result<BufferEntry> {
        self.check_identity(handle.allocation_identity())?;
        let (mut entry, weak) = {
            let records = self.lock_records()?;
            let record = records
                .get(&handle.buffer_id())
                .ok_or(AllocationAcquisitionError::NotLive)?;
            if record.phase != Phase::Live {
                return Err(AllocationAcquisitionError::NotLive.into());
            }
            (
                record
                    .entry
                    .clone()
                    .ok_or(AllocationAcquisitionError::NotLive)?,
                record
                    .lease
                    .clone()
                    .ok_or(AllocationAcquisitionError::NotLive)?,
            )
        };
        let lease = weak.acquire().ok_or(AllocationAcquisitionError::NotLive)?;
        if !lease.shares_allocation_with(handle.allocation_lease()) {
            return Err(AllocationAcquisitionError::WrongOwner.into());
        }
        self.validate_view(handle.allocation_identity(), handle.view())?;
        entry.shape = handle.shape().to_vec();
        entry.logical = handle.view().is_logical();
        Ok(entry)
    }

    pub(super) fn requires_logical_encoding(&self, handle: &GpuTensorHandle) -> Result<bool> {
        self.entry_for_handle(handle)?;
        let records = self.lock_records()?;
        Ok(records
            .get(&handle.buffer_id())
            .ok_or(AllocationAcquisitionError::NotLive)?
            .logical_producer)
    }

    fn check_identity(
        &self,
        identity: AllocationIdentity,
    ) -> Result<(), AllocationAcquisitionError> {
        if identity.device_id() != self.device_id {
            return Err(AllocationAcquisitionError::WrongOwner);
        }
        // Monotonic allocation IDs are never reused; this owner has one generation.
        if identity.generation() != 1 {
            return Err(AllocationAcquisitionError::StaleGeneration);
        }
        if self.is_lost() {
            return Err(AllocationAcquisitionError::NotLive);
        }
        Ok(())
    }

    pub(super) fn acquire_tensor(
        &self,
        wire: &GpuTensorWireDescriptor,
    ) -> Result<GpuTensorHandle, AllocationAcquisitionError> {
        self.check_identity(wire.allocation)?;
        let weak = {
            let records = self
                .buffers
                .lock()
                .map_err(|_| AllocationAcquisitionError::NotLive)?;
            let record = records
                .get(&wire.allocation.allocation_id())
                .ok_or(AllocationAcquisitionError::NotLive)?;
            if record.phase != Phase::Live {
                return Err(AllocationAcquisitionError::NotLive);
            }
            record
                .lease
                .clone()
                .ok_or(AllocationAcquisitionError::NotLive)?
        };
        // Acquisition validates the owner again; never upgrade under its lock.
        weak.acquire_tensor(wire)
    }
}

impl AllocationOwner for AllocationState {
    fn register_lease(
        &self,
        identity: AllocationIdentity,
        lease: WeakAllocationLease,
    ) -> Result<GpuPhysicalDescriptor, AllocationRegistrationError> {
        if identity.device_id() != self.device_id {
            return Err(AllocationRegistrationError::WrongOwner);
        }
        if identity.generation() != 1 {
            return Err(AllocationRegistrationError::StaleGeneration);
        }
        let mut records = self
            .buffers
            .lock()
            .map_err(|_| AllocationRegistrationError::NotLive)?;
        if self.is_lost() {
            return Err(AllocationRegistrationError::NotLive);
        }
        let record = records
            .get_mut(&identity.allocation_id())
            .ok_or(AllocationRegistrationError::UnknownAllocation)?;
        if record.lease.is_some() {
            return Err(AllocationRegistrationError::AlreadyRegistered);
        }
        if !matches!(record.phase, Phase::Registering(_)) || record.entry.is_none() {
            return Err(AllocationRegistrationError::NotLive);
        }
        let physical = record.physical;
        record.lease = Some(lease);
        record.phase = Phase::Live;
        Ok(physical)
    }

    fn validate_live(
        &self,
        identity: AllocationIdentity,
    ) -> Result<(), AllocationAcquisitionError> {
        self.check_identity(identity)?;
        let records = self
            .buffers
            .lock()
            .map_err(|_| AllocationAcquisitionError::NotLive)?;
        let record = records
            .get(&identity.allocation_id())
            .ok_or(AllocationAcquisitionError::NotLive)?;
        if record.phase != Phase::Live || record.entry.is_none() || record.lease.is_none() {
            return Err(AllocationAcquisitionError::NotLive);
        }
        Ok(())
    }

    fn validate_view(
        &self,
        identity: AllocationIdentity,
        view: &GpuTensorView,
    ) -> Result<(), AllocationAcquisitionError> {
        self.check_identity(identity)?;
        let records = self
            .buffers
            .lock()
            .map_err(|_| AllocationAcquisitionError::NotLive)?;
        let record = records
            .get(&identity.allocation_id())
            .ok_or(AllocationAcquisitionError::NotLive)?;
        if record.phase != Phase::Live || record.entry.is_none() || record.lease.is_none() {
            return Err(AllocationAcquisitionError::NotLive);
        }
        view.validate(&record.physical)
            .map_err(AllocationAcquisitionError::InvalidView)?;
        let count = if view.shape().contains(&0) {
            Some(0)
        } else {
            view.shape()
                .iter()
                .try_fold(1_u64, |count, &dim| count.checked_mul(dim as u64))
        };
        // Existing WGPU kernels address complete allocations; prefix views
        // would expose a physical length inconsistent with their shorter shape.
        if count != Some(record.physical.element_capacity()) {
            return Err(AllocationAcquisitionError::Unsupported);
        }
        if view.is_logical() && !record.logical_producer {
            return Err(AllocationAcquisitionError::Unsupported);
        }
        Ok(())
    }

    fn retire(&self, identity: AllocationIdentity) -> AllocationRetirementOutcome {
        if identity.device_id() != self.device_id || identity.generation() != 1 {
            return AllocationRetirementOutcome::Uncertain {
                detail: "WGPU retirement owner identity mismatch".into(),
            };
        }
        match self.retire_physical(identity.allocation_id()) {
            Ok(()) => AllocationRetirementOutcome::Retired,
            Err(error) => {
                let disposition = {
                    let mut records = self
                        .buffers
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    records.get_mut(&identity.allocation_id()).map(|record| {
                        if record.phase == Phase::Live {
                            // A poisoned initial map lock prevented the first
                            // transition; quarantine without releasing payload.
                            record.phase = Phase::Failed {
                                stage: Stage::Cache,
                                failure: Failure::MapPoisoned,
                                disposition: Disposition::Retained,
                            };
                        }
                        match record.phase {
                            Phase::Failed { disposition, .. } => disposition,
                            _ => Disposition::Uncertain,
                        }
                    })
                };
                match disposition {
                    Some(Disposition::Released) => AllocationRetirementOutcome::Retired,
                    Some(Disposition::Retained) => AllocationRetirementOutcome::RetainedFailure {
                        detail: error.to_string(),
                    },
                    _ => AllocationRetirementOutcome::Uncertain {
                        detail: error.to_string(),
                    },
                }
            }
        }
    }

    fn record_retirement_panic(&self, identity: AllocationIdentity) {
        if identity.device_id() == self.device_id && identity.generation() == 1 {
            self.fail(
                identity.allocation_id(),
                Stage::Resource,
                Failure::Unwind,
                Disposition::Uncertain,
                None,
            );
        }
    }
}
