//! Lease admission uses the existing allocation table, never an ID-to-owner map.
use super::*;
use runmat_accelerate_api::allocation::{
    AllocationAcquisitionError, AllocationIdentity, AllocationOwner, AllocationRegistrationError,
    AllocationRetirementOutcome, GpuPhysicalDescriptor, GpuTensorView, GpuTensorWireDescriptor,
    WeakAllocationLease,
};

impl AllocationState {
    pub(in super::super) fn entry_for_handle(
        &self,
        handle: &GpuTensorHandle,
    ) -> Result<BufferEntry> {
        self.check_identity(handle.allocation_identity())?;
        let (entry, weak) = {
            let records = self.lock_records()?;
            let record = records
                .get(&handle.buffer_id())
                .ok_or(AllocationAcquisitionError::NotLive)?;
            if record.phase != Phase::Live {
                return Err(AllocationAcquisitionError::NotLive.into());
            }
            let stored = record
                .entry
                .as_ref()
                .ok_or(AllocationAcquisitionError::NotLive)?;
            (
                // The stored physical shape may have a different rank from
                // this view's logical shape. Construct the authenticated
                // operation entry directly rather than first allocating a
                // clone of that obsolete physical shape and replacing it.
                BufferEntry {
                    logical: handle.view().is_logical(),
                    buffer: stored.buffer.clone(),
                    len: stored.len,
                    shape: runmat_value::exact_capacity_vec_from_slice(handle.shape())?,
                    storage: stored.storage,
                    element_type: stored.element_type,
                    precision: stored.precision,
                    allocated_bytes: stored.allocated_bytes,
                    usage: stored.usage,
                    last_submission_id: stored.last_submission_id,
                },
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
        // A zero-element carrier may retain a larger allocation, but existing
        // kernels still require a full-allocation view and physical length.
        if handle.shape().contains(&0) && lease.physical().element_capacity() != 0 {
            return Err(AllocationAcquisitionError::Unsupported.into());
        }
        Ok(entry)
    }

    /// Authenticate an empty carrier without exposing a physical BufferEntry
    /// whose length could accidentally authorize work on its retained backing.
    pub(in super::super) fn validate_empty_view(&self, handle: &GpuTensorHandle) -> Result<()> {
        self.check_identity(handle.allocation_identity())?;
        self.validate_view(handle.allocation_identity(), handle.view())?;
        let records = self.lock_records()?;
        let record = records
            .get(&handle.buffer_id())
            .ok_or(AllocationAcquisitionError::NotLive)?;
        if record.phase != Phase::Live
            || record.entry.is_none()
            || record.physical != *handle.physical()
            || !handle.shape().contains(&0)
        {
            return Err(AllocationAcquisitionError::Unsupported.into());
        }
        handle
            .view()
            .validate(&record.physical)
            .map_err(AllocationAcquisitionError::InvalidView)?;
        let lease = record
            .lease
            .as_ref()
            .and_then(WeakAllocationLease::acquire)
            .ok_or(AllocationAcquisitionError::NotLive)?;
        if !lease.shares_allocation_with(handle.allocation_lease()) {
            return Err(AllocationAcquisitionError::WrongOwner.into());
        }
        Ok(())
    }

    pub(in super::super) fn requires_logical_encoding(
        &self,
        handle: &GpuTensorHandle,
    ) -> Result<bool> {
        self.check_identity(handle.allocation_identity())?;
        let (logical, weak) = {
            let records = self.lock_records()?;
            let record = records
                .get(&handle.buffer_id())
                .ok_or(AllocationAcquisitionError::NotLive)?;
            if record.phase != Phase::Live {
                return Err(AllocationAcquisitionError::NotLive.into());
            }
            (
                record.logical_producer,
                record
                    .lease
                    .clone()
                    .ok_or(AllocationAcquisitionError::NotLive)?,
            )
        };
        // Acquiring a lease re-enters its owner; never do so under the table lock.
        // This metadata query must not allocate an operation entry's shape vector.
        let lease = weak.acquire().ok_or(AllocationAcquisitionError::NotLive)?;
        if !lease.shares_allocation_with(handle.allocation_lease()) {
            return Err(AllocationAcquisitionError::WrongOwner.into());
        }
        self.validate_view(handle.allocation_identity(), handle.view())?;
        Ok(logical)
    }

    fn check_identity(
        &self,
        identity: AllocationIdentity,
    ) -> Result<(), AllocationAcquisitionError> {
        if identity.device_id() != self.device_id {
            return Err(AllocationAcquisitionError::WrongOwner);
        }
        if self.is_lost() {
            return Err(AllocationAcquisitionError::NotLive);
        }
        let records = self
            .buffers
            .lock()
            .map_err(|_| AllocationAcquisitionError::NotLive)?;
        let record = records
            .get(&identity.allocation_id())
            .ok_or(AllocationAcquisitionError::NotLive)?;
        if identity.generation() != record.scope.map_or(1, |scope| scope.epoch()) {
            return Err(AllocationAcquisitionError::StaleGeneration);
        }
        Ok(())
    }

    pub(in super::super) fn acquire_tensor(
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
        let entry = record
            .entry
            .as_ref()
            .ok_or(AllocationRegistrationError::NotLive)?;
        entry
            .buffer
            .validate_creation_scope()
            .map_err(|_| AllocationRegistrationError::StaleGeneration)?;
        if identity.generation()
            != entry
                .buffer
                .creation_scope()
                .map_or(1, |scope| scope.epoch())
        {
            return Err(AllocationRegistrationError::StaleGeneration);
        }
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
        // Carrier admission permits numeric zero views without exposing the
        // retained payload. entry_for_handle separately requires complete
        // allocations before any existing kernel/transfer sees BufferEntry.
        // Nonempty prefixes remain unsupported even as carriers.
        if count != Some(record.physical.element_capacity()) && count != Some(0) {
            return Err(AllocationAcquisitionError::Unsupported);
        }
        if view.is_logical() && !record.logical_producer {
            return Err(AllocationAcquisitionError::Unsupported);
        }
        Ok(())
    }

    fn retire(&self, identity: AllocationIdentity) -> AllocationRetirementOutcome {
        if identity.device_id() != self.device_id {
            return AllocationRetirementOutcome::Uncertain {
                detail: "WGPU retirement owner identity mismatch".into(),
            };
        }
        {
            let records = match self.lock_records() {
                Ok(records) => records,
                Err(error) => {
                    return AllocationRetirementOutcome::RetainedFailure {
                        detail: error.to_string(),
                    }
                }
            };
            // A scoped reset already physically retired this monotonic id.
            if records.get(&identity.allocation_id()).is_none() {
                return AllocationRetirementOutcome::Retired;
            }
            if let Some(record) = records.get(&identity.allocation_id()) {
                if identity.generation() != record.scope.map_or(1, |scope| scope.epoch()) {
                    return AllocationRetirementOutcome::Uncertain {
                        detail: "WGPU retirement generation mismatch".into(),
                    };
                }
            }
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
