use super::*;
use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::allocation::{AllocationLease, GpuTensorView};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, NumericElementType};

/// Owns only a not-yet-published record in the existing payload map.
/// Publishing transfers cleanup to the ordinary allocation lease; failure or
/// unwind detaches this pending payload before its destructor runs.
pub(in crate::simple_provider) struct PendingAllocation {
    owner: NumericOwner,
    identity: AllocationIdentity,
    view: Option<GpuTensorView>,
}

impl NumericAllocationState {
    pub(in crate::simple_provider) fn stage(
        self: &NumericOwner,
        identity: AllocationIdentity,
        payload: InProcessNumericData,
        shape: Vec<usize>,
        storage: GpuTensorStorage,
        producer_logical: bool,
    ) -> Result<PendingAllocation> {
        let element = match &payload {
            InProcessNumericData::F64(_) => NumericElementType::F64,
            InProcessNumericData::F32(_) => NumericElementType::F32,
            InProcessNumericData::Integer(data) => NumericElementType::from(data.element_type()),
        };
        let count = if shape.contains(&0) {
            0
        } else {
            shape
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .ok_or_else(|| anyhow!("in-process allocation shape overflow"))?
        };
        let lanes = if storage == GpuTensorStorage::ComplexInterleaved {
            2
        } else {
            1
        };
        ensure!(
            count.checked_mul(lanes) == Some(payload.len()),
            "in-process allocation shape does not match typed payload length"
        );
        let physical = GpuPhysicalDescriptor::new(
            element,
            storage,
            count as u64,
            payload.byte_len(),
            payload.requested_byte_len(),
        )?;
        let view = GpuTensorView::contiguous(shape).with_logical(producer_logical);
        view.validate(&physical)?;
        {
            let mut records = self
                .allocations
                .lock()
                .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
            ensure!(
                !records.contains_key(&identity.allocation_id()),
                "in-process allocation identity already exists"
            );
            records.insert(
                identity.allocation_id(),
                Record {
                    identity,
                    physical,
                    producer_logical,
                    payload: Some(payload),
                    phase: Phase::Pending,
                },
            );
        }
        Ok(PendingAllocation {
            owner: self.clone(),
            identity,
            view: Some(view),
        })
    }
}

impl PendingAllocation {
    pub(in crate::simple_provider) fn publish(mut self) -> Result<GpuTensorHandle> {
        let lease = AllocationLease::register(self.identity, self.owner.clone())?;
        // Take is infallible and immediately disarms pending cleanup. The live
        // lease already owns cleanup if subsequent view admission rejects.
        let view = self
            .view
            .take()
            .ok_or_else(|| anyhow!("in-process publication already consumed"))?;
        GpuTensorHandle::from_allocation(lease, view).map_err(Into::into)
    }
}

impl Drop for PendingAllocation {
    fn drop(&mut self) {
        if self.view.is_none() {
            return;
        }
        let detached = {
            let mut records = self
                .owner
                .allocations
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if records
                .get(&self.identity.allocation_id())
                .is_some_and(|record| {
                    record.identity == self.identity && matches!(record.phase, Phase::Pending)
                })
            {
                records.remove(&self.identity.allocation_id())
            } else {
                None
            }
        };
        drop(detached);
    }
}
