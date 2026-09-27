use super::*;
use runmat_accelerate_api::allocation::{
    AllocationIdentity, AllocationLease, GpuPhysicalDescriptor, GpuTensorView,
};

impl AllocationState {
    /// Only provider kernels/uploads which produce canonical boolean encoding
    /// may request logical classification. No public view flag grants it.
    pub(in super::super) fn register_produced(
        self: &Arc<Self>,
        entry: BufferEntry,
        logical: bool,
    ) -> Result<GpuTensorHandle> {
        if self.is_lost() {
            return Err(AllocationStateError::DeviceLost.into());
        }
        let lanes = match entry.storage {
            runmat_accelerate_api::GpuTensorStorage::Real => 1,
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved => 2,
        };
        anyhow::ensure!(
            entry.len % lanes == 0,
            "WGPU allocation has incomplete complex lanes"
        );
        let capacity = (entry.len / lanes) as u64;
        let width = entry
            .integer_type()
            .map_or(entry.element_type.element_size() as u64, |kind| {
                crate::backend::wgpu_storage::integer_words_per_element(kind) as u64 * 4
            });
        let payload = (entry.len as u64)
            .checked_mul(width)
            .ok_or(AllocationStateError::Capacity)?;
        let physical = GpuPhysicalDescriptor::new(
            entry.element_type,
            entry.storage,
            capacity,
            payload,
            entry.buffer.size(),
        )?;
        let view = GpuTensorView::contiguous(entry.shape.clone()).with_logical(logical);
        view.validate(&physical)?;
        let id = self.next_identifier()?;
        {
            let mut records = self.lock_records()?;
            records
                .try_reserve(1)
                .map_err(|_| AllocationStateError::Capacity)?;
            if records.contains_key(&id) {
                return Err(AllocationStateError::IdentifierExhausted.into());
            }
            records.insert(
                id,
                Record {
                    phase: Phase::Registering(Stage::Registration),
                    retained_bytes: entry.buffer.size(),
                    entry: Some(entry),
                    physical,
                    lease: None,
                    logical_producer: logical,
                },
            );
        }
        let identity = AllocationIdentity::new(self.device_id, id, 1);
        let lease = match AllocationLease::register(identity, self.clone()) {
            Ok(lease) => lease,
            Err(error) => {
                // No lease is armed. Keep an observable unpublished failure
                // record; final owner teardown disposes its resource reference.
                self.fail(
                    id,
                    Stage::Registration,
                    if self.is_lost() {
                        Failure::DeviceLost
                    } else {
                        Failure::MapPoisoned
                    },
                    Disposition::Retained,
                    None,
                );
                return Err(anyhow!("WGPU lease registration rejected: {error:?}"));
            }
        };
        Ok(GpuTensorHandle::from_allocation(lease, view)?)
    }

    pub(super) fn next_identifier(&self) -> Result<u64> {
        self.next_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| AllocationStateError::IdentifierExhausted.into())
    }

    pub(super) fn fail(
        &self,
        id: u64,
        stage: Stage,
        failure: Failure,
        disposition: Disposition,
        entry: Option<BufferEntry>,
    ) {
        let mut detached = entry;
        {
            let mut records = self
                .buffers
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(record) = records.get_mut(&id) {
                if record.entry.is_none() {
                    record.entry = detached.take();
                }
                record.phase = Phase::Failed {
                    stage,
                    failure,
                    disposition,
                };
                match disposition {
                    Disposition::Released => record.retained_bytes = 0,
                    Disposition::Uncertain => record.retained_bytes = u64::MAX,
                    Disposition::Retained => {}
                }
            }
        }
        let _ = catch_allocation_cleanup_unwind(|| drop(detached));
    }
}
