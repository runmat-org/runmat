use super::*;
use runmat_accelerate_api::allocation::AllocationCleanupPanicked;
use runmat_accelerate_api::allocation::{
    AllocationIdentity, AllocationLease, AllocationRegistrationError, GpuPhysicalDescriptor,
    GpuTensorView,
};

#[derive(Debug)]
pub(super) struct PublicationRegistrationFailure {
    pub(super) primary: AllocationRegistrationError,
    pub(super) cleanup_panicked: Option<AllocationCleanupPanicked>,
}
impl std::fmt::Display for PublicationRegistrationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WGPU lease registration rejected: {}", self.primary)?;
        if self.cleanup_panicked.is_some() {
            f.write_str("; unpublished payload cleanup unwound")?;
        }
        Ok(())
    }
}
impl std::error::Error for PublicationRegistrationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.primary)
    }
}

impl AllocationState {
    /// Only provider kernels/uploads which produce canonical boolean encoding
    /// may request logical classification. No public view flag grants it.
    pub(in super::super) fn register_produced(
        self: &super::super::owner_shared::OwnerShared<Self>,
        entry: BufferEntry,
        logical: bool,
    ) -> Result<GpuTensorHandle> {
        self.register_produced_with(
            entry,
            logical,
            |_| {},
            |entry| {
                let _ = self.release_payload(entry);
            },
        )
    }

    /// The seam is private to the owner; tests inject the registration race and
    /// cleanup unwind without changing the public allocation ABI.
    fn register_produced_with(
        self: &super::super::owner_shared::OwnerShared<Self>,
        entry: BufferEntry,
        logical: bool,
        after_insert: impl FnOnce(u64),
        cleanup: impl FnOnce(BufferEntry),
    ) -> Result<GpuTensorHandle> {
        if self.is_lost() {
            return Err(AllocationStateError::DeviceLost.into());
        }
        let lanes = match entry.storage {
            runmat_accelerate_api::GpuTensorStorage::Real => 1,
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved => 2,
        };
        anyhow::ensure!(
            entry.len.is_multiple_of(lanes),
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
                    resource_ticket: None,
                    logical_producer: logical,
                },
            );
        }
        after_insert(id);
        let identity = AllocationIdentity::new(self.device_id, id, 1);
        let lease = match AllocationLease::register(identity, self.clone()) {
            Ok(lease) => lease,
            Err(error) => {
                // No lease was armed. Remove its unpublished record under the
                // map lock, then dispose the payload without holding that lock.
                let unpublished = {
                    let mut records = self
                        .buffers
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    records.remove(&id)
                };
                let cleanup_panicked = unpublished
                    .and_then(|record| record.entry)
                    .and_then(|entry| catch_allocation_cleanup_unwind(|| cleanup(entry)).err());
                return Err(PublicationRegistrationFailure {
                    primary: error,
                    cleanup_panicked,
                }
                .into());
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
        let mut released_ticket = None;
        let mut failed_ticket = None;
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
                    Disposition::Released => {
                        record.retained_bytes = 0;
                        released_ticket = record.resource_ticket.take();
                    }
                    Disposition::Uncertain => {
                        record.retained_bytes = u64::MAX;
                        failed_ticket = record.resource_ticket.clone();
                    }
                    Disposition::Retained => {
                        failed_ticket = record.resource_ticket.clone();
                    }
                }
            }
        }
        if let Some(ticket) = &failed_ticket {
            ticket.fail_owner();
        }
        drop(released_ticket);
        let _ = catch_allocation_cleanup_unwind(|| drop(detached));
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod registration_failure_tests {
    use super::*;
    use crate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
    use crate::backend::wgpu::residency::BufferUsageClass;
    use crate::backend::wgpu::types::NumericPrecision;
    use runmat_accelerate_api::allocation::AllocationOwner;
    use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};
    use std::cell::Cell;

    fn rejected_after_insert(cleanup_unwinds: bool) {
        let provider = WgpuProvider::new(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required; unavailable is not a pass");
        let buffer = provider
            .with_emission(|emission, _| {
                provider.create_storage_buffer_checked_with_element_size(
                    emission,
                    1,
                    "registration-failure-witness",
                    BufferUsageClass::FusionOut,
                    4,
                )
            })
            .unwrap()
            .unwrap();
        let allocated_bytes = buffer.size();
        let entry = BufferEntry {
            logical: false,
            buffer,
            len: 1,
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
            element_type: NumericElementType::F32,
            precision: NumericPrecision::F32,
            allocated_bytes,
            usage: BufferUsageClass::FusionOut,
            last_submission_id: None,
        };
        let before = provider.allocations.record_count().unwrap();
        let id = Cell::new(0u64);
        let cleaned = Cell::new(false);
        let failure = provider
            .allocations
            .register_produced_with(
                entry,
                false,
                |inserted| {
                    id.set(inserted);
                    provider.allocations.mark_device_lost();
                },
                |payload| {
                    cleaned.set(true);
                    drop(payload);
                    if cleanup_unwinds {
                        panic!("injected cleanup unwind");
                    }
                },
            )
            .unwrap_err();
        let typed = failure
            .downcast_ref::<PublicationRegistrationFailure>()
            .expect("typed registration and cleanup failure");
        assert_eq!(typed.primary, AllocationRegistrationError::NotLive);
        assert!(failure
            .chain()
            .any(|cause| cause.downcast_ref::<AllocationRegistrationError>()
                == Some(&AllocationRegistrationError::NotLive)));
        assert_eq!(typed.cleanup_panicked.is_some(), cleanup_unwinds);
        assert!(cleaned.get());
        assert_ne!(id.get(), 0);
        assert_eq!(provider.allocations.record_count().unwrap(), before);
        assert_eq!(provider.allocations.retained_bytes(), 0);
        assert!(!provider
            .allocations
            .buffers
            .lock()
            .unwrap()
            .contains_key(&id.get()));
        assert!(provider
            .allocations
            .validate_live(AllocationIdentity::new(provider.device_id(), id.get(), 1),)
            .is_err());
    }

    #[test]
    #[ignore = "requires actual WGPU adapter; unavailable is failure"]
    fn failed_lease_registration_releases_unpublished_payload() {
        rejected_after_insert(false);
    }

    #[test]
    #[ignore = "requires actual WGPU adapter; unavailable is failure"]
    fn cleanup_unwind_is_typed_and_does_not_orphan_record() {
        rejected_after_insert(true);
    }
}
