use super::*;

impl AllocationState {
    pub(in super::super) fn register(&self, entry: BufferEntry) -> Result<GpuTensorHandle> {
        self.register_with_residency(entry, runmat_accelerate_api::clear_residency_checked)
    }

    pub(super) fn next_identifier(&self) -> Result<u64> {
        self.next_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| AllocationStateError::IdentifierExhausted.into())
    }

    pub(super) fn register_with_residency(
        &self,
        entry: BufferEntry,
        clear_residency: impl FnOnce(
            &GpuTensorHandle,
        ) -> std::result::Result<
            (),
            runmat_accelerate_api::allocation::AllocationCleanupPanicked,
        >,
    ) -> Result<GpuTensorHandle> {
        if self.is_lost() {
            return Err(AllocationStateError::DeviceLost.into());
        }
        // Burn failed IDs too. Exhaustion never wraps or replaces a prior view.
        let id = self.next_identifier()?;
        let handle = GpuTensorHandle {
            device_id: self.device_id,
            buffer_id: id,
            shape: entry.shape.clone(),
            descriptor: GpuTensorDescriptor::numeric(entry.element_type, entry.storage),
        };
        {
            let mut records = self.lock_records()?;
            records
                .try_reserve(1)
                .map_err(|_| AllocationStateError::Capacity)?;
            // IDs are monotonic and never exposed before the record is Live.
            if records.contains_key(&id) {
                return Err(AllocationStateError::IdentifierExhausted.into());
            }
            records.insert(
                id,
                Record {
                    phase: Phase::Registering(Stage::RegistrationResidency),
                    retained_bytes: entry.allocated_bytes,
                    entry: Some(entry),
                },
            );
        }
        if !matches!(
            catch_allocation_cleanup_unwind(|| clear_residency(&handle)),
            Ok(Ok(()))
        ) {
            self.fail(
                id,
                Stage::RegistrationResidency,
                Failure::Callback,
                Disposition::Retained,
                None,
            );
            return Err(AllocationStateError::Cleanup {
                stage: Stage::RegistrationResidency,
                failure: Failure::Callback,
            }
            .into());
        }
        if let Err(error) = self.registration_stage(id, Stage::RegistrationAnnotations) {
            self.fail(
                id,
                Stage::RegistrationAnnotations,
                Failure::MapPoisoned,
                Disposition::Retained,
                None,
            );
            return Err(error);
        }
        if let Err(error) = runmat_accelerate_api::clear_handle_annotations_checked(&handle) {
            self.fail(
                id,
                Stage::RegistrationAnnotations,
                Failure::AnnotationLock,
                Disposition::Retained,
                None,
            );
            return Err(error.into());
        }
        let publication: Result<()> = (|| {
            let mut records = self.lock_records()?;
            if self.is_lost() {
                return Err(AllocationStateError::DeviceLost.into());
            }
            let record = records.get_mut(&id).ok_or(AllocationStateError::Missing)?;
            record.phase = Phase::Live;
            Ok(())
        })();
        if let Err(error) = publication {
            self.fail(
                id,
                Stage::RegistrationAnnotations,
                if self.is_lost() {
                    Failure::DeviceLost
                } else {
                    Failure::MapPoisoned
                },
                Disposition::Retained,
                None,
            );
            return Err(error);
        }
        Ok(handle)
    }

    fn registration_stage(&self, id: u64, stage: Stage) -> Result<()> {
        self.lock_records()?
            .get_mut(&id)
            .ok_or(AllocationStateError::Missing)?
            .phase = Phase::Registering(stage);
        Ok(())
    }

    pub(super) fn fail(
        &self,
        id: u64,
        stage: Stage,
        failure: Failure,
        disposition: Disposition,
        entry: Option<BufferEntry>,
    ) {
        // Poison recovery is restricted to quarantine bookkeeping, never live
        // operations. The existing record remains reachable; no insert or
        // allocation is needed on this failure path.
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
        // Defensive orphan disposition is outside the map lock as well.
        let _ = catch_allocation_cleanup_unwind(|| drop(detached));
    }
}
