use super::*;

impl AllocationState {
    // Invoked only by the captured final physical lease, never a borrowed view.
    pub(super) fn retire_physical(&self, id: u64) -> Result<()> {
        let (mut entry, resource_ticket) = {
            let mut records = self.lock_records()?;
            let Some(record) = records.get_mut(&id) else {
                return Ok(());
            };
            if record.phase != Phase::Live {
                return Err(AllocationStateError::NotLive.into());
            }
            record.phase = Phase::Retiring(Stage::Cache);
            (record.entry.take(), record.resource_ticket.clone())
        };
        let stage = std::cell::Cell::new(Stage::Cache);
        let failure = std::cell::Cell::new(Failure::Cache);
        let disposition = std::cell::Cell::new(Disposition::Retained);
        // Keep the ticket outside the caught closure. A callback unwind cannot
        // accidentally destroy the payload before its failed phase is recorded.
        let result = catch_allocation_cleanup_unwind(|| -> Result<()> {
            let payload = entry.as_ref().ok_or(AllocationStateError::NotLive)?;
            if let Some(cache) = self.reduction_reuse.get().and_then(OwnerWeak::upgrade) {
                let identity = runmat_accelerate_api::allocation::AllocationIdentity::new(
                    self.device_id,
                    id,
                    1,
                );
                let detached = match cache.lock() {
                    Ok(mut reuse) => reuse.retire_allocation(identity),
                    Err(poisoned) => {
                        log::error!(
                            "WGPU reduction reuse cache poisoned during allocation retirement; discarding optional reuse"
                        );
                        poisoned.into_inner().detach_all()
                    }
                };
                if let Err(error) = super::super::moments_cache::drain_detached(detached) {
                    // The released handle's captured owner records its own
                    // disposition. Optional reuse cannot strand this source.
                    log::error!("WGPU cached output release failed: {error}");
                }
            }
            self.bind_groups
                .invalidate_buffer(
                    runmat_accelerate_api::WgpuBufferOwner::as_ptr(&payload.buffer) as usize,
                )?;
            failure.set(Failure::MapPoisoned);
            self.retirement_stage(id, Stage::Provenance)?;
            stage.set(Stage::Provenance);
            failure.set(Failure::Provenance);
            if let Some(resources) = self.kernel_resources.upgrade() {
                resources.clear_matmul_source_checked(id)?;
            }
            failure.set(Failure::MapPoisoned);
            // Per-view residency cleanup already belongs to the final view
            // owner. Physical retirement must not repeat those callbacks.
            self.retirement_stage(id, Stage::Resource)?;
            stage.set(Stage::Resource);
            // Once consumed, an unwind is uncertain release, never retryable.
            disposition.set(Disposition::Uncertain);
            failure.set(Failure::Unwind);
            if let Some(payload) = entry.take() {
                if self.release_payload(payload) {
                    if let Some(ticket) = &resource_ticket {
                        ticket.mark_pooled_handoff();
                    }
                }
            }
            disposition.set(Disposition::Released);
            stage.set(Stage::Complete);
            failure.set(Failure::MapPoisoned);
            let (removed, empty_table) = {
                let mut records = self.lock_records()?;
                let removed = records.remove(&id);
                let empty_table = records.detach_empty_managed();
                (removed, empty_table)
            };
            drop(removed);
            drop(empty_table);
            Ok(())
        });
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => {
                self.fail(id, stage.get(), failure.get(), disposition.get(), entry);
                Err(error)
            }
            Err(_) => {
                self.fail(id, stage.get(), Failure::Unwind, disposition.get(), entry);
                Err(AllocationStateError::Cleanup {
                    stage: stage.get(),
                    failure: Failure::Unwind,
                }
                .into())
            }
        }
    }

    fn retirement_stage(&self, id: u64, stage: Stage) -> Result<()> {
        self.lock_records()?
            .get_mut(&id)
            .ok_or(AllocationStateError::Missing)?
            .phase = Phase::Retiring(stage);
        Ok(())
    }

    pub(super) fn release_payload(&self, entry: BufferEntry) -> bool {
        let poolable = !self.is_lost()
            && entry.len > 0
            && self.max_poolable_bytes > 0
            && entry.allocated_bytes <= self.max_poolable_bytes
            && runmat_accelerate_api::WgpuBufferOwner::strong_count(&entry.buffer) == 1;
        if poolable {
            // A pool rejection returns this reference for ordinary destruction;
            // optional reuse failure is not device retirement failure.
            return match self
                .pool
                .release(entry.usage, entry.allocated_bytes, entry.buffer)
            {
                Ok(()) => true,
                Err(rejected) => {
                    drop(rejected.buffer);
                    false
                }
            };
        }
        // For non-poolable entries, ordinary field destruction releases the Arc.
        false
    }
}

impl Drop for AllocationState {
    fn drop(&mut self) {
        // This is final owner teardown, not a retry service. Detach everything
        // even if bookkeeping was poisoned, then dispose resource references
        // individually without any map/cache/pool lock held. Failed records are
        // reachable while the owner lives; they are not claimed durable after
        // this terminal disposition. WGPU has no driver-pointer free to retry.
        let records = std::mem::take(self.buffers.get_mut().unwrap_or_else(|e| e.into_inner()));
        let (records, _table_claim) = records.into_records();
        for (id, record) in records {
            let Some(entry) = record.entry else {
                if matches!(
                    record.phase,
                    Phase::Failed {
                        disposition: Disposition::Uncertain,
                        ..
                    }
                ) {
                    let _ = catch_allocation_cleanup_unwind(|| {
                        log::error!("WGPU terminal allocation id={id}: reference-release outcome remains uncertain; no resource retry attempted");
                    });
                }
                continue;
            };
            let cache = catch_allocation_cleanup_unwind(|| {
                self.bind_groups
                    .invalidate_buffer(
                        runmat_accelerate_api::WgpuBufferOwner::as_ptr(&entry.buffer) as usize,
                    )
            });
            let provenance = catch_allocation_cleanup_unwind(|| -> Result<()> {
                if let Some(resources) = self.kernel_resources.upgrade() {
                    resources.clear_matmul_source_checked(id)?;
                }
                Ok(())
            });
            let bookkeeping_complete =
                matches!(cache, Ok(Ok(()))) && matches!(provenance, Ok(Ok(())));
            let release = catch_allocation_cleanup_unwind(|| drop(entry));
            if !bookkeeping_complete || release.is_err() {
                // Reporting is best effort and cannot turn Drop into a second
                // unwind. A failed annotation/cache cleanup is not device loss.
                let _ = catch_allocation_cleanup_unwind(|| {
                    log::error!("WGPU terminal allocation disposition id={id}: bookkeeping_complete={} reference_release_completed={}",
                        bookkeeping_complete, release.is_ok());
                });
            }
        }
    }
}
