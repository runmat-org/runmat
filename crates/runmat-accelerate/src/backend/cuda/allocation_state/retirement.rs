use super::context::ContextScope;
use super::*;

impl AllocationState {
    pub(in crate::backend::cuda) fn retire_record(&self, id: u64) -> Result<()> {
        self.ensure_available()?;
        let allocation = {
            let mut records = self.records();
            let record = records
                .get_mut(&id)
                .ok_or_else(|| anyhow!("CUDA handle is stale or already freed"))?;
            match &record.phase {
                Phase::Live | Phase::EntryFailed(_) => {}
                _ => {
                    return Err(anyhow!(
                        "CUDA allocation cannot be retired again: {:?}",
                        record.phase
                    ))
                }
            }
            let allocation = record
                .allocation
                .ok_or_else(|| anyhow!("CUDA allocation has no payload"))?;
            record.phase = Phase::Retiring;
            allocation
        };
        if let Err(error) = self.driver.push(self.context) {
            self.set_phase(id, Phase::EntryFailed(error.to_string()));
            return Err(error);
        }
        let scope = ContextScope {
            owner: self,
            active: true,
        };
        // Record uncertainty before calling the driver: even unwinding cannot
        // leave a record eligible for a second physical free.
        self.set_phase(id, Phase::Uncertain("driver free did not return".into()));
        let freed = self.driver.free(allocation.pointer);
        match &freed {
            Ok(()) => self.set_phase(id, Phase::FreedExitFailed("context exit pending".into())),
            Err(error) => self.set_phase(id, Phase::Uncertain(error.to_string())),
        }
        let leave = scope.finish();
        match (freed, leave) {
            (Ok(()), Ok(())) => {
                self.records().remove(&id);
                Ok(())
            }
            (Ok(()), Err(error)) => {
                self.set_phase(id, Phase::FreedExitFailed(error.to_string()));
                Err(error)
            }
            (Err(error), _) => Err(error),
        }
    }

    fn set_phase(&self, id: u64, phase: Phase) {
        if let Some(record) = self.records().get_mut(&id) {
            record.phase = phase;
        }
    }
}

impl Drop for AllocationState {
    fn drop(&mut self) {
        // Driver callbacks and exceptional diagnostics cannot unwind a state
        // finalizer. The driver's own destructor must remain nonpanicking.
        let _ = catch_allocation_cleanup_unwind(|| self.teardown_allocations());
        let _ = catch_allocation_cleanup_unwind(|| {
            if let Err(error) = self.driver.release(self.cuda_device) {
                log::error!("CUDA primary context release failed: {error}");
            }
        });
    }
}

impl AllocationState {
    fn teardown_allocations(&self) {
        let candidates = self
            .records()
            .iter()
            .filter_map(|(&id, record)| {
                matches!(record.phase, Phase::Live | Phase::EntryFailed(_)).then_some(id)
            })
            .collect::<Vec<_>>();
        for id in candidates {
            match catch_allocation_cleanup_unwind(|| self.retire_record(id)) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    log::error!("CUDA final state teardown did not retire allocation {id}: {error}")
                }
                Err(_) => {
                    *self.unavailable.lock().unwrap_or_else(|e| e.into_inner()) = Some(
                        context::ContextUnavailable("allocation cleanup unwound".into()),
                    );
                    log::error!("CUDA final state teardown unwound while retiring allocation {id}");
                }
            }
        }
        let unresolved = self
            .records()
            .iter()
            .map(|(id, record)| {
                let detail = match &record.phase {
                    Phase::EntryFailed(detail)
                    | Phase::Uncertain(detail)
                    | Phase::FreedExitFailed(detail) => detail.as_str(),
                    _ => "allocation was not retired",
                };
                format!(
                    "CUDA unresolved record at final state teardown {id}: {:?}: {detail}",
                    record.phase
                )
            })
            .collect::<Vec<_>>();
        for diagnostic in unresolved {
            log::error!("{diagnostic}");
        }
    }
}
