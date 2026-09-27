//! Fresh argument uploads belong to one preparation/call until adopted by its result.
use super::*;
use runmat_accelerate_api::GpuTensorHandle;
use std::collections::BTreeSet;

pub(super) struct UploadCleanupFailure {
    failed: usize,
    first: (u32, u64),
}
impl UploadCleanupFailure {
    pub(super) fn report(&self, phase: &'static str) {
        // Fixed schema and bounded scalar fields, not arbitrary backend text.
        log::warn!(target: "runmat::placement::cleanup",
            "event=prepared_operand_retirement_failed phase={} device_id={} buffer_id={} failed_allocations={}",
            phase, self.first.0, self.first.1, self.failed);
    }
}

pub(super) struct OwnedUploads {
    provider: &'static dyn AccelProvider,
    borrowed: BTreeSet<(u32, u64)>,
    borrowed_handles: Vec<GpuTensorHandle>,
    owned: Vec<GpuTensorHandle>,
}

impl OwnedUploads {
    pub(super) fn new(
        provider: &'static dyn AccelProvider,
        args: &[Value],
    ) -> PreparationResult<Self> {
        let mut borrowed = BTreeSet::new();
        let mut borrowed_handles = Vec::new();
        for value in args {
            runmat_runtime::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
                if borrowed.insert((handle.device_id, handle.buffer_id)) {
                    borrowed_handles.push(handle.clone());
                }
            })?;
        }
        Ok(Self {
            provider,
            borrowed,
            borrowed_handles,
            owned: Vec::new(),
        })
    }

    pub(super) fn protected(&self) -> Vec<&GpuTensorHandle> {
        self.borrowed_handles.iter().chain(&self.owned).collect()
    }

    pub(super) fn track(&mut self, value: &Value) {
        runmat_runtime::dispatcher::visit_gpu_handles(value, &mut |handle| {
            let identity = (handle.device_id, handle.buffer_id);
            if !self.borrowed.contains(&identity)
                && !self
                    .owned
                    .iter()
                    .any(|owned| (owned.device_id, owned.buffer_id) == identity)
            {
                self.owned.push(handle.clone());
            }
        });
    }

    pub(super) fn adopt(&mut self, value: &Value) -> PreparationResult<()> {
        if self.owned.is_empty() {
            return Ok(());
        }
        let mut retained = BTreeSet::new();
        runmat_runtime::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
            retained.insert((handle.device_id, handle.buffer_id));
        })?;
        self.owned
            .retain(|handle| !retained.contains(&(handle.device_id, handle.buffer_id)));
        Ok(())
    }

    pub(super) fn transfer_all(mut self) {
        self.owned.clear();
    }

    /// Attempt each remaining retirement once. A failed free may already have
    /// released backend storage, so drain it even on error: Drop must not retry.
    /// Failure persistence/teardown remains the captured provider's authority.
    pub(super) fn release(&mut self) -> std::result::Result<(), UploadCleanupFailure> {
        let mut failure: Option<UploadCleanupFailure> = None;
        for handle in self.owned.drain(..) {
            if self.provider.free(&handle).is_err() {
                let failure = failure.get_or_insert(UploadCleanupFailure {
                    failed: 0,
                    first: (handle.device_id, handle.buffer_id),
                });
                failure.failed += 1; // bounded by the original owned Vec length
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for OwnedUploads {
    fn drop(&mut self) {
        // Invocation observation is already unsuccessful on abandonment. Never
        // mutate an ambient recorder here: it may belong to another invocation.
        if let Err(error) = self.release() {
            error.report("abandonment");
        }
    }
}
