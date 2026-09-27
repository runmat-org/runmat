//! Rollback for fresh registrations within one provider operation. Borrowed
//! operand IDs are never inserted into this scoped owner.
use super::allocation_state::AllocationState;
use anyhow::{anyhow, Result};
use runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind;
use runmat_accelerate_api::GpuTensorHandle;

pub(super) struct RegistrationBatch<'a> {
    owner: &'a AllocationState,
    handles: Vec<PendingOutput>,
}

struct PendingOutput {
    handle: GpuTensorHandle,
    published: bool,
}

impl<'a> RegistrationBatch<'a> {
    pub(super) fn new(owner: &'a AllocationState) -> Self {
        Self {
            owner,
            handles: Vec::new(),
        }
    }

    pub(super) fn track(&mut self, registered: Result<GpuTensorHandle>) -> Result<GpuTensorHandle> {
        let handle = registered?;
        if self.handles.iter().any(|owned| {
            owned.handle.device_id == handle.device_id && owned.handle.buffer_id == handle.buffer_id
        }) {
            return Ok(handle);
        }
        if self.handles.try_reserve(1).is_err() {
            self.owner.free(&handle)?;
            return Err(anyhow!("WGPU registration rollback capacity unavailable"));
        }
        self.handles.push(PendingOutput {
            handle: handle.clone(),
            published: false,
        });
        Ok(handle)
    }

    pub(super) fn publish(&mut self, handle: &GpuTensorHandle) {
        if let Some(owned) = self.handles.iter_mut().find(|owned| {
            owned.handle.device_id == handle.device_id && owned.handle.buffer_id == handle.buffer_id
        }) {
            owned.published = true;
        }
    }

    pub(super) fn finish<T>(
        mut self,
        result: Result<T>,
        publish: impl FnOnce(&mut Self, &T),
    ) -> Result<T> {
        if let Ok(output) = &result {
            publish(&mut self, output);
            while let Some(index) = self.handles.iter().position(|owned| !owned.published) {
                let owned = self.handles.swap_remove(index);
                self.owner.free(&owned.handle)?;
            }
            // Disarm only after checked intermediate cleanup succeeded. If it
            // fails, Drop also retires the still-unpublished result handles.
            self.handles.clear();
        }
        // Intermediates still tracked here are cleaned even on success.
        result
    }
}

impl Drop for RegistrationBatch<'_> {
    fn drop(&mut self) {
        for owned in self.handles.drain(..).rev() {
            let result = catch_allocation_cleanup_unwind(|| self.owner.free(&owned.handle));
            if !matches!(result, Ok(Ok(()))) {
                // The checked owner retains its precise failed disposition;
                // do not retry or erase it merely because the operation failed.
                let _ = catch_allocation_cleanup_unwind(|| {
                    log::error!(
                        "WGPU registration rollback incomplete for buffer {}",
                        owned.handle.buffer_id
                    );
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::cache::bind_group::BindGroupCache;
    use std::sync::{Arc, Weak};

    #[test]
    fn publication_selects_only_the_exact_registered_identity() {
        let device = runmat_accelerate_api::next_device_id();
        let owner = AllocationState::new(
            device,
            1,
            1024,
            Arc::new(BindGroupCache::new()),
            Weak::new(),
        );
        let first = GpuTensorHandle::new(vec![1], device, 1);
        let intermediate = GpuTensorHandle::new(vec![1], device, 2);
        let foreign = GpuTensorHandle::new(vec![1], device.wrapping_add(1), 1);
        let mut batch = RegistrationBatch::new(&owner);
        batch.track(Ok(first.clone())).unwrap();
        batch.track(Ok(first.clone())).unwrap();
        batch.track(Ok(intermediate)).unwrap();
        assert_eq!(
            batch.handles.len(),
            2,
            "owned intermediate aliases are tracked once"
        );
        batch.publish(&foreign);
        assert!(batch.handles.iter().all(|entry| !entry.published));
        batch.publish(&first);
        assert!(batch.handles[0].published);
        assert!(!batch.handles[1].published);
        let output = batch
            .finish(Ok(first.clone()), |batch, output| batch.publish(output))
            .unwrap();
        assert_eq!(output, first);
        assert_eq!(owner.record_count().unwrap(), 0);
    }
}
