//! Fresh argument uploads belong to one preparation/call until adopted by its result.
use super::*;
use runmat_accelerate_api::GpuTensorHandle;
use std::collections::BTreeSet;

pub(super) struct OwnedUploads {
    provider: &'static dyn AccelProvider,
    borrowed: BTreeSet<(u32, u64)>,
    borrowed_handles: Vec<GpuTensorHandle>,
    owned: Vec<GpuTensorHandle>,
}

impl OwnedUploads {
    pub(super) fn new(provider: &'static dyn AccelProvider, args: &[Value]) -> Self {
        let mut borrowed = BTreeSet::new();
        let mut borrowed_handles = Vec::new();
        for value in args {
            runmat_runtime::dispatcher::visit_gpu_handles(value, &mut |handle| {
                if borrowed.insert((handle.device_id, handle.buffer_id)) {
                    borrowed_handles.push(handle.clone());
                }
            });
        }
        Self {
            provider,
            borrowed,
            borrowed_handles,
            owned: Vec::new(),
        }
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

    pub(super) fn adopt(&mut self, value: &Value) {
        let mut retained = BTreeSet::new();
        runmat_runtime::dispatcher::visit_gpu_handles(value, &mut |handle| {
            retained.insert((handle.device_id, handle.buffer_id));
        });
        self.owned
            .retain(|handle| !retained.contains(&(handle.device_id, handle.buffer_id)));
    }

    pub(super) fn transfer_all(mut self) {
        self.owned.clear();
    }
}

impl Drop for OwnedUploads {
    fn drop(&mut self) {
        for handle in self.owned.drain(..) {
            let _ = self.provider.free(&handle);
        }
    }
}
