use runmat_accelerate_api::{
    AccelContextHandle, AccelContextKind, GpuTensorHandle, ProviderPrecision,
    SpawnHandleConcurrency, WgpuBufferRef, WgpuContextHandle,
};

use super::{NumericPrecision, WgpuProvider};

const WGPU_SPAWN_HANDLE_CONCURRENCY: SpawnHandleConcurrency = SpawnHandleConcurrency::Reject;

impl WgpuProvider {
    pub(crate) fn device_id_exec(&self) -> u32 {
        self.runtime_device_id
    }

    pub(crate) fn export_context_exec(&self, kind: AccelContextKind) -> Option<AccelContextHandle> {
        match kind {
            AccelContextKind::Plotting => Some(AccelContextHandle::Wgpu(WgpuContextHandle {
                instance: self.instance.clone(),
                device: self.device.clone(),
                queue: self.queue.clone(),
                adapter: self.adapter.clone(),
                adapter_info: self.adapter_info.clone(),
                limits: self.adapter_limits.clone(),
                features: self.device.features(),
            })),
        }
    }

    #[cfg(feature = "wgpu")]
    pub(crate) fn export_wgpu_buffer_exec(
        &self,
        handle: &GpuTensorHandle,
    ) -> Option<WgpuBufferRef> {
        self.get_entry(handle).ok().map(|entry| WgpuBufferRef {
            buffer: entry.buffer,
            len: entry.len,
            shape: entry.shape,
            element_size: self.element_size,
            precision: match entry.precision {
                NumericPrecision::F32 => ProviderPrecision::F32,
                NumericPrecision::F64 => ProviderPrecision::F64,
            },
        })
    }

    pub(crate) fn provider_precision_exec(&self) -> ProviderPrecision {
        match self.precision {
            NumericPrecision::F32 => ProviderPrecision::F32,
            NumericPrecision::F64 => ProviderPrecision::F64,
        }
    }

    pub(crate) fn spawn_handle_concurrency_exec(&self) -> SpawnHandleConcurrency {
        WGPU_SPAWN_HANDLE_CONCURRENCY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_spawn_handle_concurrency_rejects_handle_capture() {
        assert_eq!(
            WGPU_SPAWN_HANDLE_CONCURRENCY,
            SpawnHandleConcurrency::Reject
        );
    }
}
