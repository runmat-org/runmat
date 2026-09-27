use runmat_accelerate_api::{
    AccelContextHandle, AccelContextKind, GpuTensorHandle, ProviderPrecision,
    SpawnHandleConcurrency, WgpuBufferRef, WgpuContextHandle,
};

use super::{NumericPrecision, WgpuProvider};

const WGPU_SPAWN_HANDLE_CONCURRENCY: SpawnHandleConcurrency = SpawnHandleConcurrency::Reject;

#[cfg(test)]
#[path = "context/export_tests.rs"]
mod export_tests;

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
        let mut entry = self
            .get_entry_for_bitwise_move(handle, "WGPU export")
            .ok()?;
        if entry.integer_type().is_some()
            || entry.storage != runmat_accelerate_api::GpuTensorStorage::Real
        {
            return None;
        }
        if let Some(info) = runmat_accelerate_api::handle_transpose_info(handle) {
            // Validate the full view before any device dispatch. This local
            // entry clone changes no borrowed handle or allocation-map record.
            if entry.shape.as_slice() != [info.base_cols, info.base_rows]
                || info.base_rows.checked_mul(info.base_cols) != Some(entry.len)
            {
                return None;
            }
            entry.shape = vec![info.base_rows, info.base_cols];
            entry = self.permute_buffer_exec(&entry, &[1, 0]).ok()?;
        }
        Some(WgpuBufferRef {
            buffer: entry.buffer,
            len: entry.len,
            shape: entry.shape,
            element_size: entry.element_type.element_size(),
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
