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
                access: self.access.clone(),
                adapter: self.adapter.clone(),
                adapter_info: self.adapter_info.clone(),
                limits: self.adapter_limits.clone(),
                features: self.enabled_features,
            })),
        }
    }

    #[cfg(feature = "wgpu")]
    pub(crate) fn export_wgpu_buffer_exec(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<WgpuBufferRef>> {
        use runmat_accelerate_api::ProviderExecutionError;
        let mut entry = self.get_entry_for_bitwise_move(handle, "WGPU export")?;
        if entry.integer_type().is_some()
            || entry.storage != runmat_accelerate_api::GpuTensorStorage::Real
        {
            return Ok(None);
        }
        if let Some(info) = runmat_accelerate_api::handle_transpose_info(handle) {
            // Validate the full view before any device dispatch. This local
            // entry clone changes no borrowed handle or allocation-map record.
            if entry.shape.as_slice() != [info.base_cols, info.base_rows]
                || info.base_rows.checked_mul(info.base_cols) != Some(entry.len)
            {
                return Err(anyhow::anyhow!("WGPU export: malformed transpose view").into());
            }
            entry.shape = vec![info.base_rows, info.base_cols];
            let allocated = std::cell::Cell::new(false);
            let result =
                self.permute_buffer_optional_exec(&entry, &[1, 0], &allocated, &|_| Ok(()));
            entry = match result {
                Ok(Some(entry)) => entry,
                Ok(None) => return Ok(None),
                Err(error) => {
                    let error = ProviderExecutionError::from(error);
                    return Err(if allocated.get() {
                        error.after_effect()
                    } else {
                        error
                    });
                }
            };
            let materialized = self
                .register_storage_move_buffer(
                    entry.buffer.clone(),
                    &entry,
                    entry.shape.clone(),
                    entry.len,
                    entry.allocated_bytes,
                )
                .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
            return WgpuBufferRef::from_owned_buffer(
                entry.buffer,
                materialized,
                self.access.clone(),
            )
            .map(Some)
            .map_err(|error| ProviderExecutionError::from(error).after_effect());
        }
        WgpuBufferRef::from_owned_buffer(entry.buffer, handle.clone(), self.access.clone())
            .map(Some)
            .map_err(ProviderExecutionError::from)
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
