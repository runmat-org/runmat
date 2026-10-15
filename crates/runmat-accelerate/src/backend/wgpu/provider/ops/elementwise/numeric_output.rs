//! Effect-free extent of a resident dense real output through the ordinary
//! typed fused-elementwise allocator. This is allocation only: no operation,
//! arithmetic-domain, kernel availability, or placement authority.

use super::*;
use crate::backend::wgpu::residency::BufferResidency;
use runmat_accelerate_api::{
    ProviderElementType, ProviderExecutionResult, ProviderLayout, ProviderRepresentation,
    ProviderResidency, ProviderStorage,
};

impl WgpuProvider {
    pub(crate) fn plan_dense_real_output_buffer_bytes_exec(
        &self,
        output: &ProviderRepresentation,
    ) -> ProviderExecutionResult<Option<u64>> {
        if output.storage != ProviderStorage::DenseReal
            || output.layout != ProviderLayout::ColumnMajorContiguous
            || output.residency != ProviderResidency::Device
        {
            return Ok(None);
        }
        let element_size = match output.element_type {
            ProviderElementType::F32 => 4,
            ProviderElementType::F64 if self.precision == NumericPrecision::F64 => 8,
            _ => return Ok(None),
        };
        let Some(elements) = output.checked_element_count() else {
            return Ok(None);
        };
        if elements > u64::from(u32::MAX) {
            return Ok(None);
        }
        let Some(len) = usize::try_from(elements).ok() else {
            return Ok(None);
        };
        let Some(bytes) = BufferResidency::requested_bytes(len, element_size) else {
            return Ok(None);
        };
        if bytes > self.adapter_limits.max_buffer_size
            || bytes > u64::from(self.adapter_limits.max_storage_buffer_binding_size)
        {
            return Ok(None);
        }
        Ok(Some(bytes))
    }
}

#[cfg(test)]
#[path = "numeric_output_tests.rs"]
mod tests;
