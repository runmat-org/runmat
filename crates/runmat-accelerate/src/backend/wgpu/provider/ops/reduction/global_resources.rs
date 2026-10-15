//! Pure physical-resource admission for the selected global extrema tree.
use super::super::backend_types::WgpuProvider;
use super::super::validated_fusion::ReductionExtremaPipelineKind;
use crate::backend::wgpu::residency::BufferResidency;
use runmat_accelerate_api::unsupported_provider_operation;
use runmat_accelerate_api::{
    AccelProvider, ProviderElementType, ProviderExecutionError, ProviderExecutionResult,
    ProviderGlobalExtremaOutputRoute, ProviderGlobalExtremaPhysicalProposal,
    ProviderGlobalExtremaRequest, ProviderLayout, ProviderResidency, ProviderStorage,
};

impl WgpuProvider {
    pub(crate) fn propose_global_extrema_exec(
        &self,
        request: &ProviderGlobalExtremaRequest,
    ) -> ProviderExecutionResult<Option<ProviderGlobalExtremaPhysicalProposal>> {
        if request.source.storage != ProviderStorage::DenseReal
            || request.source.residency != ProviderResidency::Device
            || !matches!(
                request.source.layout,
                ProviderLayout::ColumnMajorContiguous | ProviderLayout::Strided
            )
            || request.source.element_type != ProviderElementType::from(self.precision())
            || [
                "RUNMAT_DISABLE_REDUCE_GLOBAL",
                "RUNMAT_PROVIDER_REDUCTION_SNAPSHOT",
                "RUNMAT_FORCE_REDUCE_SNAPSHOT",
            ]
            .iter()
            .any(|flag| std::env::var(flag).is_ok())
        {
            return Ok(None);
        }
        let Some(len) = request
            .source
            .checked_element_count()
            .and_then(|n| usize::try_from(n).ok())
        else {
            return Ok(None);
        };
        if len == 0 {
            return Ok(None);
        }
        self.require_ieee_words(self.precision)
            .map_err(ProviderExecutionError::from)?;
        self.pipelines
            .reduce_global
            .preflight()
            .map_err(ProviderExecutionError::from)?;
        admit(
            len,
            request.source_buffer_bytes,
            self.element_size,
            self.global_reduce_launch.elements_per_group(),
            self.global_reduce_launch.max_input_per_pass(),
            u64::from(self.adapter_limits.max_storage_buffer_binding_size),
            self.adapter_limits.max_buffer_size,
        )
        .map_err(ProviderExecutionError::from)?;
        let uniform_bytes =
            std::mem::size_of::<crate::backend::wgpu::params::ReduceGlobalParams>() as u64;
        if self.adapter_limits.max_storage_buffers_per_shader_stage < 2
            || self.adapter_limits.max_uniform_buffers_per_shader_stage < 1
            || u64::from(self.adapter_limits.max_uniform_buffer_binding_size) < uniform_bytes
        {
            return Ok(None);
        }
        let mut current = len;
        let mut allocated = 0_u64;
        let mut output_bytes = request.source_buffer_bytes;
        let mut dispatches = 0_u32;
        let Some(chunk) = self.global_reduce_launch.max_input_per_pass() else {
            return Ok(None);
        };
        while current > 1 {
            let output = self.global_reduce_launch.output_len(current).max(1);
            let Some(bytes) = BufferResidency::requested_bytes(output, self.element_size) else {
                return Ok(None);
            };
            let Some(total) = allocated.checked_add(bytes) else {
                return Ok(None);
            };
            allocated = total;
            output_bytes = bytes;
            let Some(count) = u32::try_from(current.div_ceil(chunk))
                .ok()
                .and_then(|count| dispatches.checked_add(count))
            else {
                return Ok(None);
            };
            dispatches = count;
            current = output;
        }
        let Some(uniform_buffer_bytes) = uniform_bytes.checked_mul(u64::from(dispatches)) else {
            return Ok(None);
        };
        let intermediate_buffer_bytes = if len == 1 {
            0
        } else {
            let Some(bytes) = allocated.checked_sub(output_bytes) else {
                return Ok(None);
            };
            bytes
        };
        Ok(Some(ProviderGlobalExtremaPhysicalProposal {
            output_route: if len == 1 {
                ProviderGlobalExtremaOutputRoute::SourceScalarAlias
            } else {
                ProviderGlobalExtremaOutputRoute::FreshAllocation
            },
            output_buffer_bytes: output_bytes,
            intermediate_buffer_bytes,
            uniform_allocations: dispatches,
            uniform_buffer_bytes,
            dispatches,
            queue_submissions: dispatches,
            workgroup_size: self.global_reduce_launch.workgroup(),
            chunk_capacity: chunk as u64,
            pipeline_key: ReductionExtremaPipelineKind::Global.recipe(self).key(),
        }))
    }
}

pub(super) fn admit(
    len: usize,
    source_buffer_bytes: u64,
    element_bytes: usize,
    elements_per_group: usize,
    max_input_per_pass: Option<usize>,
    max_storage_binding_bytes: u64,
    max_buffer_bytes: u64,
) -> anyhow::Result<()> {
    let unsupported = || {
        unsupported_provider_operation(
            "global extrema source or intermediate exceeds physical resource limits",
        )
    };
    if element_bytes == 0 || elements_per_group < 2 || max_input_per_pass == Some(0) {
        return Err(unsupported());
    }
    let element_bytes = u64::try_from(element_bytes).map_err(|_| unsupported())?;
    if len == 0 {
        // The empty scalar is uploaded; no source or reduction pipeline is bound.
        return if element_bytes <= max_buffer_bytes {
            Ok(())
        } else {
            Err(unsupported())
        };
    }
    if len > u32::MAX as usize || max_input_per_pass.is_none() {
        return Err(unsupported());
    }
    let logical_bytes = u64::try_from(len)
        .ok()
        .and_then(|value| value.checked_mul(element_bytes))
        .ok_or_else(unsupported)?;
    if source_buffer_bytes < logical_bytes || source_buffer_bytes > max_buffer_bytes {
        return Err(unsupported());
    }
    if len == 1 {
        // The emitter returns a shape-only view of the same scalar allocation.
        return Ok(());
    }
    let first_output_elements = len.div_ceil(elements_per_group).max(1);
    let first_output_bytes = u64::try_from(first_output_elements)
        .ok()
        .and_then(|value| value.checked_mul(element_bytes))
        .ok_or_else(unsupported)?;
    if source_buffer_bytes > max_storage_binding_bytes
        || first_output_bytes > max_storage_binding_bytes
        || first_output_bytes > max_buffer_bytes
    {
        return Err(unsupported());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn exact_source_and_first_intermediate_boundaries() {
        let admitted = |len, source, binding, buffer| {
            admit(len, source, 4, 2, Some(16), binding, buffer).is_ok()
        };
        assert!(admitted(8, 32, 32, 32));
        assert!(!admitted(8, 36, 32, 36)); // Entire physical source is bound.
        assert!(!admitted(8, 32, 15, 32)); // Entire 32-byte source is bound.
        assert!(!admitted(8, 32, 32, 15)); // Source cannot fit max buffer.
        assert!(!admitted(8, 28, 32, 32)); // Logical source exceeds backing.
        assert!(admitted(1, 4, 0, 4)); // Singleton does not bind the source.
        assert!(admitted(0, 0, 0, 4)); // Empty scalar upload.
        assert!(!admitted(0, 0, 0, 3));
        assert!(admit(3, 12, 4, 2, Some(16), 12, 12).is_ok());
        assert!(admit(3, 12, 4, 2, None, 12, 12).is_err());
        assert!(admit(3, 12, 4, 0, Some(16), 12, 12).is_err());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn overwide_length_is_unsupported_before_any_allocation() {
        if let Ok(overwide) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(admit(overwide, u64::MAX, 4, 2, Some(16), u64::MAX, u64::MAX).is_err());
        }
    }
}
