//! Frozen softmax-row input, shader, and launch facts before scoped creation.
use super::*;

pub(super) struct SoftmaxRowsFrame {
    pub(super) entry: super::super::backend_types::BufferEntry,
    pub(super) recipe: super::super::pipeline_recipe::PipelineRecipe,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) chunk_rows: usize,
    pub(super) logical_producer: bool,
}

pub(super) fn effective_softmax_chunk_rows(limits: &wgpu::Limits) -> Option<u32> {
    let rows = crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS
        .min(limits.max_compute_workgroups_per_dimension);
    (rows != 0).then_some(rows)
}

pub(super) fn softmax_rows_buffer_demand_for_limits(
    limits: &wgpu::Limits,
    elements: u64,
    rows: u64,
    element_size: usize,
    workgroup: u32,
) -> Option<u64> {
    let output = elements.max(1).checked_mul(element_size as u64)?;
    if elements == 0 {
        return Some(output);
    }
    if rows == 0 {
        return None;
    }
    check_softmax_workgroup_limits(limits, workgroup, element_size).ok()?;
    let chunks = rows.div_ceil(u64::from(effective_softmax_chunk_rows(limits)?));
    output.checked_add(chunks.checked_mul(std::mem::size_of::<SoftmaxRowsParams>() as u64)?)
}

fn check_softmax_workgroup_limits(
    limits: &wgpu::Limits,
    workgroup: u32,
    element_size: usize,
) -> Result<()> {
    ensure!(
        workgroup >= 2
            && workgroup.is_power_of_two()
            && workgroup <= limits.max_compute_workgroup_size_x
            && workgroup <= limits.max_compute_invocations_per_workgroup,
        "activation_softmax_rows: workgroup exceeds enabled limits"
    );
    let scratch_bytes = u64::from(workgroup)
        .checked_add(1)
        .and_then(|words| words.checked_mul(element_size as u64))
        .ok_or_else(|| {
            anyhow!("activation_softmax_rows: workgroup storage byte extent overflows")
        })?;
    ensure!(
        scratch_bytes <= u64::from(limits.max_compute_workgroup_storage_size),
        "activation_softmax_rows: workgroup storage exceeds enabled limits"
    );
    Ok(())
}

pub(super) fn check_softmax_limits(
    limits: &wgpu::Limits,
    input_bytes: u64,
    output_bytes: u64,
    workgroup: u32,
    element_size: usize,
) -> Result<usize> {
    super::super::backend_shared::validate_compute_binding_counts(
        "activation_softmax_rows",
        2,
        3,
        limits,
    )?;
    ensure!(
        limits.max_uniform_buffers_per_shader_stage >= 1
            && limits.max_uniform_buffer_binding_size
                >= std::mem::size_of::<SoftmaxRowsParams>() as u32,
        "activation_softmax_rows: uniform binding unavailable"
    );
    check_softmax_workgroup_limits(limits, workgroup, element_size)?;
    let maximum_bytes =
        u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
    ensure!(
        input_bytes <= maximum_bytes && output_bytes <= maximum_bytes,
        "activation_softmax_rows: storage binding exceeds enabled limits"
    );
    effective_softmax_chunk_rows(limits)
        .map(|rows| rows as usize)
        .ok_or_else(|| anyhow!("activation_softmax_rows: dispatch unavailable"))
}

impl SoftmaxRowsFrame {
    pub(super) fn plan(
        provider: &WgpuProvider,
        entry: super::super::backend_types::BufferEntry,
        logical_producer: bool,
    ) -> Result<Self> {
        ensure!(
            !logical_producer && provider.supports_native_real_element_type(entry.element_type),
            "activation_softmax_rows: frozen physical class is not native real floating"
        );
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        ensure!(
            rows.checked_mul(cols) == Some(entry.len),
            "activation_softmax_rows: input shape does not match storage length"
        );
        ensure!(
            entry.precision == provider.precision,
            "activation_softmax_rows: input precision does not match provider"
        );
        let output_bytes = (entry.len as u64)
            .checked_mul(provider.element_size as u64)
            .ok_or_else(|| anyhow!("activation_softmax_rows: output byte extent overflows"))?;
        let workgroup = provider.workgroup_config.scalar;
        let chunk_rows = check_softmax_limits(
            &provider.adapter_limits,
            entry.allocated_bytes,
            output_bytes,
            workgroup,
            provider.element_size,
        )
        .map_err(|error| {
            runmat_accelerate_api::unsupported_provider_operation(error.to_string())
        })?;
        Ok(Self {
            entry,
            recipe: super::softmax_rows_recipe(provider.precision, workgroup),
            rows,
            cols,
            chunk_rows,
            logical_producer,
        })
    }

    pub(super) fn require_live_input(
        &self,
        provider: &WgpuProvider,
        input: &GpuTensorHandle,
    ) -> Result<()> {
        let live = provider.get_entry(input)?;
        let live_logical_producer = provider.allocations.requires_logical_encoding(input)?;
        ensure!(
            runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live.buffer, &self.entry.buffer),
            "activation_softmax_rows: input changed during preparation"
        );
        ensure!(
            live_logical_producer == self.logical_producer && !live_logical_producer,
            "activation_softmax_rows: borrowed input producer class changed during preparation"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn softmax_limits_reject_bindings_before_driver_creation() {
        let limits = wgpu::Limits::default();
        let workgroup = 256;
        assert!(check_softmax_limits(&limits, 24, 24, workgroup, 4).is_ok());
        assert!(check_softmax_limits(&limits, 24, 24, 128, 4).is_ok());
        for invalid in [0, 1, 3] {
            assert!(check_softmax_limits(&limits, 24, 24, invalid, 4)
                .unwrap_err()
                .to_string()
                .contains("workgroup"));
        }
        let mut webgpu_minimum = limits.clone();
        webgpu_minimum.max_compute_workgroup_size_x = 256;
        webgpu_minimum.max_compute_invocations_per_workgroup = 256;
        assert!(check_softmax_limits(&webgpu_minimum, 24, 24, 256, 4).is_ok());
        assert!(check_softmax_limits(&webgpu_minimum, 24, 24, 512, 4).is_err());
        assert_ne!(
            softmax_rows_pipeline_key(NumericPrecision::F32, 256),
            softmax_rows_pipeline_key(NumericPrecision::F32, 512)
        );
        assert_eq!(
            softmax_rows_buffer_demand_for_limits(&webgpu_minimum, 10, 5, 4, 256),
            Some(40 + 16)
        );
        assert_eq!(
            softmax_rows_buffer_demand_for_limits(&webgpu_minimum, 10, 5, 4, 512),
            None
        );
        let mut tiny = limits.clone();
        tiny.max_storage_buffer_binding_size = 16;
        assert!(check_softmax_limits(&tiny, 24, 24, workgroup, 4)
            .unwrap_err()
            .to_string()
            .contains("storage binding"));
        let mut tiny_scratch = limits.clone();
        tiny_scratch.max_compute_workgroup_storage_size = workgroup * 4;
        assert!(check_softmax_limits(&tiny_scratch, 24, 24, workgroup, 4)
            .unwrap_err()
            .to_string()
            .contains("workgroup storage"));
        let mut reduced_dispatch = limits;
        reduced_dispatch.max_compute_workgroups_per_dimension = 2;
        assert_eq!(effective_softmax_chunk_rows(&reduced_dispatch), Some(2));
        assert_eq!(
            check_softmax_limits(&reduced_dispatch, 24, 24, workgroup, 4).unwrap(),
            2
        );
        assert_eq!(
            softmax_rows_buffer_demand_for_limits(&reduced_dispatch, 10, 5, 4, workgroup),
            Some(40 + 3 * 16)
        );
        assert_eq!(
            softmax_rows_buffer_demand_for_limits(&reduced_dispatch, 10, 0, 4, workgroup),
            None
        );
        reduced_dispatch.max_compute_workgroups_per_dimension = 0;
        assert_eq!(effective_softmax_chunk_rows(&reduced_dispatch), None);
        assert_eq!(
            softmax_rows_buffer_demand_for_limits(&reduced_dispatch, 10, 5, 4, workgroup),
            None
        );
        assert_eq!(
            softmax_rows_buffer_demand_for_limits(&reduced_dispatch, 0, 0, 4, workgroup),
            Some(4)
        );
    }
}

#[cfg(test)]
#[path = "softmax_validated/actual_tests.rs"]
mod actual_tests;
