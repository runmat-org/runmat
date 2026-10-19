//! GPU modulation facts checked before upload or scoped driver creation.
use super::*;
use crate::backend::wgpu::residency::BufferResidency;

pub(super) const LAYOUT_TAG: &str = crate::backend::wgpu::bindings::MODULATION_LAYOUT_TAG;

pub(super) struct ModulationWork<'a> {
    pub(super) input: &'a super::super::backend_types::BufferEntry,
    pub(super) table_words: usize,
    pub(super) output_words: usize,
    pub(super) dispatch_len: usize,
    pub(super) workgroup: u32,
}

pub(super) fn validate_bit_group_width(
    bits_per_symbol: usize,
) -> runmat_accelerate_api::ProviderExecutionResult<()> {
    if bits_per_symbol > u32::BITS as usize {
        return Err(runmat_accelerate_api::unsupported_provider_operation(
            "modulate_bits_constellation: bit groups wider than the GPU symbol accumulator",
        )
        .into());
    }
    Ok(())
}

pub(super) fn decode_error_code(bytes: &[u8]) -> Result<u32> {
    ensure!(
        bytes.len() == 16,
        "modulation: mapped error payload size mismatch"
    );
    let first = u32::from_ne_bytes(bytes[..4].try_into()?);
    Ok(if first == u32::MAX { 0 } else { first >> 30 })
}

fn validate_modulation_limits(
    limits: &wgpu::Limits,
    input_bytes: u64,
    table_words: usize,
    output_words: usize,
    dispatch_len: usize,
    element_size: usize,
    workgroup: u32,
) -> Result<()> {
    ensure!(
        table_words <= u32::MAX as usize
            && output_words <= u32::MAX as usize
            && dispatch_len <= u32::MAX as usize,
        "modulation: GPU word extent exceeds shader indexing"
    );
    super::super::backend_shared::validate_compute_binding_counts("modulation", 4, 5, limits)?;
    ensure!(
        limits.max_uniform_buffers_per_shader_stage >= 1
            && limits.max_uniform_buffer_binding_size >= 16,
        "modulation: uniform binding unavailable"
    );
    ensure!(
        workgroup != 0
            && workgroup <= limits.max_compute_workgroup_size_x
            && workgroup <= limits.max_compute_invocations_per_workgroup,
        "modulation: workgroup exceeds enabled limits"
    );
    let groups =
        crate::backend::wgpu::dispatch::common::dispatch_size(dispatch_len as u32, workgroup);
    ensure!(
        groups <= limits.max_compute_workgroups_per_dimension,
        "modulation: dispatch exceeds enabled limits"
    );
    let binding_limit = u64::from(limits.max_storage_buffer_binding_size);
    for (name, bytes) in [
        ("input", input_bytes),
        (
            "table",
            BufferResidency::requested_bytes(table_words, element_size)
                .ok_or_else(|| anyhow!("modulation: table byte extent overflows"))?,
        ),
        (
            "output",
            BufferResidency::requested_bytes(output_words, element_size)
                .ok_or_else(|| anyhow!("modulation: output byte extent overflows"))?,
        ),
        ("error", 16),
    ] {
        ensure!(
            bytes <= binding_limit && bytes <= limits.max_buffer_size,
            "modulation: {name} binding exceeds enabled limits"
        );
    }
    Ok(())
}

impl WgpuProvider {
    pub(super) fn plan_modulation_pipeline(
        &self,
        shader: String,
        work: ModulationWork<'_>,
    ) -> Result<super::super::pipeline_recipe::PipelineRecipe> {
        validate_modulation_limits(
            &self.adapter_limits,
            work.input.allocated_bytes,
            work.table_words,
            work.output_words,
            work.dispatch_len,
            self.element_size,
            work.workgroup,
        )?;
        Ok(super::super::pipeline_recipe::PipelineRecipe {
            shader,
            layout: LAYOUT_TAG.into(),
            workgroup: work.workgroup,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn modulation_preflight_rejects_extent_and_binding_limits() {
        let limits = wgpu::Limits::default();
        validate_modulation_limits(&limits, 16, 4, 8, 4, 4, 64).unwrap();
        let mut tiny_storage = limits.clone();
        tiny_storage.max_storage_buffer_binding_size = 15;
        assert!(
            validate_modulation_limits(&tiny_storage, 16, 4, 8, 4, 4, 64)
                .unwrap_err()
                .to_string()
                .contains("input binding")
        );
        let mut too_few_bindings = limits;
        too_few_bindings.max_bindings_per_bind_group = 4;
        assert!(validate_modulation_limits(&too_few_bindings, 16, 4, 8, 4, 4, 64).is_err());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn exact_error_word_decoder_preserves_codes_and_rejects_short_map() {
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(&u32::MAX.to_ne_bytes());
        assert_eq!(decode_error_code(&bytes).unwrap(), 0);
        bytes[..4].copy_from_slice(&(2u32 << 30).to_ne_bytes());
        assert_eq!(decode_error_code(&bytes).unwrap(), 2);
        assert!(decode_error_code(&bytes[..4]).is_err());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn bit_group_width_matches_shader_accumulator() {
        assert!(validate_bit_group_width(32).is_ok());
        assert!(validate_bit_group_width(33).unwrap_err().is_unsupported());
    }
}
