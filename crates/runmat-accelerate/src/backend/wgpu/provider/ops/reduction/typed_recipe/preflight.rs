//! Provider-side physical admission for an immutable selected reduction.
use super::*;
use runmat_accelerate_api::reduction::{ReductionSelection, ReductionStrategy};

pub(super) fn physical(precision: ProviderPrecision) -> NumericElementType {
    match precision {
        ProviderPrecision::F32 => NumericElementType::F32,
        ProviderPrecision::F64 => NumericElementType::F64,
    }
}

pub(crate) fn unsupported(reason: &'static str) -> ProviderExecutionError {
    ProviderExecutionError::from(runmat_accelerate_api::unsupported_provider_operation(
        reason,
    ))
}

fn selected_binding_counts(
    strategy: ReductionStrategy,
    producer_inputs: usize,
) -> ProviderExecutionResult<(usize, usize)> {
    let producer = producer_inputs
        .checked_add(1)
        .ok_or_else(|| unsupported("typed reduction binding count overflow"))?;
    let final_stage = match strategy {
        ReductionStrategy::SinglePass => 0,
        ReductionStrategy::TwoPass => 2,
    };
    Ok((producer, final_stage))
}

fn bindings_fit(producer: usize, final_stage: usize, storage_limit: u32, group_limit: u32) -> bool {
    [producer, final_stage]
        .iter()
        .all(|&count| count <= storage_limit as usize && count <= group_limit as usize)
}

impl WgpuProvider {
    pub(crate) fn admit_reduction_selection(
        &self,
        selection: &ReductionSelection<'_>,
        host_metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<()> {
        let recipe = selection.recipe();
        let d = recipe.descriptor();
        let limits = &self.adapter_limits;
        if d.geometry.slices == 0 {
            return Err(unsupported("empty typed reduction submission"));
        }
        crate::fusion::reduction_recipe::validate_lowered_sources(recipe, host_metadata)
            .map_err(ProviderExecutionError::from)?;
        let f64_required = d.execution_precision == ProviderPrecision::F64
            || d.output_precision == ProviderPrecision::F64
            || d.partial.precision == ProviderPrecision::F64
            || d.inputs
                .iter()
                .any(|b| b.precision == ProviderPrecision::F64);
        self.require_selected_shader_requirements(if f64_required {
            ProviderShaderRequirements::F64IeeeWords
        } else {
            ProviderShaderRequirements::Basic
        })
        .map_err(ProviderExecutionError::from)?;
        let dispatches: &[[u32; 3]] = match selection.strategy() {
            ReductionStrategy::SinglePass => std::slice::from_ref(&d.geometry.single.workgroups),
            ReductionStrategy::TwoPass => {
                // Both selected stages must be admitted before any allocation.
                &[d.geometry.pass1.workgroups, d.geometry.pass2.workgroups]
            }
        };
        let partial_binding = selection.strategy() == ReductionStrategy::TwoPass;
        let (producer_bindings, final_bindings) =
            selected_binding_counts(selection.strategy(), d.inputs.len())?;
        let partial_elements = u64::from(d.geometry.partial_elements);
        let exact_partial_payload = partial_elements.checked_mul(u64::from(d.partial.stride));
        if d.config.workgroup_size > limits.max_compute_workgroup_size_x
            || d.config.workgroup_size > limits.max_compute_invocations_per_workgroup
            || d.work.workgroup_bytes > u64::from(limits.max_compute_workgroup_storage_size)
            || dispatches
                .iter()
                .flatten()
                .any(|&n| n > limits.max_compute_workgroups_per_dimension)
            || d.work.output_requested_bytes > limits.max_buffer_size
            || d.work.output_requested_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || !bindings_fit(
                producer_bindings,
                final_bindings,
                limits.max_storage_buffers_per_shader_stage,
                limits.max_bindings_per_bind_group,
            )
            || (partial_binding
                && (d.work.partial_requested_bytes > limits.max_buffer_size
                    || d.work.partial_requested_bytes
                        > u64::from(limits.max_storage_buffer_binding_size)
                    || exact_partial_payload != Some(d.work.partial_payload_bytes)
                    || d.partial.stride == 0))
        {
            return Err(unsupported("typed reduction exceeds enabled limits"));
        }
        Ok(())
    }

    pub(super) fn admit_reduction_recipe(
        &self,
        submission: &ReductionSubmission<'_>,
    ) -> ProviderExecutionResult<SubmittedReductionInputs> {
        if submission.device_id() != self.device_id() {
            return Err(unsupported("foreign typed reduction submission"));
        }
        let selection = submission.selection();
        self.admit_reduction_selection(selection, submission.host_metadata())?;
        let d = selection.recipe().descriptor();
        let limits = &self.adapter_limits;
        // The original vector's exact backing is admitted BEFORE allocation,
        // and consumed by the canonical physical retirement owner below.
        let count = submission.inputs().len();
        let mut inputs = SubmittedReductionInputs::reserve(count, submission.host_metadata())
            .map_err(ProviderExecutionError::from)?;
        for (handle, binding) in submission.inputs().iter().zip(&d.inputs) {
            // The allocation owner constructs exactly this view's shape (it
            // no longer clones an obsolete physical-record shape first).
            let shape_bytes = std::alloc::Layout::array::<usize>(handle.shape().len())
                .map_err(ProviderExecutionError::from_source)?
                .size();
            let _entry_shape_claim =
                crate::wgsl::host_buffers::claim_bytes(submission.host_metadata(), shape_bytes)
                    .map_err(ProviderExecutionError::from_source)?;
            let e = self
                .get_entry_raw(handle)
                .map_err(ProviderExecutionError::from)?;
            if e.logical
                || e.storage != GpuTensorStorage::Real
                || e.element_type != physical(binding.precision)
                || e.allocated_bytes != e.buffer.size()
                || e.buffer.size() < binding.payload_bytes
                || e.buffer.size() > limits.max_buffer_size
                || e.buffer.size() > u64::from(limits.max_storage_buffer_binding_size)
            {
                return Err(unsupported(
                    "typed reduction actual input differs from binding",
                ));
            }
            inputs
                .push(
                    SubmittedReductionInput::new(handle, e.buffer)
                        .map_err(ProviderExecutionError::from)?,
                )
                .map_err(ProviderExecutionError::from)?;
        }
        Ok(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn zero_input_two_pass_still_admits_two_final_bindings() {
        assert_eq!(
            selected_binding_counts(ReductionStrategy::TwoPass, 0).unwrap(),
            (1, 2)
        );
        assert_eq!(
            selected_binding_counts(ReductionStrategy::SinglePass, 0).unwrap(),
            (1, 0)
        );
        assert!(!bindings_fit(1, 2, 1, 2));
        assert!(!bindings_fit(1, 2, 2, 1));
        assert!(bindings_fit(1, 2, 2, 2));
        assert!(selected_binding_counts(ReductionStrategy::TwoPass, usize::MAX).is_err());
    }
}
