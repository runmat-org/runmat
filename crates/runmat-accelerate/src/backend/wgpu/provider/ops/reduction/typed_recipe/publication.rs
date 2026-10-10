//! Transactional output publication and exact final queue tracking.
use super::*;

impl WgpuProvider {
    /// One publication boundary for actual reduction output and its exact queue ID.
    pub(super) fn publish_reduction_output(
        &self,
        output: runmat_accelerate_api::WgpuBufferOwner,
        shape: Vec<usize>,
        len: usize,
        output_type: NumericElementType,
        submission_id: u32,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let allocated_bytes = output.size();
        let handle = self
            .register_numeric_buffer(
                output,
                super::super::super::backend_types::NumericBufferRegistration {
                    shape,
                    len,
                    physical_element_type: output_type,
                    storage: GpuTensorStorage::Real,
                    allocated_bytes,
                    usage: BufferUsageClass::FusionOut,
                },
            )
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        if let Err(error) = self
            .allocations
            .record_submission(handle.buffer_id(), submission_id)
        {
            let primary = ProviderExecutionError::from(error).after_effect();
            return Err(match handle.release() {
                Ok(_) => primary,
                Err(cleanup) => ProviderExecutionError::from_source(
                    runmat_accelerate_api::ProviderExecutionCompletionError::new(
                        Some(primary),
                        vec![cleanup],
                    ),
                )
                .after_effect(),
            });
        }
        Ok(handle)
    }
}
