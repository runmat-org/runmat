//! Transactional output publication and exact final queue tracking.
use super::*;

impl WgpuProvider {
    /// Register once before the output's first dispatch, so that dispatch can
    /// retain the actual final allocation lease before caller publication.
    pub(super) fn register_reduction_output(
        &self,
        output: runmat_accelerate_api::WgpuBufferOwner,
        shape: Vec<usize>,
        len: usize,
        output_type: NumericElementType,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<GpuTensorHandle> {
        let allocated_bytes = output.size();
        self.register_numeric_buffer_produced_admitted(
            output,
            super::super::super::backend_types::NumericBufferRegistration {
                shape,
                len,
                physical_element_type: output_type,
                storage: GpuTensorStorage::Real,
                allocated_bytes,
                usage: BufferUsageClass::FusionOut,
            },
            false,
            admission,
        )
    }

    /// Publish the already-registered output with its exact final queue ID.
    pub(super) fn publish_reduction_output(
        &self,
        handle: GpuTensorHandle,
        submission_id: u32,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
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
