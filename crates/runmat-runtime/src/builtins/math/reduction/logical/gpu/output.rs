//! Ownership of fresh reduction temporaries only; the invocation owns its input.
use super::{gpu_helpers, LogicalReductionConfig};
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};

pub(super) struct ReductionOutputs {
    source: GpuTensorHandle,
    owned: Vec<GpuTensorHandle>,
}

impl ReductionOutputs {
    pub(super) fn new(source: &GpuTensorHandle) -> Self {
        Self {
            source: source.clone(),
            owned: Vec::new(),
        }
    }

    pub(super) fn current(&self) -> &GpuTensorHandle {
        self.owned.last().unwrap_or(&self.source)
    }

    pub(super) fn adopt(
        &mut self,
        config: &LogicalReductionConfig,
        provider: &'static dyn AccelProvider,
        output: GpuTensorHandle,
        shape: Vec<usize>,
    ) -> BuiltinResult<()> {
        let protected: Vec<_> = std::iter::once(&self.source)
            .chain(self.owned.iter())
            .collect();
        // A truth result can use either floating precision or logical metadata;
        // neither numeric precision is observable after the exact truth download.
        let contract = gpu_helpers::BinaryGpuOutputContract {
            shape,
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(&output),
            integer: None,
            logical: runmat_accelerate_api::handle_is_logical(&output),
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        };
        let valid =
            gpu_helpers::expected_handle_numeric_element_type(&output).is_ok_and(|element| {
                gpu_helpers::numeric_descriptor_matches(&output, element, GpuTensorStorage::Real)
            }) && !protected
                .iter()
                .any(|input| gpu_helpers::same_gpu_handle(&output, input))
                && gpu_helpers::binary_gpu_output_matches(
                    &output,
                    &self.source,
                    self.current(),
                    provider,
                    &contract,
                );
        if !valid {
            gpu_helpers::free_rejected_provider_output(&output, &protected, provider);
            return Err(config.error(
                config.internal,
                "provider returned an invalid or aliased logical reduction result",
            ));
        }
        self.owned.push(output);
        Ok(())
    }
}

impl Drop for ReductionOutputs {
    fn drop(&mut self) {
        // Accepted outputs are distinct from all inputs/earlier temporaries.
        // This also runs when a reduction or download future is cancelled.
        for output in &self.owned {
            gpu_helpers::free_unprotected_exact_owner(output, &[&self.source]);
        }
    }
}
