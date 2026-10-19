//! Request-local output validation. Actual Values own allocation lifetimes.
use anyhow::anyhow;
use runmat_accelerate_api::{
    allocation::{GpuHandleReleaseError, WeakGpuTensorHandle},
    AccelProvider, GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderExecutionError,
    ProviderExecutionResult as Result, ProviderPrecision,
};
use runmat_runtime::builtins::common::gpu_helpers::{
    exact_provider_for_handle, gpu_class_metadata_matches,
};
use runmat_value::Value;

mod upload;

pub(super) struct FusionOutputAdmission {
    borrowed: Vec<GpuTensorHandle>,
    produced: Vec<WeakGpuTensorHandle>,
}
impl FusionOutputAdmission {
    pub fn new(inputs: &[Value]) -> Self {
        Self::with_borrowed_handles(inputs.iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        }))
    }

    /// Construct admission from already-resolved resident leaves without
    /// cloning unrelated Values. The cloned handles are request-scoped
    /// allocation protection, just like the GPU leaves selected by `new`.
    pub fn with_borrowed_handles<'h>(
        inputs: impl IntoIterator<Item = &'h GpuTensorHandle>,
    ) -> Self {
        Self {
            borrowed: inputs.into_iter().cloned().collect(),
            produced: Vec::new(),
        }
    }

    fn valid(
        &self,
        handle: &GpuTensorHandle,
        provider: &'static dyn AccelProvider,
        shape: &[usize],
        precision: ProviderPrecision,
    ) -> bool {
        let element = match precision {
            ProviderPrecision::F32 => NumericElementType::F32,
            ProviderPrecision::F64 => NumericElementType::F64,
        };
        handle.shape() == shape
            && handle.device_id() == provider.device_id()
            && handle.physical().storage() == GpuTensorStorage::Real
            && handle.physical().element_type() == element
            && !runmat_accelerate_api::handle_is_logical(handle)
            && !runmat_accelerate_api::handle_is_transposed(handle)
            && exact_provider_for_handle(handle)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
            && gpu_class_metadata_matches(handle, Some(precision), None, false)
            && !self
                .borrowed
                .iter()
                .any(|input| input.shares_allocation_with(handle))
    }

    pub fn admit(
        &mut self,
        handle: GpuTensorHandle,
        provider: &'static dyn AccelProvider,
        shape: &[usize],
        precision: ProviderPrecision,
    ) -> Result<GpuTensorHandle> {
        if !self.valid(&handle, provider, shape, precision) {
            return Err(reject_outputs(
                vec![handle],
                "fusion provider returned an invalid floating allocation",
            ));
        }
        self.produced.push(handle.downgrade());
        Ok(handle)
    }

    pub fn admit_batch(
        &mut self,
        handles: Vec<GpuTensorHandle>,
        provider: &'static dyn AccelProvider,
        count: usize,
        shape: &[usize],
        precision: ProviderPrecision,
    ) -> Result<Vec<GpuTensorHandle>> {
        if handles.len() != count {
            return Err(reject_outputs(
                handles,
                "fusion provider returned the wrong output count",
            ));
        }
        if handles
            .iter()
            .any(|handle| !self.valid(handle, provider, shape, precision))
        {
            return Err(reject_outputs(
                handles,
                "fusion provider returned a malformed output batch",
            ));
        }
        self.produced
            .extend(handles.iter().map(GpuTensorHandle::downgrade));
        Ok(handles)
    }
}

pub(super) fn reject_outputs(
    handles: Vec<GpuTensorHandle>,
    message: &'static str,
) -> ProviderExecutionError {
    let cleanup: Vec<_> = handles
        .into_iter()
        .filter_map(|handle| handle.release().err())
        .collect();
    if cleanup.is_empty() {
        ProviderExecutionError::from(anyhow!(message)).after_effect()
    } else {
        ProviderExecutionError::from_source(RejectedOutputs { message, cleanup }).after_effect()
    }
}
#[derive(Debug)]
struct RejectedOutputs {
    message: &'static str,
    cleanup: Vec<GpuHandleReleaseError>,
}
impl std::fmt::Display for RejectedOutputs {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}; {} allocation retirement failure(s)",
            self.message,
            self.cleanup.len()
        )
    }
}
impl std::error::Error for RejectedOutputs {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cleanup.first().map(|error| error as _)
    }
}

#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
