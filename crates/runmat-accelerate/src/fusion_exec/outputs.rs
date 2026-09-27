//! Request-local output validation. Actual Values own allocation lifetimes.
use anyhow::anyhow;
use runmat_accelerate_api::{
    allocation::GpuHandleReleaseError, AccelProvider, GpuTensorHandle, GpuTensorStorage,
    NumericElementType, ProviderExecutionError, ProviderExecutionResult as Result,
    ProviderPrecision, WeakGpuTensorHandle,
};
use runmat_runtime::builtins::common::gpu_helpers::{
    exact_provider_for_handle, gpu_class_metadata_matches,
};
use runmat_value::Value;

pub(super) struct FusionOutputAdmission {
    borrowed: Vec<GpuTensorHandle>,
    produced: Vec<WeakGpuTensorHandle>,
}
impl FusionOutputAdmission {
    pub fn new(inputs: &[Value]) -> Self {
        Self {
            borrowed: inputs
                .iter()
                .filter_map(|value| match value {
                    Value::GpuTensor(handle) => Some(handle.clone()),
                    _ => None,
                })
                .collect(),
            produced: Vec::new(),
        }
    }

    pub fn upload(
        &mut self,
        provider: &'static dyn AccelProvider,
        data: &[f64],
        shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        use runmat_accelerate_api::{
            upload_numeric_protecting, HostNumericDataView, HostNumericTensorView,
        };
        // Temporary upgrades protect live intermediates during upload validation.
        // Weak records neither prolong output lifetimes nor recreate allocations.
        let mut live = Vec::new();
        for handle in &self.produced {
            match handle.acquire() {
                Ok(handle) => live.push(handle),
                Err(runmat_accelerate_api::allocation::AllocationAcquisitionError::NotLive) => {}
                Err(error) => return Err(ProviderExecutionError::from_source(error)),
            }
        }
        let protected: Vec<_> = self.borrowed.iter().chain(live.iter()).collect();
        let single;
        let data = match provider.precision() {
            ProviderPrecision::F64 => HostNumericDataView::F64(data),
            ProviderPrecision::F32 => {
                single = data.iter().map(|value| *value as f32).collect::<Vec<_>>();
                HostNumericDataView::F32(&single)
            }
        };
        let handle = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data,
                shape,
                storage: GpuTensorStorage::Real,
            },
            &protected,
        )
        .map_err(ProviderExecutionError::from_source)?;
        self.admit(handle, provider, shape, provider.precision())
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
            && exact_provider_for_handle(handle).is_some_and(|owner| std::ptr::eq(owner, provider))
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
        anyhow!(message).into()
    } else {
        ProviderExecutionError::from_source(RejectedOutputs { message, cleanup })
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
