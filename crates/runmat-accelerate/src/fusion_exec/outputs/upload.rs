use super::FusionOutputAdmission;
use runmat_accelerate_api::{
    allocation::AllocationAcquisitionError, AccelProvider, GpuTensorHandle, GpuTensorStorage,
    HostNumericDataView, HostNumericTensorView, ProviderExecutionError,
    ProviderExecutionResult as Result, ProviderPrecision,
};

impl FusionOutputAdmission {
    pub(in crate::fusion_exec) fn upload(
        &mut self,
        provider: &'static dyn AccelProvider,
        data: &[f64],
        shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        let single;
        let data = match provider.precision() {
            ProviderPrecision::F64 => HostNumericDataView::F64(data),
            ProviderPrecision::F32 => {
                single = data.iter().map(|value| *value as f32).collect::<Vec<_>>();
                HostNumericDataView::F32(&single)
            }
        };
        self.upload_typed(
            provider,
            &HostNumericTensorView {
                data,
                shape,
                storage: GpuTensorStorage::Real,
            },
        )
    }

    /// Upload one already-classified real floating host tensor without routing
    /// through the provider's default precision. Unsupported integer or
    /// complex inputs are declined before provider or allocation effects.
    pub(in crate::fusion_exec) fn upload_typed(
        &mut self,
        provider: &'static dyn AccelProvider,
        host: &HostNumericTensorView<'_>,
    ) -> Result<GpuTensorHandle> {
        self.upload_typed_admitted(provider, host, None)
    }

    pub(in crate::fusion_exec) fn upload_typed_admitted(
        &mut self,
        provider: &'static dyn AccelProvider,
        host: &HostNumericTensorView<'_>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<GpuTensorHandle> {
        let precision = match (host.storage, host.data) {
            (GpuTensorStorage::Real, HostNumericDataView::F32(_)) => ProviderPrecision::F32,
            (GpuTensorStorage::Real, HostNumericDataView::F64(_)) => ProviderPrecision::F64,
            _ => {
                let error = runmat_accelerate_api::unsupported_provider_operation(
                    "fusion typed upload requires real single or double input",
                )
                .into();
                return Err(self.classify_upload_phase_error(error));
            }
        };

        // Temporary upgrades protect live intermediates during upload
        // validation. Weak records neither prolong output lifetimes nor
        // recreate allocations.
        let mut live = Vec::new();
        for handle in &self.produced {
            match handle.acquire() {
                Ok(handle) => live.push(handle),
                Err(AllocationAcquisitionError::NotLive) => {}
                Err(error) => {
                    let error = ProviderExecutionError::from_source(error);
                    return Err(self.classify_upload_phase_error(error));
                }
            }
        }
        let protected: Vec<_> = self.borrowed.iter().chain(live.iter()).collect();
        let handle = runmat_accelerate_api::upload_numeric_protecting_admitted(
            provider, host, &protected, admission,
        )
        .map_err(|error| self.classify_upload_phase_error(error.into_execution_error()))?;
        self.admit(handle, provider, host.shape, precision)
    }

    /// A local upload failure is replay-safe only before this request has
    /// admitted any produced allocation. Once a prior stage was admitted, the
    /// whole fusion invocation has crossed its physical-effect boundary even
    /// if that weak record has since expired.
    fn classify_upload_phase_error(&self, error: ProviderExecutionError) -> ProviderExecutionError {
        if self.produced.is_empty() {
            error
        } else {
            error.after_effect()
        }
    }
}
