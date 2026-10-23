//! A real native-four in-process owner whose preference alone is F32.
//! Do not use F32TestProvider here: it intentionally refuses F64 uploads.
use runmat_accelerate_api::{
    register_device_provider, AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider,
    ApiDeviceInfo, GpuTensorHandle, HostNumericTensorView, HostTensorView,
    ProviderCapabilitySnapshot, ProviderExecutionResult, ProviderFeasibility,
    ProviderFeasibilityQuery, ProviderNumericUploadPhysicalEvidence, ProviderPrecision,
    ProviderUploadPhysicalEvidence, SpawnHandleConcurrency, ThreadProviderGuard,
};

struct F32PreferredOwner(runmat_accelerate::simple_provider::InProcessProvider);

impl AccelProvider for F32PreferredOwner {
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F32
    }
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        self.0.device_info()
    }
    fn device_info_struct(&self) -> ApiDeviceInfo {
        self.0.device_info_struct()
    }
    fn spawn_handle_concurrency(&self) -> SpawnHandleConcurrency {
        self.0.spawn_handle_concurrency()
    }
    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        self.0.capability_snapshot()
    }
    fn capability_snapshot_ref(&self) -> Option<&ProviderCapabilitySnapshot> {
        self.0.capability_snapshot_ref()
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        self.0.query_feasibility(query)
    }
    fn upload_physical_evidence(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderUploadPhysicalEvidence> {
        self.0.upload_physical_evidence(query)
    }
    fn numeric_upload_physical_evidence(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderNumericUploadPhysicalEvidence> {
        self.0.numeric_upload_physical_evidence(query)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.0.upload_numeric_raw(host)
    }
    fn upload_numeric_raw_admitted(
        &self,
        host: &HostNumericTensorView<'_>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.0.upload_numeric_raw_admitted(host, admission)
    }
    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.0.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.0.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.0.download_numeric(handle)
    }
    fn query_numeric_download_feasibility(
        &self,
        handle: &GpuTensorHandle,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderFeasibility {
        self.0.query_numeric_download_feasibility(handle, query)
    }
    fn download_numeric_host_workspace_bound(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        self.0.download_numeric_host_workspace_bound(handle)
    }
    fn download_numeric_provider_scratch_bound(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        self.0.download_numeric_provider_scratch_bound(handle)
    }
    fn download_numeric_queue_slots(&self, handle: &GpuTensorHandle) -> u32 {
        self.0.download_numeric_queue_slots(handle)
    }
}

pub(super) fn with_provider<R>(f: impl FnOnce(&'static dyn AccelProvider) -> R) -> R {
    let _guard = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static dyn AccelProvider = Box::leak(Box::new(F32PreferredOwner(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    )));
    // SAFETY: fresh in-process device identity, never previously registered;
    // the wrapper and its actual storage owner both live for the process.
    unsafe { register_device_provider(provider) };
    let _thread = ThreadProviderGuard::set(Some(provider));
    f(provider)
}
