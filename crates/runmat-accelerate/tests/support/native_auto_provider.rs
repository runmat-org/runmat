//! Deterministic selection evidence, not hardware or performance qualification.
//! Arithmetic, payloads, ownership and transfers remain actual InProcess work.
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{observation::*, *};
use std::{
    num::NonZeroU32,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

pub struct Provider {
    inner: InProcessProvider,
    budget: Arc<ProviderObservationBudget>,
    plus_operation: ProviderOperationIdentity,
    upload_operation: ProviderOperationIdentity,
    pub executed_adds: AtomicUsize,
}

impl Default for Provider {
    fn default() -> Self {
        Self {
            inner: InProcessProvider::new(),
            budget: Arc::new(ProviderObservationBudget::new(NonZeroU32::new(8).unwrap())),
            plus_operation: ProviderOperationIdentity::new(
                runmat_builtins::PLUS_CATALOG_ENTRY.identity.name,
            ),
            upload_operation: ProviderOperationIdentity::new("transfer.upload"),
            executed_adds: AtomicUsize::new(0),
        }
    }
}

impl Provider {
    fn observes(&self, operation: &ProviderOperationIdentity) -> bool {
        operation == &self.plus_operation || operation == &self.upload_operation
    }
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "deterministic native-auto fixture (actual InProcess arithmetic)".into()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        self.inner.capability_snapshot()
    }
    fn capability_snapshot_ref(&self) -> Option<&ProviderCapabilitySnapshot> {
        self.inner.capability_snapshot_ref()
    }
    fn placement_resources(&self) -> runmat_execution::ProviderResourceSnapshot {
        self.inner.placement_resources()
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        self.inner.query_feasibility(query)
    }
    fn upload_physical_evidence(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderUploadPhysicalEvidence> {
        self.inner.upload_physical_evidence(query)
    }
    fn estimate_cost(&self, query: &ProviderCostQuery) -> Option<ProviderCostEstimate> {
        // This controlled prior tests selection, not measured profitability.
        if !self.observes(&query.operation.operation) {
            return None;
        }
        self.query_feasibility(&query.operation)
            .is_supported()
            .then(|| {
                ProviderCostEstimate::static_prior(
                    runmat_execution::ExecutionCostComponents {
                        execution_ns: 1_000,
                        ..Default::default()
                    },
                    0,
                )
            })
    }
    fn operation_observation_capability(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderObservationCapability {
        if self.observes(&query.operation) && self.query_feasibility(query).is_supported() {
            // All actual native computation, allocation and publication ends
            // before these hooks return; there is no deferred device work.
            ProviderObservationCapability::Supported(ProviderObservationOverhead {
                submissions: 0,
                requested_peak_buffer_bytes: 0,
                clear_commands: 0,
                map_callbacks: 0,
                wgpu_compute_query: None,
            })
        } else {
            ProviderObservationCapability::Unsupported
        }
    }
    fn begin_operation_observation(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderObservationSupport {
        match self.operation_observation_capability(query) {
            ProviderObservationCapability::Supported(_) => ProviderObservationSupport::Observed(
                ProviderOperationReceipt::with_budget(self.budget.clone()),
            ),
            ProviderObservationCapability::Unsupported => ProviderObservationSupport::Unsupported,
        }
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(input)
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.inner.download_numeric(input)
    }
    fn query_numeric_download_feasibility(
        &self,
        source: &GpuTensorHandle,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderFeasibility {
        self.inner.query_numeric_download_feasibility(source, query)
    }
    fn download_numeric_host_workspace_bound(
        &self,
        source: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        self.inner.download_numeric_host_workspace_bound(source)
    }
    fn download_numeric_provider_scratch_bound(
        &self,
        source: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        self.inner.download_numeric_provider_scratch_bound(source)
    }
    fn download_numeric_completion_host_bound(
        &self,
        source: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        self.inner.download_numeric_completion_host_bound(source)
    }
    fn download_numeric_queue_slots(&self, source: &GpuTensorHandle) -> u32 {
        self.inner.download_numeric_queue_slots(source)
    }
    fn prepare_numeric_download_stage(
        &self,
        source: &GpuTensorHandle,
        query: &ProviderFeasibilityQuery,
        claim: Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
    ) -> anyhow::Result<PreparedNumericDownloadStage> {
        self.inner
            .prepare_numeric_download_stage(source, query, claim)
    }
    fn download_numeric_admitted<'a>(
        &'a self,
        source: &'a GpuTensorHandle,
        claim: Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
    ) -> AccelNumericDownloadFuture<'a> {
        self.inner.download_numeric_admitted(source, claim)
    }
    fn download_numeric_admitted_stage<'a>(
        &'a self,
        source: &'a GpuTensorHandle,
        stage: PreparedNumericDownloadStage,
    ) -> AccelNumericDownloadFuture<'a> {
        self.inner.download_numeric_admitted_stage(source, stage)
    }
    fn elem_add<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let output = self.inner.elem_add(left, right).await?;
            self.executed_adds.fetch_add(1, Ordering::SeqCst);
            let operation = &self.plus_operation;
            if let Some(recorder) = current_provider_observation()
                .filter(|recorder| recorder.expects_operation(operation))
            {
                recorder.confirm_provider_execution(self.device_id(), operation);
            }
            Ok(output)
        })
    }
    fn unary_sinc<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.inner.unary_sinc(input)
    }
    fn telemetry_snapshot(&self) -> ProviderTelemetry {
        self.inner.telemetry_snapshot()
    }
}
