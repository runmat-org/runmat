use super::gate::Gate;
use crate::backend::wgpu::provider::WgpuProvider;
use anyhow::Result;
use runmat_accelerate_api::{observation::*, *};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

pub(super) struct SubmittedCast {
    pub(super) inner: WgpuProvider,
    pub(super) casts: AtomicUsize,
    pub(super) uploads: AtomicUsize,
    pub(super) downloads: AtomicUsize,
    pub(super) gate: Mutex<Option<Arc<Gate>>>,
}
impl SubmittedCast {
    pub(super) fn new() -> Self {
        let inner =
            WgpuProvider::new(Default::default()).expect("actual shader-f64 WGPU device required");
        assert_eq!(inner.precision(), ProviderPrecision::F64, "run with RUNMAT_WGPU_FORCE_PRECISION=f64 on a shader-f64 adapter; unavailable F64 is a failed gate, not a skip");
        Self {
            inner,
            casts: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            gate: Mutex::new(None),
        }
    }
}
impl AccelProvider for SubmittedCast {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn upload(&self, host: &HostTensorView<'_>) -> Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        assert!(
            matches!(host.data, HostNumericDataView::F64(_)),
            "staging must not preconvert away the F64 kernel"
        );
        self.inner.upload_numeric_raw(host)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        self.inner.download_numeric(handle)
    }
    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        self.inner.capability_snapshot()
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        self.inner.query_feasibility(query)
    }
    fn estimate_cost(&self, query: &ProviderCostQuery) -> Option<ProviderCostEstimate> {
        self.inner.estimate_cost(query)
    }
    fn elementwise_geometry(&self) -> Option<ProviderElementwiseGeometry> {
        self.inner.elementwise_geometry()
    }
    fn preparation_snapshot(
        &self,
        query: &ProviderFeasibilityQuery,
        program: ProviderPreparationProgram<'_>,
    ) -> Option<ProviderPreparationSnapshot> {
        self.inner.preparation_snapshot(query, program)
    }
    fn operation_observation_capability(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderObservationCapability {
        self.inner.operation_observation_capability(query)
    }
    fn begin_operation_observation(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderObservationSupport {
        let ProviderFeasibility::Supported { estimate } = self.inner.query_feasibility(query)
        else {
            panic!("actual F64 cast must be feasible")
        };
        assert_eq!(estimate.dispatches, Some(1));
        let ProviderObservationSupport::Observed(receipt) =
            self.inner.begin_operation_observation(query)
        else {
            panic!("actual native fence observation required")
        };
        let progress = receipt
            .recorder()
            .wgpu_progress()
            .expect("native exact-ticket progress owner");
        let gate = Arc::new(Gate::new(progress));
        assert!(
            self.gate.lock().unwrap().replace(gate.clone()).is_none(),
            "exactly one receipt"
        );
        // The original empty receipt only supplies the real provider's progress
        // port. The replacement observes the actual dispatch/fence tokens.
        ProviderObservationSupport::Observed(ProviderOperationReceipt::with_wgpu_progress(
            Arc::new(ProviderObservationBudget::new(
                std::num::NonZeroU32::new(1).unwrap(),
            )),
            gate,
        ))
    }
    fn unary_single<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.casts.fetch_add(1, Ordering::Relaxed);
        assert_eq!(
            handle.descriptor().element_type,
            Some(NumericElementType::F64)
        );
        self.inner.unary_single(handle)
    }
}
