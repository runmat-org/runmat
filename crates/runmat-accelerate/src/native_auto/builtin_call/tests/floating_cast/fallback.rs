//! Real transfer/body fallback with one deliberately unsupported provider hook.
//! Placement is forced by the fixture, not a profitability assertion.
use super::*;
use crate::backend::wgpu::provider::WgpuProvider;
use runmat_accelerate_api::{
    observation::{ProviderObservationCapability, ProviderObservationSupport},
    AccelDownloadFuture, AccelNumericDownloadFuture, GpuTensorHandle, HostNumericDataOwned,
    HostNumericTensorView, ProviderCapabilitySnapshot, ProviderCostEstimate, ProviderCostQuery,
    ProviderElementwiseGeometry, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderPreparationProgram, ProviderPreparationSnapshot,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct UnsupportedCast {
    inner: WgpuProvider,
    casts: AtomicUsize,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
    receipts: AtomicUsize,
}

impl AccelProvider for UnsupportedCast {
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
        self.inner.upload_numeric_raw(host)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        self.inner.download_numeric(handle)
    }

    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
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
        let support = self.inner.begin_operation_observation(query);
        if matches!(&support, ProviderObservationSupport::Observed(_)) {
            self.receipts.fetch_add(1, Ordering::Relaxed);
        }
        support
    }

    fn unary_single<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.casts.fetch_add(1, Ordering::Relaxed);
        Box::pin(async {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "fixture declines cast after admission",
            )
            .into())
        })
    }
}

#[test]
#[ignore = "requires actual WGPU transfers; execute explicitly for fallback receipt qualification"]
fn actual_single_unsupported_hook_restores_values_without_successful_device_feedback() {
    // Do not register the inner provider separately: the wrapper delegates
    // its allocation owner and preserves every handle identity unchanged.
    let provider = Box::leak(Box::new(UnsupportedCast {
        inner: WgpuProvider::new(Default::default()).expect("actual WGPU provider required"),
        casts: AtomicUsize::new(0),
        uploads: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        receipts: AtomicUsize::new(0),
    }));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let probe = Rc::new(Probe::default());
    probe.provider.set(true);
    let context = runtime(&probe);
    let context = context.clone().with_service_ports(
        context
            .service_ports()
            .clone()
            .with_acceleration(Rc::new(Adapter(provider, 1))),
    );
    let values = [1.25_f32, -0.0, 16_777_216.0, 3.5];
    let input = Value::Tensor(Tensor::from_f32(values.to_vec(), vec![1, 2, 2]).unwrap());
    let output = futures::executor::block_on(context.scope(async {
        runmat_runtime::dispatcher::call_builtin_async_in_context(&context, "single", &[input])
            .await
    }))
    .unwrap();
    let Value::GpuTensor(resident_output) = output.value() else {
        panic!(
            "fallback must restore resident output, got {:?}; selected graphs={}, casts={}, uploads={}, downloads={}",
            output.value(), probe.requests.borrow().len(), provider.casts.load(Ordering::Relaxed),
            provider.uploads.load(Ordering::Relaxed), provider.downloads.load(Ordering::Relaxed)
        );
    };
    assert_eq!(resident_output.shape(), &[1, 2, 2]);
    assert_eq!(provider.casts.load(Ordering::Relaxed), 1);
    assert_eq!(provider.receipts.load(Ordering::Relaxed), 1);
    assert_eq!(
        provider.uploads.load(Ordering::Relaxed),
        2,
        "staging and restoration"
    );
    assert_eq!(
        provider.downloads.load(Ordering::Relaxed),
        1,
        "runtime fallback gathered"
    );
    let feedback = probe.feedback.borrow();
    // This is an admitted whole-call candidate, not a legacy preparation
    // sample. Its unsupported body falls back successfully, so neither the
    // promised provider route nor a failed whole call completed. Restoration
    // must not manufacture either kind of completed placement feedback.
    assert!(feedback.is_empty());
    drop(feedback);
    let HostNumericDataOwned::F32(actual) =
        futures::executor::block_on(provider.inner.download_numeric(resident_output))
            .unwrap()
            .data
    else {
        panic!("fallback must preserve F32 storage");
    };
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        values.map(f32::to_bits)
    );
    drop(output);
}
