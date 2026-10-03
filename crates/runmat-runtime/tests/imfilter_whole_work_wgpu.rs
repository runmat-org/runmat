//! Required actual-adapter witness for the explicit imfilter Unsupported
//! continuation. The facade keeps WGPU's real allocation, readback, upload
//! and fence paths; only the selected hook's outcome is forced Unsupported.
#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

use futures::executor::block_on;
use runmat_accelerate::{backend::wgpu::provider::WgpuProvider, placement::PlacementSession};
use runmat_accelerate_api::{
    observation::{ProviderObservationCapability, ProviderObservationSupport},
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuHandleProvenance,
    GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorView, HostTensorView, ImfilterOptions, ProviderCapabilitySnapshot,
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderImfilterPhysicalEvidence,
    ProviderNumericUploadPhysicalEvidence, ProviderPrecision, ThreadProviderGuard,
};
use runmat_runtime::{
    call_builtin_async_in_context,
    context::{
        ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
        RuntimeServicePorts,
    },
};
use runmat_value::{Tensor, Value};
use std::{
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

struct HostBudget(u64);
impl RuntimeParallelService for HostBudget {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(self.0),
            epoch: 0,
        }
    }
}

fn context(session: Rc<PlacementSession>, host_bytes: u64) -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(session)
            .with_parallel(Rc::new(HostBudget(host_bytes))),
    )
}

struct UnsupportedImfilter(&'static WgpuProvider, AtomicUsize, bool);
impl AccelProvider for UnsupportedImfilter {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.0.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.0.download(handle)
    }
    fn upload_numeric_raw(&self, host: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.0.upload_numeric_raw(host)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.0.download_numeric(handle)
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
    fn query_numeric_download_feasibility(
        &self,
        source: &GpuTensorHandle,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderFeasibility {
        self.0.query_numeric_download_feasibility(source, query)
    }
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        self.0.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        self.0.precision()
    }
    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        self.0.capability_snapshot()
    }
    fn placement_resources(&self) -> runmat_execution::ProviderResourceSnapshot {
        self.0.placement_resources()
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        self.0.query_feasibility(query)
    }
    fn numeric_upload_physical_evidence(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderNumericUploadPhysicalEvidence> {
        (!self.2)
            .then(|| self.0.numeric_upload_physical_evidence(query))
            .flatten()
    }
    fn imfilter_physical_evidence(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderImfilterPhysicalEvidence> {
        self.0.imfilter_physical_evidence(query)
    }
    fn operation_observation_capability(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderObservationCapability {
        self.0.operation_observation_capability(query)
    }
    fn begin_operation_observation(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderObservationSupport {
        self.0.begin_operation_observation(query)
    }
    fn imfilter<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        _: &'a ImfilterOptions,
    ) -> runmat_accelerate_api::AccelProviderFuture<'a, GpuTensorHandle> {
        self.1.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "fixture: direct imfilter hook unavailable",
            ))
        })
    }
}

#[test]
#[ignore = "requires actual WGPU adapter; execute in the native image-filter gate"]
fn explicit_unsupported_restores_exact_owner_after_one_completed_whole_call() {
    assert!(std::env::var("RUNMAT_WGPU_DISABLE_IMFILTER").is_err());
    let backend: &'static WgpuProvider = Box::leak(Box::new(
        WgpuProvider::new(Default::default()).expect("actual WGPU adapter required"),
    ));
    assert_eq!(
        backend.precision(),
        ProviderPrecision::F32,
        "run with RUNMAT_WGPU_FORCE_PRECISION=f32"
    );
    let facade: &'static UnsupportedImfilter = Box::leak(Box::new(UnsupportedImfilter(
        backend,
        AtomicUsize::new(0),
        false,
    )));
    let _owner = ThreadProviderGuard::set(Some(facade));
    let input = backend
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0, 3.0, 4.0]),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .expect("input upload")
        .with_provenance(GpuHandleProvenance::Explicit)
        .expect("explicit provenance");
    let kernel = Tensor::new(vec![1.0], vec![1, 1]).expect("host H");
    let args = [Value::GpuTensor(input.clone()), Value::Tensor(kernel)];
    let session = Rc::new(PlacementSession::new(Default::default()));
    let result = block_on(call_builtin_async_in_context(
        &context(session.clone(), 4096),
        "imfilter",
        &args,
    ))
    .expect("admitted explicit fallback");
    let Value::GpuTensor(output) = result else {
        panic!("explicit output must remain resident")
    };
    assert_eq!(output.device_id(), input.device_id());
    assert!(!output.shares_allocation_with(&input));
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(
        block_on(backend.download_numeric(&output))
            .expect("native readback")
            .data,
        HostNumericDataOwned::F32(vec![1.0, 2.0, 3.0, 4.0]),
    );
    assert_eq!(facade.1.load(Ordering::SeqCst), 1);
    assert!(
        (0..1000).any(|_| {
            if session.profile_snapshot().feedback.iter().any(|sample| {
                sample.signature.operation == "imfilter"
                    && sample.samples == 1
                    && sample.failures == 0
            }) {
                true
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1));
                false
            }
        }),
        "feedback must await restored upload's WGPU fence"
    );

    let before = backend.telemetry_snapshot();
    let hooks = facade.1.load(Ordering::SeqCst);
    let rejected = block_on(call_builtin_async_in_context(
        &context(Rc::new(PlacementSession::new(Default::default())), 0),
        "imfilter",
        &args,
    ))
    .expect_err("zero host quota must reject before H preparation");
    assert!(rejected
        .identifier()
        .is_some_and(|id| id.starts_with("RunMat:Placement:")));
    assert_eq!(facade.1.load(Ordering::SeqCst), hooks);
    let after = backend.telemetry_snapshot();
    assert_eq!(after.upload_bytes, before.upload_bytes);
    assert_eq!(after.download_bytes, before.download_bytes);
    input
        .allocation_lease()
        .validate_live()
        .expect("sibling survived rejection");

    let unavailable: &'static UnsupportedImfilter = Box::leak(Box::new(UnsupportedImfilter(
        backend,
        AtomicUsize::new(0),
        true,
    )));
    let _unavailable_owner = ThreadProviderGuard::set(Some(unavailable));
    let before = backend.telemetry_snapshot();
    let error = block_on(call_builtin_async_in_context(
        &context(Rc::new(PlacementSession::new(Default::default())), 4096),
        "imfilter",
        &args,
    ))
    .expect_err("finite quota cannot assume missing upload evidence is zero");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(unavailable.1.load(Ordering::SeqCst), 0);
    let after = backend.telemetry_snapshot();
    assert_eq!(after.upload_bytes, before.upload_bytes);
    assert_eq!(after.download_bytes, before.download_bytes);
}
