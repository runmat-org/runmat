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
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct HostBudget(runmat_runtime::context::host_effect::ManagedHostEffectLedger);
impl RuntimeParallelService for HostBudget {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
        }
    }
    fn managed_host_effect_ledger(
        &self,
    ) -> Option<runmat_runtime::context::host_effect::ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

fn context(session: Rc<PlacementSession>, host_bytes: u64) -> RuntimeContext {
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(session)
            .with_parallel(Rc::new(HostBudget(
                runmat_runtime::context::host_effect::ManagedHostEffectLedger::new(host_bytes),
            ))),
    );
    context.set_runmat_extensions_enabled(true);
    context
}

struct UnsupportedImfilter(&'static WgpuProvider, AtomicUsize, bool, AtomicBool);
impl AccelProvider for UnsupportedImfilter {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.0.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.0.download(handle)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.3.swap(false, Ordering::SeqCst) {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "fixture: H preparation unavailable",
            )
            .into());
        }
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
        AtomicBool::new(false),
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
    let Value::GpuTensor(output) = result.value() else {
        panic!("explicit output must remain resident")
    };
    assert_eq!(output.device_id(), input.device_id());
    assert!(!output.shares_allocation_with(&input));
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(
        block_on(backend.download_numeric(output))
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
        AtomicBool::new(false),
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

    let preparation: &'static UnsupportedImfilter = Box::leak(Box::new(UnsupportedImfilter(
        backend,
        AtomicUsize::new(0),
        false,
        AtomicBool::new(true),
    )));
    let _preparation_owner = ThreadProviderGuard::set(Some(preparation));
    let session = Rc::new(PlacementSession::new(Default::default()));
    let prepared_result = block_on(call_builtin_async_in_context(
        &context(session.clone(), 4096),
        "imfilter",
        &args,
    ))
    .expect("admitted preparation Unsupported continuation");
    let Value::GpuTensor(prepared_output) = prepared_result.value() else {
        panic!("explicit preparation fallback must restore the original owner")
    };
    assert_eq!(preparation.1.load(Ordering::SeqCst), 0);
    assert_eq!(prepared_output.device_id(), input.device_id());
    assert_eq!(
        block_on(backend.download_numeric(prepared_output))
            .expect("prepared fallback readback")
            .data,
        HostNumericDataOwned::F32(vec![1.0, 2.0, 3.0, 4.0]),
    );
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
        "preparation Unsupported must have one completion-qualified receipt"
    );
}

#[test]
#[ignore = "requires actual WGPU adapter; run this exact test alone in the native image-filter gate"]
fn actual_provider_host_mediated_route_has_one_completed_whole_call() {
    struct ResetImfilterDisable;
    impl Drop for ResetImfilterDisable {
        fn drop(&mut self) {
            std::env::remove_var("RUNMAT_WGPU_DISABLE_IMFILTER");
        }
    }
    assert!(
        std::env::var("RUNMAT_WGPU_DISABLE_IMFILTER").is_err(),
        "run exact test with no pre-existing disabled-filter setting"
    );
    std::env::set_var("RUNMAT_WGPU_DISABLE_IMFILTER", "1");
    let _restore_env = ResetImfilterDisable;
    let backend: &'static WgpuProvider = Box::leak(Box::new(
        WgpuProvider::new(Default::default()).expect("actual WGPU adapter required"),
    ));
    assert_eq!(backend.precision(), ProviderPrecision::F32);
    let _owner = ThreadProviderGuard::set(Some(backend));
    let input = backend
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0, 3.0, 4.0]),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .expect("input upload")
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let kernel = backend
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .expect("kernel upload");
    let query = runmat_runtime::builtins::image::filters::placement::exact_hook_query(
        &input,
        &kernel,
        &ImfilterOptions::default(),
    )
    .expect("live exact query");
    assert_eq!(
        backend.imfilter_physical_evidence(&query).unwrap().route,
        runmat_accelerate_api::ProviderImfilterPhysicalRoute::HostMediatedEarly
    );
    let args = [
        Value::GpuTensor(input.clone()),
        Value::GpuTensor(kernel.clone()),
    ];
    let session = Rc::new(PlacementSession::new(Default::default()));
    let result = block_on(call_builtin_async_in_context(
        &context(session.clone(), 4096),
        "imfilter",
        &args,
    ))
    .expect("admitted provider-owned host route");
    let Value::GpuTensor(output) = result.value() else {
        panic!("explicit output must remain resident")
    };
    assert_eq!(output.device_id(), input.device_id());
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(
        block_on(backend.download_numeric(output)).unwrap().data,
        HostNumericDataOwned::F32(vec![1.0, 2.0, 3.0, 4.0])
    );
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
        "whole-call feedback requires the provider's final upload fence"
    );
}

#[test]
#[ignore = "requires actual WGPU adapter; run in native image-filter qualification"]
fn automatic_integer_two_readbacks_obey_finite_host_quota_before_effects() {
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_value::IntegerStorage;
    assert!(std::env::var("RUNMAT_WGPU_DISABLE_IMFILTER").is_err());
    let backend: &'static WgpuProvider = Box::leak(Box::new(
        WgpuProvider::new(Default::default()).expect("actual WGPU adapter required"),
    ));
    let _owner = ThreadProviderGuard::set(Some(backend));
    let image = backend
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U8(&[1, 2, 3, 4]),
            shape: &[2, 2],
        })
        .expect("native integer image");
    let kernel = backend
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .expect("resident double H");
    let args = [
        Value::GpuTensor(image.clone()),
        Value::GpuTensor(kernel.clone()),
    ];
    let before = backend.telemetry_snapshot();
    let error = block_on(call_builtin_async_in_context(
        &context(Rc::new(PlacementSession::new(Default::default())), 0),
        "imfilter",
        &args,
    ))
    .expect_err("finite zero host quota must reject before either readback");
    assert!(error
        .identifier()
        .is_some_and(|id| id.starts_with("RunMat:Placement:")));
    let after = backend.telemetry_snapshot();
    assert_eq!(after.download_bytes, before.download_bytes);
    assert_eq!(after.upload_bytes, before.upload_bytes);
    image.allocation_lease().validate_live().unwrap();
    kernel.allocation_lease().validate_live().unwrap();

    let session = Rc::new(PlacementSession::new(Default::default()));
    let result = block_on(call_builtin_async_in_context(
        &context(session.clone(), 4096),
        "imfilter",
        &args,
    ))
    .expect("admitted integer host result");
    let Value::Tensor(output) = result.value() else {
        panic!("automatic integer result stays host-side")
    };
    assert_eq!(output.shape(), [2, 2]);
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U8(vec![1, 2, 3, 4]))
    );
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
        "one whole-call host-sink feedback sample"
    );
    image.allocation_lease().validate_live().unwrap();
    kernel.allocation_lease().validate_live().unwrap();
}

#[test]
#[ignore = "requires actual WGPU adapter; run in native image-filter qualification"]
fn explicit_integer_and_logical_restore_keep_exact_class_owner_and_fence() {
    use runmat_accelerate_api::{
        HostIntegerDataView, HostIntegerTensorView, HostLogicalTensorView,
    };
    assert!(std::env::var("RUNMAT_WGPU_DISABLE_IMFILTER").is_err());
    let backend: &'static WgpuProvider = Box::leak(Box::new(
        WgpuProvider::new(Default::default()).expect("actual WGPU adapter required"),
    ));
    assert_eq!(backend.precision(), ProviderPrecision::F32);
    let _owner = ThreadProviderGuard::set(Some(backend));
    let integer = backend
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U8(&[1, 2, 3, 4]),
            shape: &[2, 2],
        })
        .expect("integer upload")
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let logical = backend
        .upload_logical(&HostLogicalTensorView {
            data: &[0, 1, 1, 0],
            shape: &[2, 2],
        })
        .expect("logical upload")
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let kernel = backend
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .expect("double H");
    for (image, expected, logical_class) in [
        (integer, HostNumericDataOwned::U8(vec![1, 2, 3, 4]), false),
        (
            logical,
            HostNumericDataOwned::F32(vec![0.0, 1.0, 1.0, 0.0]),
            true,
        ),
    ] {
        let args = [
            Value::GpuTensor(image.clone()),
            Value::GpuTensor(kernel.clone()),
        ];
        let before = backend.telemetry_snapshot();
        let error = block_on(call_builtin_async_in_context(
            &context(Rc::new(PlacementSession::new(Default::default())), 0),
            "imfilter",
            &args,
        ))
        .expect_err("finite zero host quota rejects before H readback");
        assert!(error
            .identifier()
            .is_some_and(|id| id.starts_with("RunMat:Placement:")));
        let after = backend.telemetry_snapshot();
        assert_eq!(after.download_bytes, before.download_bytes);
        assert_eq!(after.upload_bytes, before.upload_bytes);
        let session = Rc::new(PlacementSession::new(Default::default()));
        let result = block_on(call_builtin_async_in_context(
            &context(session.clone(), 4096),
            "imfilter",
            &args,
        ))
        .expect("admitted class-preserving restore");
        let Value::GpuTensor(output) = result.value() else {
            panic!("explicit result remains on original owner")
        };
        assert_eq!(output.device_id(), image.device_id());
        assert_eq!(output.shape(), &[2, 2]);
        assert!(!output.shares_allocation_with(&image));
        assert!(!output.shares_allocation_with(&kernel));
        assert_eq!(
            runmat_accelerate_api::handle_is_logical(output),
            logical_class
        );
        assert_eq!(
            block_on(backend.download_numeric(output)).unwrap().data,
            expected
        );
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
            "one completion-qualified class-preserving upload sample"
        );
        image.allocation_lease().validate_live().unwrap();
        kernel.allocation_lease().validate_live().unwrap();
    }
}
