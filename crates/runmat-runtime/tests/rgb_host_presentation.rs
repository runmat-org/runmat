//! Exercise RGB whole-work dispatch in one runtime library. The acceleration
//! service implements this library's runtime ports, not a unit-test duplicate.
use futures::executor::block_on;
use runmat_accelerate::{placement::PlacementSession, simple_provider::InProcessProvider};
use runmat_accelerate_api::{
    handle_precision, provider_for_handle, same_provider_instance, AccelDownloadFuture,
    AccelNumericDownloadFuture, AccelProvider, GpuHandleProvenance, GpuTensorHandle,
    HostNumericTensorView, HostTensorView, ProviderPrecision, ThreadProviderGuard,
};
use runmat_execution::{
    Digest, PlacementDecision, PlacementFeedback, PlacementInvalidation, PlacementPlanRequest,
    PlacementRevision, ProviderResourceSnapshot, ResourceByteBound, WholeWorkDecision,
    WholeWorkDecisionKind, WholeWorkPlacementRequest,
};
use runmat_runtime::{
    builtins::common::gpu_helpers,
    call_builtin_async_in_context,
    context::{
        preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands},
        ParallelCapability, RuntimeAccelerationService, RuntimeContext, RuntimeParallelResources,
        RuntimeParallelService, RuntimePlacementFeedbackSource, RuntimePlacementService,
        RuntimeServiceFuture, RuntimeServicePorts, RuntimeWholeWorkSelection,
    },
    RuntimeError,
};
use runmat_value::{IntegerStorage, Tensor, Value};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Default)]
struct Witness {
    selections: Cell<usize>,
    feedback: RefCell<Vec<PlacementFeedback>>,
    pending: RefCell<Vec<Box<dyn RuntimePlacementFeedbackSource>>>,
    admission: Option<PlacementSession>,
}

impl Witness {
    fn budgeted() -> Rc<Self> {
        Rc::new(Self {
            admission: Some(PlacementSession::new(Default::default())),
            ..Default::default()
        })
    }
    fn context(self: &Rc<Self>) -> RuntimeContext {
        base_context()
            .with_service_ports(RuntimeServicePorts::default().with_placement(self.clone()))
    }

    fn drain(&self) {
        let mut pending = std::mem::take(&mut *self.pending.borrow_mut());
        for mut source in pending.drain(..) {
            if let Some(feedback) = source.take_ready() {
                let snapshot = feedback.snapshot().expect("completed presentation");
                if let Some(admission) = &self.admission {
                    RuntimePlacementService::observe_deferred_feedback(admission, feedback)
                        .expect("admitted feedback");
                }
                self.feedback.borrow_mut().push(snapshot);
            } else if !source.terminal_without_feedback() {
                self.pending.borrow_mut().push(source);
            }
        }
    }
}

impl RuntimePlacementService for Witness {
    fn whole_work_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        provider: Digest,
    ) -> Option<PlacementRevision> {
        let digest = Digest::sha256(b"rgb-host-presentation-witness");
        Some(PlacementRevision {
            program,
            catalog: digest,
            compiler: digest,
            authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(provider),
            policy: digest,
        })
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        request.validate().expect("admitted RGB route");
        if let Some(admission) = &self.admission {
            let selected = RuntimePlacementService::select_whole_work(admission, request)?;
            self.selections.set(self.selections.get() + 1);
            return Ok(selected);
        }
        let candidate = request.candidates.first().expect("host upload candidate");
        self.selections.set(self.selections.get() + 1);
        Ok(RuntimeWholeWorkSelection {
            decision: WholeWorkDecision {
                signature: request.signature.clone(),
                selected: candidate.selected(&request.signature.operation),
                kind: WholeWorkDecisionKind::Unmeasured,
                predicted_total_ns: None,
            },
            probe: None,
            observation_permit: None,
        })
    }

    fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
        unreachable!("RGB receipt owns whole-work selection")
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        if let Some(admission) = &self.admission {
            RuntimePlacementService::observe(admission, feedback.clone())?;
        }
        self.feedback.borrow_mut().push(feedback);
        Ok(())
    }

    fn defer_feedback(
        &self,
        source: Box<dyn RuntimePlacementFeedbackSource>,
    ) -> Result<(), Box<dyn RuntimePlacementFeedbackSource>> {
        self.pending.borrow_mut().push(source);
        Ok(())
    }

    fn invalidate(&self, _: PlacementInvalidation) {}
}

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

fn assert_argument_capacity_refusal(error: &RuntimeError) {
    use runmat_runtime::context::{
        host_allocation::HostAllocationError, host_effect::HostEffectAllocationError,
    };
    // A one-byte whole-call budget cannot admit even its exact Value slots.
    // This original ingress refusal precedes placement and provider effects.
    assert_eq!(error.identifier(), Some("RunMat:Call:Arguments"));
    assert!(matches!(
        std::error::Error::source(error)
            .and_then(|cause| cause.downcast_ref::<HostAllocationError>()),
        Some(HostAllocationError::Resource(
            HostEffectAllocationError::ExceedsCapacity
        ))
    ));
    assert_eq!(
        error.gpu_gather_retry(),
        runmat_runtime::GpuGatherRetry::Never
    );
}

struct RejectPreparation;
impl RuntimeAccelerationService for RejectPreparation {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        _: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        Box::pin(async {
            Err(
                runmat_runtime::build_runtime_error("RGB preparation failed")
                    .with_identifier("RunMat:Preparation:FixtureFailure")
                    .build(),
            )
        })
    }
}

fn base_context() -> RuntimeContext {
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    context
        .set_runmat_extensions_enabled(runmat_runtime::compatibility::runmat_extensions_enabled());
    context
}

fn provider() -> (&'static InProcessProvider, ThreadProviderGuard) {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    // Distinct resident owners remain addressable after thread selection
    // switches to the output provider. Each leaked fixture has a unique id.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    let guard = ThreadProviderGuard::set(Some(provider));
    (provider, guard)
}

/// A real in-process source with a finite, nonzero download staging demand.
/// Its output-owner peer remains independently budgeted.
struct FiniteReadbackOwner {
    inner: InProcessProvider,
    scratch_available: AtomicU64,
}

impl AccelProvider for FiniteReadbackOwner {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
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
    fn capability_snapshot(&self) -> runmat_accelerate_api::ProviderCapabilitySnapshot {
        self.inner.capability_snapshot()
    }
    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        self.inner.query_feasibility(query)
    }
    fn download_numeric_host_workspace_bound(&self, handle: &GpuTensorHandle) -> ResourceByteBound {
        self.inner.download_numeric_host_workspace_bound(handle)
    }
    fn download_numeric_provider_scratch_bound(&self, _: &GpuTensorHandle) -> ResourceByteBound {
        ResourceByteBound::Known(8)
    }
    fn download_numeric_queue_slots(&self, _: &GpuTensorHandle) -> u32 {
        1
    }
    fn placement_resources(&self) -> ProviderResourceSnapshot {
        ProviderResourceSnapshot {
            device_id: self.device_id(),
            scratch_available_bytes: Some(self.scratch_available.load(Ordering::Relaxed)),
            queue_depth: Some(0),
            queue_limit: Some(1),
            ..Default::default()
        }
    }
}

fn resident_u8(provider: &'static InProcessProvider) -> runmat_accelerate_api::GpuTensorHandle {
    let rgb = Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).unwrap();
    gpu_helpers::upload_tensor(provider, &rgb).unwrap()
}

#[test]
fn both_rgb_calls_select_one_owner_upload_and_observe_complete_publication() {
    let (provider, _guard) = provider();
    let source = resident_u8(provider)
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    for name in ["rgb2gray", "rgb2hsv"] {
        let witness = Rc::new(Witness::default());
        let runtime =
            runmat_accelerate::native_auto::install_preparation_service(witness.context());
        let result = block_on(call_builtin_async_in_context(
            &runtime,
            name,
            &[Value::GpuTensor(source.clone())],
        ))
        .unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("{name} must restore to source owner")
        };
        assert!(!output.shares_allocation_with(&source));
        assert!(provider_for_handle(output)
            .is_some_and(|owner| same_provider_instance(owner, provider)));
        assert_eq!(
            witness.selections.get(),
            1,
            "{name} selected one whole route"
        );
        witness.drain();
        assert_eq!(
            witness.feedback.borrow().len(),
            1,
            "{name} has one completed sample"
        );
    }
    assert!(source.allocation_lease().validate_live().is_ok());
}

#[test]
fn finite_host_quota_rejects_rgb_route_before_readback_or_upload() {
    let (provider, _guard) = provider();
    let source = resident_u8(provider);
    let runtime = base_context().with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(Rc::new(PlacementSession::new(Default::default())))
            .with_parallel(Rc::new(HostBudget(
                runmat_runtime::context::host_effect::ManagedHostEffectLedger::new(1),
            ))),
    );
    let runtime = runmat_accelerate::native_auto::install_preparation_service(runtime);
    provider.reset_telemetry();
    let error = block_on(call_builtin_async_in_context(
        &runtime,
        "rgb2gray",
        &[Value::GpuTensor(source.clone())],
    ))
    .expect_err("finite quota must reject before Sink(Gather)");
    assert_argument_capacity_refusal(&error);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
    assert!(source.allocation_lease().validate_live().is_ok());
}

#[test]
fn one_pixel_single_gray_upload_is_typed_double_scalar_hsv_stays_single() {
    let (provider, _guard) = provider();
    let rgb = Tensor::from_f32(vec![1.0, 0.0, 0.0], vec![1, 1, 3]).unwrap();
    let source = gpu_helpers::upload_tensor(provider, &rgb).unwrap();
    for (name, expected) in [
        ("rgb2gray", ProviderPrecision::F64),
        ("rgb2hsv", ProviderPrecision::F32),
    ] {
        let witness = Rc::new(Witness::default());
        let runtime =
            runmat_accelerate::native_auto::install_preparation_service(witness.context());
        let result = block_on(call_builtin_async_in_context(
            &runtime,
            name,
            &[Value::GpuTensor(source.clone())],
        ))
        .unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("{name} output must be resident")
        };
        assert_eq!(handle_precision(output), Some(expected));
        assert_eq!(witness.selections.get(), 1);
    }
}

#[test]
fn unpolled_rgb_call_has_no_selection_or_provider_effects() {
    let (provider, _guard) = provider();
    let source = resident_u8(provider);
    let witness = Rc::new(Witness::default());
    let runtime = runmat_accelerate::native_auto::install_preparation_service(witness.context());
    provider.reset_telemetry();
    let operands = [Value::GpuTensor(source.clone())];
    let future = call_builtin_async_in_context(&runtime, "rgb2hsv", &operands);
    drop(future);
    assert_eq!(witness.selections.get(), 0);
    assert!(witness.feedback.borrow().is_empty());
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
    assert!(source.allocation_lease().validate_live().is_ok());
}

#[test]
fn failed_preparation_abandons_selected_rgb_receipt_without_upload() {
    let (provider, _guard) = provider();
    let source = resident_u8(provider);
    let witness = Rc::new(Witness::default());
    let context = witness.context();
    let ports = context
        .service_ports()
        .clone()
        .with_acceleration(Rc::new(RejectPreparation));
    let runtime = context.with_service_ports(ports);
    provider.reset_telemetry();
    let error = block_on(call_builtin_async_in_context(
        &runtime,
        "rgb2gray",
        &[Value::GpuTensor(source.clone())],
    ))
    .expect_err("preparation failure must not replay or publish");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Preparation:FixtureFailure")
    );
    assert_eq!(witness.selections.get(), 1);
    witness.drain();
    assert!(witness.feedback.borrow().is_empty());
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
    assert!(source.allocation_lease().validate_live().is_ok());
}

#[test]
fn restored_color_conversions_select_once_and_observe_exact_owner_upload() {
    let (provider, _guard) = provider();
    let source = gpu_helpers::upload_tensor(
        provider,
        &Tensor::new(vec![0.5, 0.2, 0.8], vec![1, 1, 3]).unwrap(),
    )
    .unwrap();
    for name in ["hsv2rgb", "lab2rgb", "rgb2lab"] {
        let witness = Rc::new(Witness::default());
        let runtime =
            runmat_accelerate::native_auto::install_preparation_service(witness.context());
        let result = block_on(call_builtin_async_in_context(
            &runtime,
            name,
            &[Value::GpuTensor(source.clone())],
        ))
        .expect("resident color conversion");
        let Value::GpuTensor(output) = result.value() else {
            panic!("{name} must publish to the resident owner")
        };
        assert_eq!(output.shape(), source.shape());
        assert!(!output.shares_allocation_with(&source));
        assert!(provider_for_handle(output)
            .is_some_and(|owner| same_provider_instance(owner, provider)));
        assert_eq!(
            witness.selections.get(),
            1,
            "{name} selects one whole route"
        );
        witness.drain();
        assert_eq!(
            witness.feedback.borrow().len(),
            1,
            "{name} observes upload completion"
        );
    }
    assert!(source.allocation_lease().validate_live().is_ok());
}

#[test]
fn indexed_color_two_resident_readbacks_select_one_owner_and_finite_quota_rejects_before_effects() {
    let (provider, _guard) = provider();
    let indexed = gpu_helpers::upload_tensor(
        provider,
        &Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap(),
    )
    .unwrap();
    let map = gpu_helpers::upload_tensor(
        provider,
        &Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap(),
    )
    .unwrap();
    let map = map.with_provenance(GpuHandleProvenance::Explicit).unwrap();
    let operands = [
        Value::GpuTensor(indexed.clone()),
        Value::GpuTensor(map.clone()),
    ];

    let witness = Rc::new(Witness::default());
    let runtime = runmat_accelerate::native_auto::install_preparation_service(witness.context());
    let result = block_on(call_builtin_async_in_context(
        &runtime, "ind2rgb", &operands,
    ))
    .expect("two resident operands");
    let Value::GpuTensor(output) = result.value() else {
        panic!("indexed conversion must restore")
    };
    assert_eq!(output.shape(), &[1, 2, 3]);
    assert!(runmat_accelerate_api::handle_is_explicit(output));
    assert!(!output.shares_allocation_with(&indexed));
    assert!(!output.shares_allocation_with(&map));
    assert_eq!(witness.selections.get(), 1);
    witness.drain();
    assert_eq!(witness.feedback.borrow().len(), 1);

    let quota = base_context().with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(Rc::new(PlacementSession::new(Default::default())))
            .with_parallel(Rc::new(HostBudget(
                runmat_runtime::context::host_effect::ManagedHostEffectLedger::new(1),
            ))),
    );
    let quota = runmat_accelerate::native_auto::install_preparation_service(quota);
    provider.reset_telemetry();
    let error = block_on(call_builtin_async_in_context(&quota, "ind2rgb", &operands))
        .expect_err("finite quota must reject both readbacks before effects");
    assert_argument_capacity_refusal(&error);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
    assert!(indexed.allocation_lease().validate_live().is_ok());
    assert!(map.allocation_lease().validate_live().is_ok());
}

#[test]
fn indexed_color_gathers_two_providers_and_restores_only_to_explicit_map_owner() {
    let (index_provider, _index_guard) = provider();
    let (map_provider, _map_guard) = provider();
    assert_ne!(index_provider.device_id(), map_provider.device_id());
    // Both leaked fixtures remain alive for the process lifetime and own
    // distinct device namespaces. Ambient selection alone addresses only one.
    unsafe {
        runmat_accelerate_api::register_device_provider(index_provider);
        runmat_accelerate_api::register_device_provider(map_provider);
    }
    let indexed = gpu_helpers::upload_tensor(
        index_provider,
        &Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap(),
    )
    .unwrap();
    let map = gpu_helpers::upload_tensor(
        map_provider,
        &Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap(),
    )
    .unwrap()
    .with_provenance(GpuHandleProvenance::Explicit)
    .unwrap();
    index_provider.reset_telemetry();
    map_provider.reset_telemetry();
    let witness = Rc::new(Witness::default());
    let runtime = runmat_accelerate::native_auto::install_preparation_service(witness.context());
    let result = block_on(call_builtin_async_in_context(
        &runtime,
        "ind2rgb",
        &[
            Value::GpuTensor(indexed.clone()),
            Value::GpuTensor(map.clone()),
        ],
    ))
    .expect("cross-provider indexed conversion");
    let Value::GpuTensor(output) = result.value() else {
        panic!("cross-provider output must restore to explicit map owner")
    };
    assert_eq!(output.shape(), &[1, 2, 3]);
    assert!(provider_for_handle(output)
        .is_some_and(|owner| same_provider_instance(owner, map_provider)));
    assert!(runmat_accelerate_api::handle_is_explicit(output));
    assert!(!output.shares_allocation_with(&indexed));
    assert!(!output.shares_allocation_with(&map));
    assert_eq!(index_provider.telemetry_snapshot().upload_bytes, 0);
    assert!(index_provider.telemetry_snapshot().download_bytes > 0);
    assert!(map_provider.telemetry_snapshot().download_bytes > 0);
    assert!(map_provider.telemetry_snapshot().upload_bytes > 0);
    assert_eq!(witness.selections.get(), 1);
    witness.drain();
    assert_eq!(witness.feedback.borrow().len(), 1);
    assert!(indexed.allocation_lease().validate_live().is_ok());
    assert!(map.allocation_lease().validate_live().is_ok());
}

#[test]
fn cross_owner_readback_staging_exact_boundary_and_preeffect_rejection() {
    let index_provider: &'static FiniteReadbackOwner = Box::leak(Box::new(FiniteReadbackOwner {
        inner: InProcessProvider::new(),
        scratch_available: AtomicU64::new(8),
    }));
    // The leaked facade is the sole registered owner of its unique namespace.
    unsafe { runmat_accelerate_api::register_device_provider(index_provider) };
    let _index_guard = ThreadProviderGuard::set(Some(index_provider));
    let indexed = gpu_helpers::upload_tensor(
        index_provider,
        &Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap(),
    )
    .unwrap();
    let (map_provider, _map_guard) = provider();
    let map = gpu_helpers::upload_tensor(
        map_provider,
        &Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap(),
    )
    .unwrap()
    .with_provenance(GpuHandleProvenance::Explicit)
    .unwrap();
    assert_ne!(indexed.device_id(), map.device_id());

    let witness = Witness::budgeted();
    let runtime = runmat_accelerate::native_auto::install_preparation_service(witness.context());
    index_provider.inner.reset_telemetry();
    map_provider.reset_telemetry();
    let result = block_on(call_builtin_async_in_context(
        &runtime,
        "ind2rgb",
        &[
            Value::GpuTensor(indexed.clone()),
            Value::GpuTensor(map.clone()),
        ],
    ))
    .expect("exact source staging quota admits cross-owner gather");
    let Value::GpuTensor(output) = result.value() else {
        panic!("indexed result stays with explicit map owner")
    };
    assert_eq!(output.shape(), &[1, 2, 3]);
    assert!(provider_for_handle(output)
        .is_some_and(|owner| same_provider_instance(owner, map_provider)));
    assert_eq!(witness.selections.get(), 1);
    witness.drain();
    assert_eq!(witness.feedback.borrow().len(), 1);
    assert!(witness.feedback.borrow()[0].succeeded);
    assert!(index_provider.inner.telemetry_snapshot().download_bytes > 0);
    let host = block_on(map_provider.download(output)).unwrap();
    assert_eq!(host.data, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    index_provider.scratch_available.store(7, Ordering::Relaxed);
    index_provider.inner.reset_telemetry();
    map_provider.reset_telemetry();
    let error = block_on(call_builtin_async_in_context(
        &runtime,
        "ind2rgb",
        &[
            Value::GpuTensor(indexed.clone()),
            Value::GpuTensor(map.clone()),
        ],
    ))
    .expect_err("source staging quota must reject the complete route");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:NoLegalCandidate")
    );
    assert_eq!(index_provider.inner.telemetry_snapshot().download_bytes, 0);
    assert_eq!(index_provider.inner.telemetry_snapshot().upload_bytes, 0);
    assert_eq!(map_provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(map_provider.telemetry_snapshot().upload_bytes, 0);
    assert!(indexed.allocation_lease().validate_live().is_ok());
    assert!(map.allocation_lease().validate_live().is_ok());
    assert!(output.allocation_lease().validate_live().is_ok());
}

#[test]
fn gray_host_sink_observes_completed_readback_and_replicated_host_result() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let (provider, _guard) = provider();
    let source =
        gpu_helpers::upload_tensor(provider, &Tensor::new(vec![0.25, 0.5], vec![1, 2]).unwrap())
            .unwrap();
    let witness = Rc::new(Witness::default());
    let runtime = runmat_accelerate::native_auto::install_preparation_service(witness.context());
    provider.reset_telemetry();
    let result = block_on(call_builtin_async_in_context(
        &runtime,
        "gray2rgb",
        &[Value::GpuTensor(source.clone())],
    ))
    .expect("resident gray host sink");
    let Value::Tensor(output) = result.value() else {
        panic!("gray2rgb publishes host Tensor")
    };
    assert_eq!(output.shape(), &[1, 2, 3]);
    assert_eq!(
        output.materialize_f64(),
        vec![0.25, 0.5, 0.25, 0.5, 0.25, 0.5]
    );
    assert!(provider.telemetry_snapshot().download_bytes > 0);
    assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
    assert_eq!(witness.selections.get(), 1);
    witness.drain();
    assert_eq!(witness.feedback.borrow().len(), 1);

    let quota = base_context().with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(Rc::new(PlacementSession::new(Default::default())))
            .with_parallel(Rc::new(HostBudget(
                runmat_runtime::context::host_effect::ManagedHostEffectLedger::new(1),
            ))),
    );
    let quota = runmat_accelerate::native_auto::install_preparation_service(quota);
    provider.reset_telemetry();
    let error = block_on(call_builtin_async_in_context(
        &quota,
        "gray2rgb",
        &[Value::GpuTensor(source.clone())],
    ))
    .expect_err("finite host quota rejects before readback");
    assert_argument_capacity_refusal(&error);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);

    let unpolled_operands = [Value::GpuTensor(source.clone())];
    let unpolled = call_builtin_async_in_context(&runtime, "gray2rgb", &unpolled_operands);
    drop(unpolled);
    assert_eq!(witness.selections.get(), 1);
    assert!(source.allocation_lease().validate_live().is_ok());
}

#[test]
fn failed_gray_preparation_abandons_host_sink_without_feedback_or_readback() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let (provider, _guard) = provider();
    let source =
        gpu_helpers::upload_tensor(provider, &Tensor::new(vec![0.5], vec![1, 1]).unwrap()).unwrap();
    let witness = Rc::new(Witness::default());
    let context = witness.context();
    let ports = context
        .service_ports()
        .clone()
        .with_acceleration(Rc::new(RejectPreparation));
    let runtime = context.with_service_ports(ports);
    provider.reset_telemetry();
    let error = block_on(call_builtin_async_in_context(
        &runtime,
        "gray2rgb",
        &[Value::GpuTensor(source.clone())],
    ))
    .expect_err("preparation must fail without effect replay");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Preparation:FixtureFailure")
    );
    assert_eq!(witness.selections.get(), 1);
    witness.drain();
    assert!(witness.feedback.borrow().is_empty());
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
    assert!(source.allocation_lease().validate_live().is_ok());
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires an actual WGPU adapter; run explicitly for F32 and F64"]
fn actual_wgpu_gray_host_sink_completes_nonempty_empty_and_lazy_readbacks() {
    use runmat_accelerate::backend::wgpu::provider::WgpuProvider;
    use runmat_accelerate_api::{GpuTensorLayout, NumericElementType, TransposeInfo};
    use runmat_execution::ResourceByteBound;
    use runmat_value::NumericDType;

    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let provider: &'static WgpuProvider = Box::leak(Box::new(
        WgpuProvider::new(Default::default()).expect("actual WGPU adapter required"),
    ));
    let _guard = ThreadProviderGuard::set(Some(provider));
    match std::env::var("RUNMAT_WGPU_FORCE_PRECISION").as_deref() {
        Ok("f32") => assert_eq!(provider.precision(), ProviderPrecision::F32),
        Ok("f64") => assert_eq!(provider.precision(), ProviderPrecision::F64),
        _ => panic!("set RUNMAT_WGPU_FORCE_PRECISION=f32 or f64 for the required hardware gate"),
    }
    let dtype = match provider.precision() {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    };
    let expected_element = match provider.precision() {
        ProviderPrecision::F32 => NumericElementType::F32,
        ProviderPrecision::F64 => NumericElementType::F64,
    };
    for (name, values, shape, lazy, expected_plane) in [
        (
            "nonempty",
            vec![0.25, 0.5],
            vec![1, 2],
            false,
            vec![0.25, 0.5],
        ),
        ("empty", vec![], vec![0, 2], false, vec![]),
        (
            "lazy",
            vec![1., 2., 3., 4., 5., 6.],
            vec![2, 3],
            true,
            vec![1., 3., 5., 2., 4., 6.],
        ),
    ] {
        let tensor = Tensor::new_with_dtype(values, shape, dtype).unwrap();
        let backing = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        let source = if lazy {
            backing
                .with_view(backing.view().clone().with_shape(vec![3, 2]).with_layout(
                    GpuTensorLayout::Transpose(TransposeInfo {
                        base_rows: 2,
                        base_cols: 3,
                    }),
                ))
                .expect("actual owner admits lazy view")
        } else {
            backing.clone()
        };
        assert_eq!(source.physical().element_type(), expected_element);
        let expected_staging = if source.physical().element_capacity() == 0 {
            0
        } else {
            source.physical().requested_bytes()
        };
        assert_eq!(
            provider.download_numeric_provider_scratch_bound(&source),
            ResourceByteBound::Known(expected_staging),
            "{name} staging"
        );
        let witness = Rc::new(Witness::default());
        let runtime =
            runmat_accelerate::native_auto::install_preparation_service(witness.context());
        provider.reset_telemetry();
        let result = block_on(call_builtin_async_in_context(
            &runtime,
            "gray2rgb",
            &[Value::GpuTensor(source.clone())],
        ))
        .expect("required actual WGPU gray readback");
        let Value::Tensor(output) = result.value() else {
            panic!("{name} must publish host tensor")
        };
        assert_eq!(
            output.shape(),
            vec![source.shape()[0], source.shape()[1], 3]
        );
        assert_eq!(output.numeric_dtype(), dtype);
        let actual = output.materialize_f64();
        let expected = [
            expected_plane.as_slice(),
            expected_plane.as_slice(),
            expected_plane.as_slice(),
        ]
        .concat();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (*actual - expected).abs() <= 1e-6,
                "{name}: {actual} != {expected}"
            );
        }
        assert_eq!(
            provider.telemetry_snapshot().upload_bytes,
            0,
            "{name} must not upload"
        );
        assert_eq!(
            provider.telemetry_snapshot().download_bytes,
            expected_staging,
            "{name} physical readback bytes"
        );
        assert_eq!(witness.selections.get(), 1, "{name} one whole route");
        witness.drain();
        assert_eq!(
            witness.feedback.borrow().len(),
            1,
            "{name} completed whole call"
        );
        assert!(source.allocation_lease().validate_live().is_ok());
    }
}
