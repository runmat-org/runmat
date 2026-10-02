//! Exercise RGB whole-work dispatch in one runtime library. The acceleration
//! service implements this library's runtime ports, not a unit-test duplicate.
use futures::executor::block_on;
use runmat_accelerate::{placement::PlacementSession, simple_provider::InProcessProvider};
use runmat_accelerate_api::{
    handle_precision, provider_for_handle, same_provider_instance, AccelProvider,
    GpuHandleProvenance, ProviderPrecision, ThreadProviderGuard,
};
use runmat_execution::{
    Digest, PlacementDecision, PlacementFeedback, PlacementInvalidation, PlacementPlanRequest,
    PlacementRevision, WholeWorkDecision, WholeWorkDecisionKind, WholeWorkPlacementRequest,
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
};

#[derive(Default)]
struct Witness {
    selections: Cell<usize>,
    feedback: RefCell<Vec<PlacementFeedback>>,
    pending: RefCell<Vec<Box<dyn RuntimePlacementFeedbackSource>>>,
}

impl Witness {
    fn context(self: &Rc<Self>) -> RuntimeContext {
        base_context()
            .with_service_ports(RuntimeServicePorts::default().with_placement(self.clone()))
    }

    fn drain(&self) {
        let mut pending = std::mem::take(&mut *self.pending.borrow_mut());
        for mut source in pending.drain(..) {
            if let Some(feedback) = source.take_ready() {
                self.feedback.borrow_mut().push(feedback);
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
            provider,
            policy: digest,
        })
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        request.validate().expect("admitted RGB route");
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
        })
    }

    fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
        unreachable!("RGB receipt owns whole-work selection")
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
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

struct RejectPreparation;
impl RuntimeAccelerationService for RejectPreparation {
    fn supports_operation(&self, operation: &str) -> bool {
        matches!(operation, "rgb2gray" | "rgb2hsv")
    }
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
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
}

fn provider() -> (&'static InProcessProvider, ThreadProviderGuard) {
    let provider: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let guard = ThreadProviderGuard::set(Some(provider));
    (provider, guard)
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
        let Value::GpuTensor(output) = result else {
            panic!("{name} must restore to source owner")
        };
        assert!(!output.shares_allocation_with(&source));
        assert!(provider_for_handle(&output)
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
            .with_parallel(Rc::new(HostBudget(1))),
    );
    let runtime = runmat_accelerate::native_auto::install_preparation_service(runtime);
    provider.reset_telemetry();
    let error = block_on(call_builtin_async_in_context(
        &runtime,
        "rgb2gray",
        &[Value::GpuTensor(source.clone())],
    ))
    .expect_err("finite quota must reject before Sink(Gather)");
    assert!(error
        .identifier()
        .is_some_and(|id| id.starts_with("RunMat:Placement:")));
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
        let Value::GpuTensor(output) = block_on(call_builtin_async_in_context(
            &runtime,
            name,
            &[Value::GpuTensor(source.clone())],
        ))
        .unwrap() else {
            panic!("{name} output must be resident")
        };
        assert_eq!(handle_precision(&output), Some(expected));
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
        let Value::GpuTensor(output) = block_on(call_builtin_async_in_context(
            &runtime,
            name,
            &[Value::GpuTensor(source.clone())],
        ))
        .expect("resident color conversion") else {
            panic!("{name} must publish to the resident owner")
        };
        assert_eq!(output.shape(), source.shape());
        assert!(!output.shares_allocation_with(&source));
        assert!(provider_for_handle(&output)
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
    let Value::GpuTensor(output) = block_on(call_builtin_async_in_context(
        &runtime, "ind2rgb", &operands,
    ))
    .expect("two resident operands") else {
        panic!("indexed conversion must restore")
    };
    assert_eq!(output.shape(), &[1, 2, 3]);
    assert!(runmat_accelerate_api::handle_is_explicit(&output));
    assert!(!output.shares_allocation_with(&indexed));
    assert!(!output.shares_allocation_with(&map));
    assert_eq!(witness.selections.get(), 1);
    witness.drain();
    assert_eq!(witness.feedback.borrow().len(), 1);

    let quota = base_context().with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(Rc::new(PlacementSession::new(Default::default())))
            .with_parallel(Rc::new(HostBudget(1))),
    );
    let quota = runmat_accelerate::native_auto::install_preparation_service(quota);
    provider.reset_telemetry();
    let error = block_on(call_builtin_async_in_context(&quota, "ind2rgb", &operands))
        .expect_err("finite quota must reject both readbacks before effects");
    assert!(error
        .identifier()
        .is_some_and(|id| id.starts_with("RunMat:Placement:")));
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
    assert!(indexed.allocation_lease().validate_live().is_ok());
    assert!(map.allocation_lease().validate_live().is_ok());
}
