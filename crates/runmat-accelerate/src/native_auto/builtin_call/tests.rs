use super::*;
mod admitted;
mod admitted_owners;
#[cfg(feature = "wgpu")]
mod floating_cast;
mod host_fallback;
mod integer_conversion;
mod ownership;
mod pol2cart;
mod receipt;
use crate::placement::{begin_trace, LocalPlacementFeedback};
use runmat_execution::{
    Digest, PlacementFeedback, PlacementPlanRequest, PlacementRevision, PlacementSignature,
};
use runmat_runtime::context::{
    RuntimeGraphObservationPermit, RuntimePlacementService, RuntimeServicePorts,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

fn rooted(value: Value) -> runmat_runtime::execution::RootedValue {
    runmat_runtime::execution::RootedValue::new(value).expect("fixture output root")
}

use runmat_runtime::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use runmat_runtime::context::{RuntimeAccelerationService, RuntimeServiceFuture};

struct Adapter(&'static dyn AccelProvider, u16);
impl RuntimeAccelerationService for Adapter {
    fn prepare_admitted(
        &self,
        context: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        let provider = self.0;
        let expected_form = self.1;
        Box::pin(async move {
            assert_eq!(
                request.form().id,
                runmat_builtins::BuiltinPreparationFormId(expected_form)
            );
            let auto = NativeAutoOffload::new(provider, ThresholdConfig::default());
            let prepared = super::admitted::prepare_async(&auto, &context, request)
                .await
                .map_err(|error| *error)?;
            let (operands, guard) = prepared.into_execution();
            PreparedBuiltinOperands::new(operands, Some(Box::new(guard)))
        })
    }
}

struct UploadRejectingProvider {
    quote: bool,
}
impl AccelProvider for UploadRejectingProvider {
    fn upload(&self, _: &HostTensorView<'_>) -> Result<runmat_accelerate_api::GpuTensorHandle> {
        Err(anyhow!("test upload failure"))
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { Err(anyhow!("unused download")) })
    }
    fn device_info(&self) -> String {
        "builtin-observation-provider".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        known_fixture_resources(query)
    }
    fn estimate_cost(
        &self,
        _: &runmat_accelerate_api::ProviderCostQuery,
    ) -> Option<runmat_accelerate_api::ProviderCostEstimate> {
        self.quote.then(fixture_quote)
    }
}
static PROVIDER: UploadRejectingProvider = UploadRejectingProvider { quote: true };
static UNQUOTED_PROVIDER: UploadRejectingProvider = UploadRejectingProvider { quote: false };

// A fixed quote lets the existing forced graph/lifetime fixtures select either
// route. It is not measured provider performance or profitability evidence.
fn fixture_quote() -> runmat_accelerate_api::ProviderCostEstimate {
    runmat_accelerate_api::ProviderCostEstimate::static_prior(
        ExecutionCostComponents {
            execution_ns: 1,
            ..Default::default()
        },
        0,
    )
}

struct FeasibilityRejectingProvider;
impl AccelProvider for FeasibilityRejectingProvider {
    fn upload(&self, _: &HostTensorView<'_>) -> Result<runmat_accelerate_api::GpuTensorHandle> {
        panic!("rejected candidate must not upload")
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { panic!("rejected candidate must not download") })
    }
    fn device_info(&self) -> String {
        "feasibility-rejecting-provider".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn query_feasibility(
        &self,
        _: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        runmat_accelerate_api::ProviderFeasibility::rejected(
            runmat_accelerate_api::ProviderRejectionCode::UnsupportedOperation,
            "test.unsupported",
        )
    }
}
static REJECTING_PROVIDER: FeasibilityRejectingProvider = FeasibilityRejectingProvider;

// These lifecycle fixtures model dense F64 output buffers with the admitted
// element count and output arity. They do not model provider kernel execution.
fn known_fixture_resources(
    query: &runmat_accelerate_api::ProviderFeasibilityQuery,
) -> runmat_accelerate_api::ProviderFeasibility {
    let outputs = u64::from(query.workload.requested_outputs.unwrap_or(1));
    let bytes = query
        .workload
        .elements
        .expect("fixture element count")
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_mul(outputs))
        .expect("fixture physical output size");
    runmat_accelerate_api::ProviderFeasibility::supported(
        runmat_accelerate_api::ProviderResourceEstimate {
            output_bytes: Some(bytes),
            ..Default::default()
        },
    )
}

#[derive(Default)]
struct Probe {
    session: crate::placement::PlacementSession,
    provider: Cell<bool>,
    requests: RefCell<Vec<PlacementPlanRequest>>,
    feedback: RefCell<Vec<PlacementFeedback>>,
}

#[derive(Debug)]
struct LifecycleFixturePermit {
    signature: Digest,
    candidate: String,
    kind: runmat_execution::ExecutionCandidateKind,
}

impl RuntimeGraphObservationPermit for LifecycleFixturePermit {
    fn authorizes(
        &self,
        signature: &PlacementSignature,
        candidate: &str,
        kind: runmat_execution::ExecutionCandidateKind,
    ) -> bool {
        self.signature == signature.cache_key() && self.candidate == candidate && self.kind == kind
    }
}
impl RuntimePlacementService for Probe {
    fn whole_work_runtime_binding_session_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        binding: runmat_builtins::BuiltinBindingIdentity,
        registration: Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_runtime_binding_session_revision(
            &self.session,
            program,
            binding,
            registration,
        )
    }

    fn plan_with_observation_permit(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<runmat_runtime::context::RuntimeGraphPlacementSelection, RuntimeError> {
        let decision = self.plan(request)?;
        let selected = &decision.selections[0];
        let observation_permit = Some(std::sync::Arc::new(LifecycleFixturePermit {
            signature: decision.signature.cache_key(),
            candidate: selected.candidate.clone(),
            kind: selected.kind,
        })
            as std::sync::Arc<dyn RuntimeGraphObservationPermit>);
        Ok(runmat_runtime::context::RuntimeGraphPlacementSelection {
            decision,
            observation_permit,
        })
    }

    fn plan(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, RuntimeError> {
        self.requests.borrow_mut().push(request.clone());
        let node = &request.graph.nodes[0];
        let candidate = &node
            .candidates
            .iter()
            .find(|candidate| candidate.descriptor.kind.is_provider() == self.provider.get())
            .ok_or_else(|| {
                runmat_runtime::build_runtime_error(
                    "fixture requested a physically unavailable route",
                )
                .with_identifier("RunMat:Placement:NoLegalCandidate")
                .build()
            })?
            .descriptor;
        Ok(runmat_execution::PlacementDecision {
            signature: request.signature.clone(),
            selections: vec![runmat_execution::SelectedExecutionCandidate {
                node: node.identity.clone(),
                candidate: candidate.identity.clone(),
                kind: candidate.kind,
            }],
            predicted_total_ns: 1,
            from_cache: false,
            used_local_fallback: false,
            explored_states: 2,
            pruned_states: 0,
        })
    }
    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        self.feedback.borrow_mut().push(feedback);
        Ok(())
    }
    fn invalidate(&self, _: runmat_execution::PlacementInvalidation) {}
}
fn runtime(probe: &Rc<Probe>) -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(probe.clone()))
}
fn observed(probe: &Rc<Probe>, provider: bool, args: Vec<Value>) -> PreparedBuiltinCall {
    let digest = || Digest::sha256(b"builtin-lifecycle-test");
    let candidate: String = if provider { "provider" } else { "cpu" }.into();
    let kind = if provider {
        runmat_execution::ExecutionCandidateKind::ProviderOperation
    } else {
        runmat_execution::ExecutionCandidateKind::SharedRuntime
    };
    let signature = PlacementSignature {
        operation: "test".into(),
        region: None,
        runtime_facts: digest(),
        provider_provenance: None,
        graph_output_authority: None,
        revision: PlacementRevision {
            program: None,
            catalog: digest(),
            compiler: digest(),
            authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(digest()),
            policy: digest(),
        },
    };
    let authority = crate::placement::LocalPlacementFeedbackAuthority::Graph(std::sync::Arc::new(
        LifecycleFixturePermit {
            signature: signature.cache_key(),
            candidate: candidate.clone(),
            kind,
        },
    ));
    let feedback = LocalPlacementFeedback {
        runtime: runtime(probe),
        candidate,
        kind,
        signature,
        authority,
    };
    let observation = BuiltinCallObservation::new(
        Some(Box::new(feedback)),
        begin_trace("test", None),
        provider,
        &args,
    );
    PreparedBuiltinCall {
        compound: None,
        args,
        observation: Some(observation),
        uploads: None,
        request_input_root: None,
        readbacks: None,
        host_numerical_output: None,
    }
}

#[test]
fn cpu_call_is_observed_only_once_after_its_result() {
    let probe = Rc::new(Probe::default());
    let prepared = observed(&probe, false, vec![Value::Num(1.)]);
    let observation = prepared.into_execution().1;
    assert!(probe.feedback.borrow().is_empty());
    observation.finish(Ok(&Value::Num(2.))).unwrap();
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(probe.feedback.borrow()[0].succeeded);
}

#[test]
fn completed_error_reports_failure_but_abandoned_futures_do_not() {
    let probe = Rc::new(Probe::default());
    let prepared = observed(&probe, false, vec![]);
    prepared
        .into_execution()
        .1
        .finish(Err(&RuntimeError::new("test failure")))
        .unwrap();
    drop(observed(&probe, false, vec![]));
    let prepared = observed(&probe, false, vec![]);
    let observation = prepared.into_execution().1;
    let future = async move {
        std::future::pending::<()>().await;
        observation.finish(Ok(&Value::Num(1.))).unwrap();
    };
    drop(future);
    // Retaining prepared arguments must not retain a cancelled call's ticket.
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(probe
        .feedback
        .borrow()
        .iter()
        .all(|sample| !sample.succeeded));
}

#[test]
fn provider_return_is_not_a_completed_device_cost_sample() {
    let probe = Rc::new(Probe::default());
    let prepared = observed(&probe, true, vec![]);
    prepared
        .into_execution()
        .1
        .finish(Ok(&Value::Num(1.)))
        .unwrap();
    assert!(
        probe.feedback.borrow().is_empty(),
        "host retry or provider submission cannot establish device timing"
    );
    let prepared = observed(&probe, true, vec![]);
    prepared
        .into_execution()
        .1
        .finish(Err(&RuntimeError::new("provider failed")))
        .unwrap();
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(!probe.feedback.borrow()[0].succeeded);
}

#[test]
fn resident_input_or_output_prevents_unproven_cpu_sample() {
    let probe = Rc::new(Probe::default());
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let device = Value::GpuTensor(allocation.handle);
    for (input, output) in [(device.clone(), Value::Num(1.)), (Value::Num(1.), device)] {
        let prepared = observed(&probe, false, vec![input]);
        prepared.into_execution().1.finish(Ok(&output)).unwrap();
    }
    assert!(probe.feedback.borrow().is_empty());
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn preparation_fallback_is_failed_provider_attempt_not_cpu_success() {
    let probe = Rc::new(Probe::default());
    let mut prepared = observed(&probe, true, vec![Value::Num(1.)]);
    prepared
        .observation
        .as_mut()
        .unwrap()
        .preparation_fallback();
    prepared
        .into_execution()
        .1
        .finish(Ok(&Value::Num(2.)))
        .unwrap();
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(!probe.feedback.borrow()[0].succeeded);
}

#[test]
fn cancellation_flag_prevents_a_success_sample() {
    let probe = Rc::new(Probe::default());
    let mut prepared = observed(&probe, false, vec![]);
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let ticket = prepared
        .observation
        .as_mut()
        .unwrap()
        .feedback
        .as_mut()
        .unwrap();
    ticket.runtime = ticket
        .runtime
        .clone()
        .with_cancellation_token(cancellation.token());
    cancellation.cancel(runmat_execution::CancellationReason::User);
    prepared
        .into_execution()
        .1
        .finish(Ok(&Value::Num(1.)))
        .unwrap();
    assert!(probe.feedback.borrow().is_empty());
}

#[test]
fn admitted_operation_passes_session_and_exact_identity_to_existing_planner() {
    let probe = Rc::new(Probe::default());
    let runtime = runtime(&probe);
    let auto = NativeAutoOffload::new(&PROVIDER, ThresholdConfig::default());
    let input = Value::Tensor(Tensor::new(vec![1.; 4], vec![1, 4]).unwrap());
    // Canonical mean owns domain admission; this witness exercises the
    // operation-level planner, not the retired legacy name-policy route.
    let prepared = auto
        .prepare_operation(
            "mean",
            &[input],
            Some(&runtime),
            super::preparation::ArgumentOperation::Reduction,
        )
        .unwrap();
    assert_eq!(probe.requests.borrow().len(), 1);
    assert_eq!(probe.requests.borrow()[0].signature.operation, "mean");
    assert!(probe.feedback.borrow().is_empty());
    assert!(matches!(prepared.args()[0], Value::Tensor(_)));
    prepared
        .into_execution()
        .1
        .finish(Ok(&Value::Num(1.)))
        .unwrap();
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(probe.feedback.borrow()[0].succeeded);
}

#[test]
fn sum_legacy_preparation_does_not_upload_before_domain_option_admission() {
    let probe = Rc::new(Probe::default());
    probe.provider.set(true);
    let runtime = runtime(&probe);
    let auto = NativeAutoOffload::new(&PROVIDER, ThresholdConfig::default());
    let input = Value::Tensor(Tensor::new(vec![1.; 4], vec![1, 4]).unwrap());
    let controls = Value::Num(0.0);
    let prepared = auto
        .prepare_call("sum", &[input, controls], Some(&runtime))
        .unwrap();
    assert_eq!(prepared.args().len(), 2);
    assert!(matches!(prepared.args()[0], Value::Tensor(_)));
    assert!(probe.requests.borrow().is_empty());
    assert!(probe.feedback.borrow().is_empty());
}

#[test]
fn upload_failure_closes_selected_provider_attempt_before_call() {
    let probe = Rc::new(Probe::default());
    probe.provider.set(true);
    let runtime = runtime(&probe);
    // Admission comes from the genuine numeric owner; only its upload is
    // injected to fail, before a numeric operation can execute.
    let provider = admitted_owners::owner_with_upload_failure(true);
    let auto = NativeAutoOffload::new(provider, ThresholdConfig::default());
    let input = Value::Tensor(Tensor::new(vec![1.; 4], vec![1, 4]).unwrap());
    assert!(auto
        .prepare_operation(
            "mean",
            &[input],
            Some(&runtime),
            super::preparation::ArgumentOperation::Reduction,
        )
        .is_err());
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(!probe.feedback.borrow()[0].succeeded);
}
