use super::*;
mod ownership;
mod admitted;
mod integer_conversion;
mod pol2cart;
mod admitted_owners;
use crate::placement::{begin_trace, LocalPlacementFeedback};
use runmat_execution::{
    Digest, PlacementFeedback, PlacementPlanRequest, PlacementRevision, PlacementSignature,
};
use runmat_runtime::context::{RuntimePlacementService, RuntimeServicePorts};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

struct UploadRejectingProvider;
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
    fn free(&self, _: &runmat_accelerate_api::GpuTensorHandle) -> Result<()> {
        Ok(())
    }
    fn device_info(&self) -> String {
        "builtin-observation-provider".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn query_feasibility(
        &self,
        _: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        runmat_accelerate_api::ProviderFeasibility::supported(Default::default())
    }
}
static PROVIDER: UploadRejectingProvider = UploadRejectingProvider;

#[derive(Default)]
struct Probe {
    provider: Cell<bool>,
    requests: RefCell<Vec<PlacementPlanRequest>>,
    feedback: RefCell<Vec<PlacementFeedback>>,
}
impl RuntimePlacementService for Probe {
    fn plan(
        &self,
        request: PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, RuntimeError> {
        self.requests.borrow_mut().push(request.clone());
        let node = &request.graph.nodes[0];
        let candidate = &node.candidates[usize::from(self.provider.get())].descriptor;
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
    let feedback = LocalPlacementFeedback {
        runtime: runtime(probe),
        candidate: if provider { "provider" } else { "cpu" }.into(),
        signature: PlacementSignature {
            operation: "test".into(),
            region: None,
            runtime_facts: digest(),
            revision: PlacementRevision {
                program: None,
                catalog: digest(),
                compiler: digest(),
                provider: digest(),
                policy: digest(),
            },
        },
    };
    let observation = BuiltinCallObservation::new(
        Some(Box::new(feedback)),
        begin_trace("test", None),
        provider,
        &args,
    );
    PreparedBuiltinCall {
        args,
        observation: Some(observation),
        uploads: None,
    }
}

#[test]
fn cpu_call_is_observed_only_once_after_its_result() {
    let probe = Rc::new(Probe::default());
    let prepared = observed(&probe, false, vec![Value::Num(1.)]);
    let observation = prepared.into_execution().1;
    assert!(probe.feedback.borrow().is_empty());
    observation.finish(&Ok(Value::Num(2.))).unwrap();
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(probe.feedback.borrow()[0].succeeded);
}

#[test]
fn errors_abandonment_and_cancelled_call_futures_report_failure() {
    let probe = Rc::new(Probe::default());
    let prepared = observed(&probe, false, vec![]);
    prepared
        .into_execution()
        .1
        .finish(&Err(RuntimeError::new("test failure")))
        .unwrap();
    drop(observed(&probe, false, vec![]));
    let prepared = observed(&probe, false, vec![]);
    let observation = prepared.into_execution().1;
    let future = async move {
        std::future::pending::<()>().await;
        observation.finish(&Ok(Value::Num(1.))).unwrap();
    };
    drop(future);
    // Retaining prepared arguments must not retain a cancelled call's ticket.
    assert_eq!(probe.feedback.borrow().len(), 3);
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
        .finish(&Ok(Value::Num(1.)))
        .unwrap();
    assert!(
        probe.feedback.borrow().is_empty(),
        "host retry or provider submission cannot establish device timing"
    );
    let prepared = observed(&probe, true, vec![]);
    prepared
        .into_execution()
        .1
        .finish(&Err(RuntimeError::new("provider failed")))
        .unwrap();
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(!probe.feedback.borrow()[0].succeeded);
}

#[test]
fn resident_input_or_output_prevents_unproven_cpu_sample() {
    let probe = Rc::new(Probe::default());
    let device = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle::new(
        vec![1, 1],
        81,
        42,
    ));
    for (input, output) in [(device.clone(), Value::Num(1.)), (Value::Num(1.), device)] {
        let prepared = observed(&probe, false, vec![input]);
        prepared.into_execution().1.finish(&Ok(output)).unwrap();
    }
    assert!(probe.feedback.borrow().is_empty());
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
        .finish(&Ok(Value::Num(2.)))
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
        .finish(&Ok(Value::Num(1.)))
        .unwrap();
    assert!(!probe.feedback.borrow()[0].succeeded);
}

#[test]
fn preparation_passes_session_and_exact_operation_to_existing_planner() {
    let probe = Rc::new(Probe::default());
    let runtime = runtime(&probe);
    let auto = NativeAutoOffload::new(&PROVIDER, ThresholdConfig::default());
    let input = Value::Tensor(Tensor::new(vec![1.; 4], vec![1, 4]).unwrap());
    let prepared = auto.prepare_call("mean", &[input], Some(&runtime)).unwrap();
    assert_eq!(probe.requests.borrow().len(), 1);
    assert_eq!(probe.requests.borrow()[0].signature.operation, "mean");
    assert!(probe.feedback.borrow().is_empty());
    assert!(matches!(prepared.args()[0], Value::Tensor(_)));
    prepared
        .into_execution()
        .1
        .finish(&Ok(Value::Num(1.)))
        .unwrap();
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(probe.feedback.borrow()[0].succeeded);
}

#[test]
fn upload_failure_closes_selected_provider_attempt_before_call() {
    let probe = Rc::new(Probe::default());
    probe.provider.set(true);
    let runtime = runtime(&probe);
    let auto = NativeAutoOffload::new(&PROVIDER, ThresholdConfig::default());
    let input = Value::Tensor(Tensor::new(vec![1.; 4], vec![1, 4]).unwrap());
    assert!(auto.prepare_call("mean", &[input], Some(&runtime)).is_err());
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(!probe.feedback.borrow()[0].succeeded);
}
