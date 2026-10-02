use super::*;
use crate::context::{RuntimePlacementService, RuntimeServicePorts, RuntimeWholeWorkProbeLease};
use crate::RuntimeError;
use runmat_accelerate_api::observation::ProviderOperationReceipt;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{
    CancellationReason, Digest, ExecutionCandidateKind, PlacementDecision, PlacementInvalidation,
    PlacementPlanRequest, PlacementRevision, WholeWorkDecision, WholeWorkDecisionKind,
    WholeWorkPlacementRequest,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Default)]
pub(crate) struct Witness {
    selections: Cell<usize>,
    feedback: RefCell<Vec<PlacementFeedback>>,
    pending: RefCell<Vec<Box<dyn RuntimePlacementFeedbackSource>>>,
}

impl Witness {
    pub(crate) fn context(self: &Rc<Self>) -> RuntimeContext {
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_placement(self.clone()))
    }

    pub(crate) fn selections(&self) -> usize {
        self.selections.get()
    }

    pub(crate) fn feedback(&self) -> Vec<PlacementFeedback> {
        self.feedback.borrow().clone()
    }

    pub(crate) fn drain(&self) {
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
        let digest = Digest::sha256(b"host-presentation-witness");
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
        request.validate().expect("admitted host presentation");
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
        unreachable!("receipt owner never schedules")
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

struct Probe(Arc<AtomicUsize>);
impl RuntimeWholeWorkProbeLease for Probe {}
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn receipt_owner(
    witness: Option<&Rc<Witness>>,
    cancellation: &CancellationSource,
    probe_count: Arc<AtomicUsize>,
    receipt: ProviderOperationReceipt,
) -> HostPresentationReceipt {
    let mut runtime =
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_cancellation_token(cancellation.token());
    if let Some(witness) = witness {
        runtime = runtime
            .with_service_ports(RuntimeServicePorts::default().with_placement(witness.clone()));
    }
    probe_count.fetch_add(1, Ordering::SeqCst);
    let digest = Digest::sha256(b"host-presentation-witness");
    let selected = RuntimeWholeWorkSelection {
        decision: WholeWorkDecision {
            signature: PlacementSignature {
                region: None,
                operation: "imhist".into(),
                runtime_facts: digest,
                revision: PlacementRevision {
                    program: None,
                    catalog: digest,
                    compiler: digest,
                    provider: digest,
                    policy: digest,
                },
            },
            selected: SelectedExecutionCandidate {
                node: "imhist".into(),
                candidate: "host-upload".into(),
                kind: ExecutionCandidateKind::SharedRuntime,
            },
            kind: WholeWorkDecisionKind::Exploratory,
            predicted_total_ns: None,
        },
        probe: Some(Box::new(Probe(probe_count))),
    };
    HostPresentationReceipt::new(runtime, selected, receipt)
}

#[test]
fn publication_waits_for_actual_submission_completion_and_holds_probe() {
    let witness = Rc::new(Witness::default());
    let cancellation = CancellationSource::new();
    let probes = Arc::new(AtomicUsize::new(0));
    let receipt = ProviderOperationReceipt::new();
    let owner = receipt_owner(Some(&witness), &cancellation, probes.clone(), receipt);
    let submission = owner.recorder().submission().unwrap();
    owner.finish(Ok(true));
    assert!(witness.feedback.borrow().is_empty());
    assert_eq!(witness.pending.borrow().len(), 1);
    assert_eq!(probes.load(Ordering::SeqCst), 1);
    witness.drain();
    assert!(witness.feedback.borrow().is_empty());
    submission.complete();
    witness.drain();
    assert_eq!(witness.feedback.borrow().len(), 1);
    assert_eq!(probes.load(Ordering::SeqCst), 0);
}

#[test]
fn failures_cancellation_and_abandonment_do_not_fabricate_samples() {
    for (failure, expected_samples) in [
        (Some(ProviderObservationFailure::ProviderFailed), 1),
        (Some(ProviderObservationFailure::SharedRuntimeFallback), 0),
        (Some(ProviderObservationFailure::Cancelled), 0),
    ] {
        let witness = Rc::new(Witness::default());
        let cancellation = CancellationSource::new();
        let probes = Arc::new(AtomicUsize::new(0));
        let owner = receipt_owner(
            Some(&witness),
            &cancellation,
            probes.clone(),
            ProviderOperationReceipt::new(),
        );
        owner.finish(Err(failure.unwrap()));
        witness.drain();
        assert_eq!(witness.feedback.borrow().len(), expected_samples);
        assert_eq!(probes.load(Ordering::SeqCst), 0);
    }

    let witness = Rc::new(Witness::default());
    let cancellation = CancellationSource::new();
    let probes = Arc::new(AtomicUsize::new(0));
    let owner = receipt_owner(
        Some(&witness),
        &cancellation,
        probes.clone(),
        ProviderOperationReceipt::new(),
    );
    drop(owner);
    assert_eq!(probes.load(Ordering::SeqCst), 0);
    assert!(witness.feedback.borrow().is_empty());

    let owner = receipt_owner(
        Some(&witness),
        &cancellation,
        probes.clone(),
        ProviderOperationReceipt::new(),
    );
    cancellation.cancel(CancellationReason::User);
    owner.finish(Ok(true));
    witness.drain();
    assert!(witness.feedback.borrow().is_empty());
    assert_eq!(probes.load(Ordering::SeqCst), 0);
}

#[test]
fn absent_placement_service_retires_receipt_without_feedback() {
    let cancellation = CancellationSource::new();
    let probes = Arc::new(AtomicUsize::new(0));
    let owner = receipt_owner(
        None,
        &cancellation,
        probes.clone(),
        ProviderOperationReceipt::new(),
    );
    owner.finish(Ok(true));
    assert_eq!(probes.load(Ordering::SeqCst), 0);
}
