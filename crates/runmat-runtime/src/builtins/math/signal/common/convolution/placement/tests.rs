//! Shared whole-call placement witness used by the public convolution tests.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_execution::{
    Digest, PlacementDecision, PlacementFeedback, PlacementInvalidation, PlacementPlanRequest,
    PlacementRevision, WholeWorkDecision, WholeWorkDecisionKind, WholeWorkPlacementRequest,
};

use crate::context::{
    RuntimeContext, RuntimePlacementFeedbackSource, RuntimePlacementService, RuntimeServicePorts,
    RuntimeWholeWorkSelection,
};
use crate::RuntimeError;

pub(crate) struct PlacementWitnessService {
    prefer_provider: bool,
    selections: Cell<usize>,
    feedback: RefCell<Vec<PlacementFeedback>>,
    deferred: RefCell<Vec<Box<dyn RuntimePlacementFeedbackSource>>>,
    cancellation: CancellationSource,
    cancel_on_select: bool,
    selected_candidate: RefCell<Option<runmat_execution::WholeWorkCandidate>>,
}

impl PlacementWitnessService {
    pub(crate) fn new(prefer_provider: bool) -> Rc<Self> {
        Self::with_cancel_on_select(prefer_provider, false)
    }

    pub(crate) fn with_cancel_on_select(prefer_provider: bool, cancel_on_select: bool) -> Rc<Self> {
        Rc::new(Self {
            prefer_provider,
            selections: Cell::new(0),
            feedback: RefCell::new(Vec::new()),
            deferred: RefCell::new(Vec::new()),
            cancellation: CancellationSource::new(),
            cancel_on_select,
            selected_candidate: RefCell::new(None),
        })
    }

    pub(crate) fn context(self: &Rc<Self>) -> RuntimeContext {
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_cancellation_token(self.cancellation.token())
            .with_service_ports(RuntimeServicePorts::default().with_placement(self.clone()))
    }

    pub(crate) fn selections(&self) -> usize {
        self.selections.get()
    }

    pub(crate) fn feedback(&self) -> Vec<PlacementFeedback> {
        self.feedback.borrow().clone()
    }

    pub(crate) fn pending(&self) -> usize {
        self.deferred.borrow().len()
    }

    pub(crate) fn selected_candidate(&self) -> Option<runmat_execution::WholeWorkCandidate> {
        self.selected_candidate.borrow().clone()
    }
}

impl RuntimePlacementService for PlacementWitnessService {
    fn whole_work_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        provider: Digest,
    ) -> Option<PlacementRevision> {
        Some(PlacementRevision {
            program,
            catalog: Digest::sha256(b"convolution-witness-catalog"),
            compiler: Digest::sha256(b"convolution-witness-compiler"),
            authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(provider),
            policy: Digest::sha256(b"convolution-witness-policy"),
        })
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        request
            .validate()
            .expect("admitted convolution placement request");
        let candidate = request
            .candidates
            .iter()
            .find(|candidate| candidate.kind.is_provider() == self.prefer_provider)
            .expect("requested physical route must be offered");
        *self.selected_candidate.borrow_mut() = Some(candidate.clone());
        self.selections.set(self.selections.get() + 1);
        if self.cancel_on_select {
            self.cancellation.cancel(CancellationReason::User);
        }
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
        panic!("whole-call convolution must not enter graph placement")
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        self.feedback.borrow_mut().push(feedback);
        Ok(())
    }

    fn defer_feedback(
        &self,
        source: Box<dyn RuntimePlacementFeedbackSource>,
    ) -> Result<(), Box<dyn RuntimePlacementFeedbackSource>> {
        self.deferred.borrow_mut().push(source);
        Ok(())
    }

    fn drain_feedback(&self) {
        let mut pending = std::mem::take(&mut *self.deferred.borrow_mut());
        let mut still_pending = Vec::new();
        for mut source in pending.drain(..) {
            if let Some(feedback) = source.take_ready() {
                self.observe_deferred_feedback(feedback).unwrap();
            } else if !source.terminal_without_feedback() {
                still_pending.push(source);
            }
        }
        self.deferred.borrow_mut().extend(still_pending);
    }

    fn invalidate(&self, _: PlacementInvalidation) {}
}
