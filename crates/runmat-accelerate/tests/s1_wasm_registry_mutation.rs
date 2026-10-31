//! Browser-only, isolated module: submitting a duplicate canonical row mutates
//! the real WASM registry without contaminating other test binaries.
#![cfg(target_arch = "wasm32")]

use futures::executor::block_on;
use runmat_accelerate::placement::PlacementSession;
use runmat_builtins::BuiltinBindingIdentity;
use runmat_execution::{
    CandidateCostUnavailable, CandidateObservationEvidence, Digest, PlacementDecision,
    PlacementFeedback, PlacementInvalidation, PlacementPlanRequest, PlacementRevision,
    WholeWorkDecision, WholeWorkObservationOutcome, WholeWorkPlacementRequest,
};
use runmat_runtime::builtin::{
    registration::registered_runtime_builtin_bindings_by_name, wasm_registry,
};
use runmat_runtime::context::{
    RuntimeContext, RuntimePlacementService, RuntimeServicePorts, RuntimeWholeWorkSelection,
};
use runmat_runtime::execution::RuntimeExecutionService;
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use std::cell::RefCell;
use std::rc::Rc;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

struct RecordingSession {
    session: PlacementSession,
    requests: RefCell<Vec<WholeWorkPlacementRequest>>,
    decisions: RefCell<Vec<WholeWorkDecision>>,
    on_select: RefCell<Option<Box<dyn FnOnce()>>>,
}

impl RuntimePlacementService for RecordingSession {
    fn whole_work_runtime_binding_session_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        binding: BuiltinBindingIdentity,
        registration: Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_runtime_binding_session_revision(
            &self.session,
            program,
            binding,
            registration,
        )
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        self.requests.borrow_mut().push(request.clone());
        let selected = RuntimePlacementService::select_whole_work(&self.session, request)?;
        self.decisions.borrow_mut().push(selected.decision.clone());
        if let Some(mutate) = self.on_select.borrow_mut().take() {
            mutate();
        }
        Ok(selected)
    }

    fn observe_selected_whole_work(
        &self,
        selected: &RuntimeWholeWorkSelection,
        elapsed_ns: u64,
        outcome: WholeWorkObservationOutcome,
    ) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe_selected_whole_work(
            &self.session,
            selected,
            elapsed_ns,
            outcome,
        )
    }

    fn plan(&self, request: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
        RuntimePlacementService::plan(&self.session, request)
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe(&self.session, feedback)
    }

    fn invalidate(&self, invalidation: PlacementInvalidation) {
        RuntimePlacementService::invalidate(&self.session, invalidation)
    }
}

#[wasm_bindgen_test::wasm_bindgen_test]
fn global_submit_revokes_in_flight_strcmp_and_retires_old_session_feedback() {
    runmat_runtime::builtins::wasm_registry::register_all();
    let original = registered_runtime_builtin_bindings_by_name("strcmp");
    assert_eq!(
        original.len(),
        1,
        "isolated browser registry starts canonical"
    );
    let binding = original[0].binding();
    let old_registration = original[0].registration();
    assert!(old_registration.is_current());

    let placement = Rc::new(RecordingSession {
        session: PlacementSession::default(),
        requests: RefCell::new(Vec::new()),
        decisions: RefCell::new(Vec::new()),
        on_select: RefCell::new(None),
    });
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_placement(placement.clone()));
    let arguments = [Value::String("same".into()), Value::String("same".into())];
    let call = || {
        block_on(
            runmat_runtime::call_builtin_sequence_async_with_outputs_in_context(
                &runtime, "strcmp", &arguments, 1,
            ),
        )
        .expect("canonical scalar strcmp completes")
    };

    assert_eq!(call().as_slice(), &[Value::Bool(true)]);
    let old = placement.decisions.borrow()[0].clone();
    let old_request = placement.requests.borrow()[0].clone();
    assert_eq!(
        placement.session.observed_candidate_duration(
            &old.signature,
            &old.selected.candidate,
            old.selected.kind,
        ),
        CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::InsufficientObservations {
                successful: 1,
                failed: 0,
            }
        )
    );

    // The real selected call has already acquired its session permit and
    // exploratory probe when the real process-global registry is mutated.
    *placement.on_select.borrow_mut() = Some(Box::new(move || wasm_registry::submit(binding)));
    assert_eq!(call().as_slice(), &[Value::Bool(true)]);
    assert!(!old_registration.is_current());
    let current = registered_runtime_builtin_bindings_by_name("strcmp");
    assert_eq!(current.len(), 2, "duplicate canonical submission is real");
    assert!(current[0].registration().is_current());
    assert!(current[0].registration().certifies(binding));
    assert_ne!(
        old_registration.session_generation(),
        current[0].registration().session_generation()
    );
    assert_eq!(placement.decisions.borrow().len(), 2);
    assert_eq!(
        placement.session.observed_candidate_duration(
            &old.signature,
            &old.selected.candidate,
            old.selected.kind,
        ),
        CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::InsufficientObservations {
                successful: 1,
                failed: 0,
            }
        ),
        "stale in-flight call must not add feedback"
    );

    // A new probe on the old request proves the rejected selection released
    // its probe. Hold its permit while the fresh generation is admitted.
    let pending = RuntimePlacementService::select_whole_work(&placement.session, old_request)
        .expect("old request may still select until generation replacement");
    assert!(pending.probe.is_some(), "stale selected probe was released");
    assert!(pending.observation_is_current());

    assert_eq!(call().as_slice(), &[Value::Bool(true)]);
    let fresh = placement.decisions.borrow()[2].clone();
    assert_ne!(fresh.signature.revision, old.signature.revision);
    assert!(!pending.observation_is_current());
    assert_eq!(
        placement.session.observed_candidate_duration(
            &old.signature,
            &old.selected.candidate,
            old.selected.kind,
        ),
        CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
    );
    assert_eq!(
        placement.session.observed_candidate_duration(
            &fresh.signature,
            &fresh.selected.candidate,
            fresh.selected.kind,
        ),
        CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::InsufficientObservations {
                successful: 1,
                failed: 0,
            }
        )
    );

    // Exercise the real browser catalog cache after initialization. This test
    // owns an isolated WASM module, so a late declaration cannot affect another
    // test's registry. Session reuse must not hide changes to executable rows.
    let before = runmat_builtins::builtin_catalog_fingerprint();
    let declaration = (*runmat_builtins::builtin_functions()
        .into_iter()
        .find(|function| function.name == "strcmp")
        .expect("registered strcmp declaration"))
    .clone();
    runmat_builtins::wasm_registry::submit_builtin_function(declaration);
    let after = runmat_builtins::builtin_catalog_fingerprint();
    assert_ne!(before, after, "late declaration must invalidate the cache");
    assert_eq!(after, runmat_builtins::builtin_catalog_fingerprint());
}
