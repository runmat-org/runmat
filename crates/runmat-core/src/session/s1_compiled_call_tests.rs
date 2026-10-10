//! Compiler/VM/native continuity for session-owned direct builtin feedback.
use super::*;
use runmat_execution::{
    CandidateObservationEvidence, PlacementDecision, PlacementFeedback, PlacementInvalidation,
    PlacementPlanRequest, PlacementRevision, WholeWorkDecision, WholeWorkObservationOutcome,
    WholeWorkPlacementRequest,
};
use runmat_runtime::context::{RuntimePlacementService, RuntimeWholeWorkSelection};
use std::cell::RefCell;

struct RecordingPlacement {
    actual: Rc<runmat_accelerate::placement::PlacementSession>,
    decisions: RefCell<Vec<WholeWorkDecision>>,
}

impl RuntimePlacementService for RecordingPlacement {
    fn whole_work_runtime_binding_session_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        binding: runmat_builtins::BuiltinBindingIdentity,
        registration: runmat_execution::Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_runtime_binding_session_revision(
            self.actual.as_ref(),
            program,
            binding,
            registration,
        )
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        let selected = RuntimePlacementService::select_whole_work(self.actual.as_ref(), request)?;
        self.decisions.borrow_mut().push(selected.decision.clone());
        Ok(selected)
    }

    fn observe_selected_whole_work(
        &self,
        selected: &RuntimeWholeWorkSelection,
        elapsed_ns: u64,
        outcome: WholeWorkObservationOutcome,
    ) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe_selected_whole_work(
            self.actual.as_ref(),
            selected,
            elapsed_ns,
            outcome,
        )
    }

    fn plan(&self, request: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
        RuntimePlacementService::plan(self.actual.as_ref(), request)
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe(self.actual.as_ref(), feedback)
    }

    fn invalidate(&self, invalidation: PlacementInvalidation) {
        RuntimePlacementService::invalidate(self.actual.as_ref(), invalidation)
    }
}

fn compiled_strcmp_reuses_session_feedback(force_native: bool) {
    let mut session = RunMatSession::with_options(false, false).expect("Core session");
    let source = crate::ExecutableSource::new(
        "s1-compiled-strcmp@1",
        "s1Compare.m",
        "function y = s1Compare(a,b)\ny = strcmp(a,b);\nend\n",
    );
    let unit = futures::executor::block_on(session.compile_executable_unit(source, None))
        .expect("compile dynamic-argument function");
    let probe = Rc::new(RecordingPlacement {
        actual: session.placement_session.clone(),
        decisions: RefCell::new(Vec::new()),
    });
    let ports = session
        .runtime_context
        .service_ports()
        .clone()
        .with_placement(probe.clone());
    session.runtime_context = session.runtime_context.clone().with_service_ports(ports);
    let invocation = crate::ProcedureInvocation {
        target: crate::ProcedureTarget::Function("s1Compare".into()),
        arguments: vec![Value::String("same".into()), Value::String("same".into())],
        requested_outputs: 1,
    };
    for call in 0..3 {
        let control = if force_native {
            crate::InvocationControl::default().force_generic_native()
        } else {
            crate::InvocationControl::default()
        };
        let output = futures::executor::block_on(session.invoke_executable(
            &unit,
            invocation.clone(),
            &control,
        ))
        .expect("compiled scalar strcmp");
        assert_eq!(output.as_slice(), &[Value::Bool(true)]);
        let decisions = probe.decisions.borrow();
        assert_eq!(
            decisions.len(),
            call + 1,
            "one selected builtin per invocation"
        );
        let first = &decisions[0];
        assert_eq!(decisions[call].signature, first.signature);
        assert_eq!(decisions[call].selected, first.selected);
    }
    let decisions = probe.decisions.borrow();
    let first = &decisions[0];
    assert_eq!(
        first.kind,
        runmat_execution::WholeWorkDecisionKind::Exploratory
    );
    assert!(matches!(
        probe.actual.observed_candidate_duration(
            &first.signature,
            &first.selected.candidate,
            first.selected.kind,
        ),
        CandidateObservationEvidence::Observed(_)
    ));
    assert!(decisions[2].predicted_total_ns.is_some());
    assert!(probe
        .actual
        .profile_snapshot()
        .feedback
        .iter()
        .all(|series| {
            !matches!(
                series.signature.revision.authority,
                runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
            )
        }));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn interpreted_compiled_strcmp_preserves_and_learns_one_session_key() {
    compiled_strcmp_reuses_session_feedback(false);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn forced_generic_native_compiled_strcmp_preserves_and_learns_one_session_key() {
    compiled_strcmp_reuses_session_feedback(true);
}
