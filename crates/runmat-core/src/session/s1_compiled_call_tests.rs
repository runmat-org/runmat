//! Compiler/VM/native continuity for session-owned direct builtin feedback.
use super::*;
use runmat_execution::{
    CandidateObservationEvidence, PlacementDecision, PlacementFeedback, PlacementInvalidation,
    PlacementPlanRequest, PlacementRevision, WholeWorkDecision, WholeWorkObservationOutcome,
    WholeWorkPlacementRequest,
};
use runmat_runtime::context::{RuntimePlacementService, RuntimeWholeWorkSelection};
use std::cell::RefCell;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct ControlledReadyPlacement {
    actual: Rc<runmat_accelerate::placement::PlacementSession>,
    winner: runmat_execution::ExecutionCandidateKind,
    offered: RefCell<Vec<Vec<runmat_execution::ExecutionCandidateKind>>>,
    chosen: RefCell<Vec<runmat_execution::ExecutionCandidateKind>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ControlledReadyPlacement {
    pub(crate) fn chosen_kinds(&self) -> Vec<runmat_execution::ExecutionCandidateKind> {
        self.chosen.borrow().clone()
    }

    fn install(
        session: &mut RunMatSession,
        winner: runmat_execution::ExecutionCandidateKind,
    ) -> Rc<Self> {
        let service = Rc::new(Self {
            actual: session.placement_session.clone(),
            winner,
            offered: RefCell::new(Vec::new()),
            chosen: RefCell::new(Vec::new()),
        });
        let ports = session
            .runtime_context
            .service_ports()
            .clone()
            .with_placement(service.clone());
        session.runtime_context = session.runtime_context.clone().with_service_ports(ports);
        service
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RunMatSession {
    pub(crate) fn use_generic_ready_route_for_testing(&mut self) -> Rc<ControlledReadyPlacement> {
        ControlledReadyPlacement::install(
            self,
            runmat_execution::ExecutionCandidateKind::GenericNativeCpu,
        )
    }

    pub(crate) fn use_specialized_ready_route_for_testing(
        &mut self,
    ) -> Rc<ControlledReadyPlacement> {
        ControlledReadyPlacement::install(
            self,
            runmat_execution::ExecutionCandidateKind::SpecializedNativeCpu,
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RuntimePlacementService for ControlledReadyPlacement {
    fn whole_work_host_service_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        host: runmat_execution::Digest,
    ) -> Option<PlacementRevision> {
        self.actual.whole_work_host_service_revision(program, host)
    }

    fn select_whole_work(
        &self,
        mut request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        let ready_invocation = request.signature.region.is_none()
            && request.candidates.iter().any(|item| {
                item.kind == runmat_execution::ExecutionCandidateKind::GenericNativeCpu
            })
            && request.candidates.iter().all(|item| {
                matches!(
                    item.kind,
                    runmat_execution::ExecutionCandidateKind::SharedRuntime
                        | runmat_execution::ExecutionCandidateKind::GenericNativeCpu
                        | runmat_execution::ExecutionCandidateKind::SpecializedNativeCpu
                )
            });
        if ready_invocation {
            self.offered
                .borrow_mut()
                .push(request.candidates.iter().map(|item| item.kind).collect());
            for item in &mut request.candidates {
                item.cost = runmat_execution::CandidateCostEvidence::Estimated(
                    runmat_execution::ExecutionCostEstimate {
                        components: runmat_execution::ExecutionCostComponents {
                            execution_ns: if item.kind == self.winner {
                                1_000
                            } else {
                                100_000
                            },
                            ..Default::default()
                        },
                        scratch_bytes: None,
                        confidence: runmat_execution::EstimateConfidence::Exact,
                        source: runmat_execution::EstimateSource::Synthetic,
                    },
                );
            }
        }
        let selected = RuntimePlacementService::select_whole_work(self.actual.as_ref(), request)?;
        // A selected specialized invocation can itself run a numerical-region
        // comparison (including OriginalNativeRegion). That is a different
        // decision domain, not another complete-call ready alternative.
        if ready_invocation {
            self.chosen
                .borrow_mut()
                .push(selected.decision.selected.kind);
        }
        Ok(selected)
    }

    // Controlled costs are the oracle. Real-time measurements must not replace
    // them and make the deterministic winner depend on machine noise.
    fn observe_selected_whole_work(
        &self,
        selected: &RuntimeWholeWorkSelection,
        elapsed_ns: u64,
        outcome: WholeWorkObservationOutcome,
    ) -> Result<(), RuntimeError> {
        if selected.decision.signature.region.is_some() {
            return self
                .actual
                .observe_selected_whole_work(selected, elapsed_ns, outcome);
        }
        Ok(())
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

#[cfg(not(target_arch = "wasm32"))]
fn controlled_ready_routes(
    winner: runmat_execution::ExecutionCandidateKind,
    profileless: bool,
    fresh_units: bool,
) {
    use runmat_execution::ExecutionCandidateKind;
    let mut session = RunMatSession::with_options(true, false).expect("Core session");
    session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
        generic_hot_threshold: 1,
        specialized_hot_threshold: 2,
        max_profile_bytes: if profileless { 1 } else { 64 * 1024 },
        deterministic: true,
        ..Default::default()
    });
    let unit = futures::executor::block_on(session.compile_executable_unit(
        crate::ExecutableSource::new(
            "all-ready-routes@1",
            "allReady.m",
            "function y = allReady(x)\ny=x+1;\nend\n",
        ),
        None,
    ))
    .expect("compile all-ready fixture");
    let service = ControlledReadyPlacement::install(&mut session, winner);
    let mut compilation_digest = None;
    for _ in 0..12 {
        // A clone shares the digest OnceCell and cannot model ordinary source
        // requests. Recompile the same immutable source into a fresh seal.
        let fresh_unit = fresh_units.then(|| {
            futures::executor::block_on(session.compile_executable_unit(
                crate::ExecutableSource::new(
                    "all-ready-routes@1",
                    "allReady.m",
                    "function y = allReady(x)\ny=x+1;\nend\n",
                ),
                None,
            ))
            .expect("compile fresh equivalent all-ready fixture")
        });
        let invoked_unit = fresh_unit.as_ref().unwrap_or(&unit);
        if fresh_units {
            assert!(invoked_unit.cached_native_compilation_digest().is_none());
        }
        let output = futures::executor::block_on(session.invoke_executable(
            invoked_unit,
            crate::ProcedureInvocation {
                target: crate::ProcedureTarget::Function("allReady".into()),
                arguments: crate::ProcedureArguments::Source(vec![Value::Num(4.0)]),
                requested_outputs: 1,
            },
            &crate::InvocationControl::default(),
        ))
        .expect("selected ready route executes");
        assert_eq!(output.as_slice(), &[Value::Num(5.0)]);
        let digest = invoked_unit
            .cached_native_compilation_digest()
            .expect("native preparation owns the compilation digest");
        if let Some(expected) = compilation_digest {
            assert_eq!(digest, expected, "fresh units retain exact native identity");
        } else {
            compilation_digest = Some(digest);
        }
    }
    let offered = service.offered.borrow();
    let last = offered.last().expect("ready routes reach placement");
    assert!(last.contains(&ExecutionCandidateKind::SharedRuntime));
    assert!(last.contains(&ExecutionCandidateKind::GenericNativeCpu));
    assert_eq!(
        last.contains(&ExecutionCandidateKind::SpecializedNativeCpu),
        !profileless
    );
    let chosen = service.chosen.borrow();
    assert_eq!(
        chosen.len(),
        offered.len(),
        "one decision per ready comparison"
    );
    assert_eq!(chosen.last(), Some(&winner));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn all_published_ready_routes_compete_and_each_can_win() {
    use runmat_execution::ExecutionCandidateKind;
    for winner in [
        ExecutionCandidateKind::SharedRuntime,
        ExecutionCandidateKind::GenericNativeCpu,
        ExecutionCandidateKind::SpecializedNativeCpu,
    ] {
        for fresh_units in [false, true] {
            controlled_ready_routes(winner, false, fresh_units);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn profileless_ready_generic_still_competes_with_interpreter() {
    for fresh_units in [false, true] {
        controlled_ready_routes(
            runmat_execution::ExecutionCandidateKind::GenericNativeCpu,
            true,
            fresh_units,
        );
    }
}

struct RecordingPlacement {
    actual: Rc<runmat_accelerate::placement::PlacementSession>,
    decisions: RefCell<Vec<WholeWorkDecision>>,
}

impl RuntimePlacementService for RecordingPlacement {
    fn whole_work_host_service_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        host_service: runmat_execution::Digest,
    ) -> Option<PlacementRevision> {
        RuntimePlacementService::whole_work_host_service_revision(
            self.actual.as_ref(),
            program,
            host_service,
        )
    }

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
        arguments: crate::ProcedureArguments::Source(vec![
            Value::String("same".into()),
            Value::String("same".into()),
        ]),
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

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn ready_native_and_interpreter_share_one_complete_call_receipt() {
    let mut session = RunMatSession::with_options(true, false).expect("Core session");
    session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
        generic_hot_threshold: 1,
        specialized_hot_threshold: 100,
        deterministic: true,
        ..Default::default()
    });
    let unit = futures::executor::block_on(session.compile_executable_unit(
        crate::ExecutableSource::new(
            "core-ready-placement@1",
            "readyPlacement.m",
            "function y = readyPlacement(x)\ny = x + 1;\nend\n",
        ),
        None,
    ))
    .expect("compile ready-route fixture");
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
    for _ in 0..8 {
        let output = futures::executor::block_on(session.invoke_executable(
            &unit,
            crate::ProcedureInvocation {
                target: crate::ProcedureTarget::Function("readyPlacement".into()),
                arguments: crate::ProcedureArguments::Source(vec![Value::Num(4.0)]),
                requested_outputs: 1,
            },
            &crate::InvocationControl::default(),
        ))
        .expect("ready interpreter/native route");
        assert_eq!(output.as_slice(), &[Value::Num(5.0)]);
    }
    assert_eq!(session.generic_native_cache_counts(), (1, 1));
    let decisions = probe.decisions.borrow();
    let ready = decisions
        .iter()
        .filter(|decision| decision.signature.operation == "core.ready-invocation")
        .collect::<Vec<_>>();
    assert!(
        !ready.is_empty(),
        "ready code must enter whole-work placement"
    );
    assert_eq!(
        ready[0].kind,
        runmat_execution::WholeWorkDecisionKind::Exploratory,
        "unknown ready-route costs need an actual first observation"
    );
    assert!(ready.iter().all(|decision| matches!(
        decision.selected.kind,
        runmat_execution::ExecutionCandidateKind::SharedRuntime
            | runmat_execution::ExecutionCandidateKind::GenericNativeCpu
            | runmat_execution::ExecutionCandidateKind::SpecializedNativeCpu
    )));
    for kind in [
        runmat_execution::ExecutionCandidateKind::SharedRuntime,
        runmat_execution::ExecutionCandidateKind::GenericNativeCpu,
    ] {
        let decision = ready
            .iter()
            .find(|decision| decision.selected.kind == kind)
            .expect("both actual ready interpreter and native routes were explored");
        assert!(matches!(
            probe.actual.observed_candidate_duration(
                &decision.signature,
                &decision.selected.candidate,
                decision.selected.kind,
            ),
            CandidateObservationEvidence::Observed(_)
        ));
    }
}
