//! The real session policy, reached through the runtime placement port.
use super::*;
use runmat_execution::{
    CandidateCostUnavailable, CandidateObservationEvidence, PlacementDecision,
    PlacementExecutionAuthorityRevision, PlacementFeedback, PlacementInvalidation,
    PlacementPlanRequest, PlacementRevision, ProgramEnvironment, ProgramRevision,
    WholeWorkDecision, WholeWorkObservationOutcome,
};
use runmat_runtime::context::{
    RuntimeContext, RuntimePlacementService, RuntimeServicePorts, RuntimeWholeWorkSelection,
    VmFusionSemanticIdentity, VmFusionSessionRegistration,
};
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use std::cell::RefCell;
use std::rc::Rc;

struct RecordingSession {
    session: PlacementSession,
    requests: RefCell<Vec<WholeWorkPlacementRequest>>,
    decisions: RefCell<Vec<WholeWorkDecision>>,
}

fn program_revision(label: &[u8]) -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(label),
        Digest::sha256(b"vm generation test source"),
        ProgramEnvironment::new(
            2,
            2,
            Digest::sha256(b"vm generation test runtime"),
            Digest::sha256(b"vm generation test builtins"),
            "runmat",
        )
        .unwrap(),
    )
    .unwrap()
}

impl RuntimePlacementService for RecordingSession {
    fn whole_work_vm_fusion_region_session_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        region: runmat_runtime::context::VmFusionRegionRegistration,
        liveness: std::sync::Weak<()>,
    ) -> Option<VmFusionSessionRegistration> {
        RuntimePlacementService::whole_work_vm_fusion_region_session_revision(
            &self.session,
            program,
            region,
            liveness,
        )
    }

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

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        self.requests.borrow_mut().push(request.clone());
        let selected = RuntimePlacementService::select_whole_work(&self.session, request)?;
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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn scalar_direct_call_bootstraps_and_reuses_real_session_feedback() {
    let recorder = Rc::new(RecordingSession {
        session: PlacementSession::default(),
        requests: RefCell::new(Vec::new()),
        decisions: RefCell::new(Vec::new()),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(recorder.clone()));
    let arguments = [Value::String("same".into()), Value::String("same".into())];

    let call = || {
        futures::executor::block_on(
            runmat_runtime::call_builtin_sequence_async_with_outputs_in_context(
                &runtime, "strcmp", &arguments, 1,
            ),
        )
        .expect("host scalar strcmp")
    };
    assert_eq!(call().as_slice(), &[Value::Bool(true)]);
    let first = recorder.decisions.borrow()[0].clone();
    assert_eq!(first.kind, WholeWorkDecisionKind::Exploratory);
    assert_eq!(first.selected.kind, ExecutionCandidateKind::SharedRuntime);
    assert_eq!(
        recorder.session.observed_candidate_duration(
            &first.signature,
            &first.selected.candidate,
            first.selected.kind,
        ),
        CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::InsufficientObservations {
                successful: 1,
                failed: 0,
            }
        )
    );

    assert_eq!(call().as_slice(), &[Value::Bool(true)]);
    assert!(matches!(
        recorder.session.observed_candidate_duration(
            &first.signature,
            &first.selected.candidate,
            first.selected.kind,
        ),
        CandidateObservationEvidence::Observed(_)
    ));

    assert_eq!(call().as_slice(), &[Value::Bool(true)]);
    let decisions = recorder.decisions.borrow();
    assert_eq!(decisions.len(), 3);
    assert_eq!(decisions[2].signature, first.signature);
    assert_eq!(decisions[2].selected, first.selected);
    assert_ne!(decisions[2].kind, WholeWorkDecisionKind::Exploratory);
    assert!(decisions[2].predicted_total_ns.is_some());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn typed_registration_replacement_retires_old_feedback_and_pending_permit_only() {
    let recorder = Rc::new(RecordingSession {
        session: PlacementSession::default(),
        requests: RefCell::new(Vec::new()),
        decisions: RefCell::new(Vec::new()),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(recorder.clone()));
    futures::executor::block_on(
        runmat_runtime::call_builtin_sequence_async_with_outputs_in_context(
            &runtime,
            "strcmp",
            &[Value::String("a".into()), Value::String("a".into())],
            1,
        ),
    )
    .expect("original registration call");
    let old = recorder.decisions.borrow()[0].clone();
    let old_request = recorder.requests.borrow()[0].clone();
    let pending = RuntimePlacementService::select_whole_work(&recorder.session, old_request)
        .expect("second same-session selection retains permit");
    assert!(pending.observation_is_current());
    let binding = runmat_runtime::builtin::runtime_builtin_bindings_by_name("strcmp")[0];
    let other = runmat_runtime::builtin::runtime_builtin_bindings_by_name("strcmpi")[0];
    let unaffected_generation = Digest::sha256(b"unrelated exact registration");
    let unrelated = RuntimePlacementService::whole_work_runtime_binding_session_revision(
        &recorder.session,
        None,
        other.identity,
        unaffected_generation,
    )
    .expect("unrelated local authority");
    let mut other_signature = old.signature.clone();
    other_signature.revision = unrelated;
    let unrelated_feedback = PlacementFeedback {
        signature: other_signature.clone(),
        candidate: old.selected.candidate.clone(),
        kind: old.selected.kind,
        total_elapsed_ns: 13,
        succeeded: true,
    };
    RuntimePlacementService::observe(&recorder.session, unrelated_feedback)
        .expect("unrelated sample");

    let replacement = RuntimePlacementService::whole_work_runtime_binding_session_revision(
        &recorder.session,
        None,
        binding.identity,
        Digest::sha256(b"replacement exact registration"),
    )
    .expect("new local authority");
    assert_ne!(replacement, old.signature.revision);
    assert!(!pending.observation_is_current());
    assert_eq!(
        recorder.session.observed_candidate_duration(
            &old.signature,
            &old.selected.candidate,
            old.selected.kind,
        ),
        CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
    );
    assert_eq!(
        recorder.session.observed_candidate_duration(
            &other_signature,
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
    let repeat = RuntimePlacementService::whole_work_runtime_binding_session_revision(
        &recorder.session,
        None,
        binding.identity,
        Digest::sha256(b"replacement exact registration"),
    )
    .expect("unchanged generation");
    assert_eq!(repeat, replacement);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn concurrent_vm_installations_share_learning_until_canonical_profile_reset() {
    let recorder = Rc::new(RecordingSession {
        session: PlacementSession::default(),
        requests: RefCell::new(Vec::new()),
        decisions: RefCell::new(Vec::new()),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(recorder.clone()));
    // Borrow an already validated one-candidate graph. This test concerns the
    // session registration/retirement owner, not VM operation selection.
    futures::executor::block_on(
        runmat_runtime::call_builtin_sequence_async_with_outputs_in_context(
            &runtime,
            "strcmp",
            &[Value::String("a".into()), Value::String("a".into())],
            1,
        ),
    )
    .expect("fixture graph");
    let semantic = VmFusionSemanticIdentity::from_validated_digest(Digest::sha256(
        b"exact concurrent VM source",
    ));
    let installed =
        runmat_runtime::context::VmFusionInstalledRegion::install(&runtime, semantic, 7, 2, 4)
            .expect("installed VM region");
    let right = installed.right(&runtime).expect("same-context right");
    let revision = right
        .placement_revision()
        .cloned()
        .expect("real session revision");
    let mut request = recorder.requests.borrow()[0].clone();
    request.signature.revision = revision.clone();
    let pending = RuntimePlacementService::select_whole_work(&recorder.session, request)
        .expect("same-session pending selection");
    assert!(pending.observation_is_current());
    let old = pending.decision.clone();
    RuntimePlacementService::observe(
        &recorder.session,
        PlacementFeedback {
            signature: old.signature.clone(),
            candidate: old.selected.candidate.clone(),
            kind: old.selected.kind,
            total_elapsed_ns: 17,
            succeeded: true,
        },
    )
    .expect("session feedback fixture");
    assert_eq!(
        recorder.session.observed_candidate_duration(
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

    let stale = installed.right(&runtime).expect("stale-right fixture");
    assert!(pending.observation_is_current());
    let next =
        runmat_runtime::context::VmFusionInstalledRegion::install(&runtime, semantic, 7, 2, 4)
            .expect("concurrent installed VM region");
    let next_revision = next
        .right(&runtime)
        .expect("new same-context right")
        .placement_revision()
        .cloned()
        .expect("new session revision");
    assert_eq!(revision, next_revision);
    assert!(stale.planning_revision(&runtime).is_some());
    assert!(pending.observation_is_current());
    assert_eq!(
        recorder.session.observed_candidate_duration(
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

    assert!(recorder
        .session
        .restore_profile(crate::placement::PlacementProfileSnapshot {
            schema_version: crate::placement::PLACEMENT_PROFILE_SCHEMA_VERSION + 1,
            feedback: Vec::new(),
        })
        .is_err());
    assert!(pending.observation_is_current());
    assert!(stale.planning_revision(&runtime).is_some());

    recorder
        .session
        .restore_profile(crate::placement::PlacementProfileSnapshot {
            schema_version: crate::placement::PLACEMENT_PROFILE_SCHEMA_VERSION,
            feedback: Vec::new(),
        })
        .expect("valid profile replacement");
    assert!(!pending.observation_is_current());
    assert!(stale.planning_revision(&runtime).is_none());
    let fresh = installed.right(&runtime).expect("fresh post-reset right");
    assert_eq!(fresh.placement_revision(), Some(&revision));
    assert_eq!(
        recorder.session.observed_candidate_duration(
            &old.signature,
            &old.selected.candidate,
            old.selected.kind,
        ),
        CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn placement_session_drop_revokes_retained_vm_generation_permit() {
    let recorder = Rc::new(RecordingSession {
        session: PlacementSession::default(),
        requests: RefCell::new(Vec::new()),
        decisions: RefCell::new(Vec::new()),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(recorder.clone()));
    let installed = runmat_runtime::context::VmFusionInstalledRegion::install(
        &runtime,
        VmFusionSemanticIdentity::from_validated_digest(Digest::sha256(b"drop generation")),
        2,
        4,
        5,
    )
    .unwrap();
    let right = installed.right(&runtime).unwrap();
    assert!(right.placement_revision().is_some());
    drop(runtime);
    drop(recorder);
    assert!(right.placement_revision().is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn retained_feedback_lease_survives_owner_sweep_and_remains_globally_revocable() {
    let recorder = Rc::new(RecordingSession {
        session: PlacementSession::default(),
        requests: RefCell::new(Vec::new()),
        decisions: RefCell::new(Vec::new()),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(recorder.clone()));
    let semantic = VmFusionSemanticIdentity::from_validated_digest(Digest::sha256(
        b"retained feedback lease across owner sweep",
    ));
    let first =
        runmat_runtime::context::VmFusionInstalledRegion::install(&runtime, semantic, 3, 2, 5)
            .unwrap();
    let lease = first
        .right(&runtime)
        .unwrap()
        .completion_lease(&runtime)
        .expect("published feedback lease");
    assert!(lease.is_current());
    drop(first);

    let concurrent =
        runmat_runtime::context::VmFusionInstalledRegion::install(&runtime, semantic, 3, 2, 5)
            .unwrap();
    let concurrent_right = concurrent.right(&runtime).unwrap();
    assert!(concurrent_right.placement_revision().is_some());
    assert!(lease.is_current());

    recorder.session.invalidate(PlacementInvalidation::All);
    assert!(!lease.is_current());
    assert!(concurrent_right.planning_revision(&runtime).is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn vm_generation_invalidation_is_program_and_policy_scoped() {
    let recorder = Rc::new(RecordingSession {
        session: PlacementSession::default(),
        requests: RefCell::new(Vec::new()),
        decisions: RefCell::new(Vec::new()),
    });
    let services = RuntimeServicePorts::default().with_placement(recorder.clone());
    let runtime_a = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(services.clone())
    .with_program_revision(Some(program_revision(b"program A")));
    let runtime_b = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(services)
    .with_program_revision(Some(program_revision(b"program B")));
    let semantic =
        VmFusionSemanticIdentity::from_validated_digest(Digest::sha256(b"same exact VM semantics"));
    let installed_a =
        runmat_runtime::context::VmFusionInstalledRegion::install(&runtime_a, semantic, 1, 0, 1)
            .unwrap();
    let installed_b =
        runmat_runtime::context::VmFusionInstalledRegion::install(&runtime_b, semantic, 1, 0, 1)
            .unwrap();
    let stale_a = installed_a.right(&runtime_a).unwrap();
    let stale_b = installed_b.right(&runtime_b).unwrap();
    let revision_b = stale_b.placement_revision().cloned().unwrap();

    recorder.session.invalidate(PlacementInvalidation::Program {
        revision: runtime_a.program_revision().unwrap().clone(),
    });
    assert!(stale_a.planning_revision(&runtime_a).is_none());
    assert!(stale_b.planning_revision(&runtime_b).is_some());
    let fresh_a = installed_a.right(&runtime_a).unwrap();
    let revision_a = fresh_a.placement_revision().cloned().unwrap();
    let PlacementExecutionAuthorityRevision::RuntimeBindingSession(authority_a) =
        revision_a.authority
    else {
        panic!("VM revision must retain runtime-binding authority")
    };
    recorder
        .session
        .invalidate(PlacementInvalidation::RuntimeBindingSession {
            digest: authority_a,
        });
    assert!(fresh_a.planning_revision(&runtime_a).is_none());
    assert!(stale_b.planning_revision(&runtime_b).is_some());
    let fresh_a = installed_a.right(&runtime_a).unwrap();

    recorder.session.invalidate(PlacementInvalidation::Policy {
        digest: revision_b.policy,
    });
    assert!(fresh_a.planning_revision(&runtime_a).is_none());
    assert!(stale_b.planning_revision(&runtime_b).is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn vm_generation_capacity_never_evicts_live_owner_and_sweeps_dead_owner() {
    let recorder = Rc::new(RecordingSession {
        session: PlacementSession::new(crate::placement::PlacementSessionConfig {
            max_cached_decisions: 1,
            ..crate::placement::PlacementSessionConfig::default()
        }),
        requests: RefCell::new(Vec::new()),
        decisions: RefCell::new(Vec::new()),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(recorder));
    let first = runmat_runtime::context::VmFusionInstalledRegion::install(
        &runtime,
        VmFusionSemanticIdentity::from_validated_digest(Digest::sha256(b"first live semantic")),
        1,
        0,
        1,
    )
    .unwrap();
    let first_right = first.right(&runtime).unwrap();
    assert!(first_right.placement_revision().is_some());
    let second = runmat_runtime::context::VmFusionInstalledRegion::install(
        &runtime,
        VmFusionSemanticIdentity::from_validated_digest(Digest::sha256(b"second semantic")),
        2,
        2,
        3,
    )
    .unwrap();
    assert!(second
        .right(&runtime)
        .unwrap()
        .placement_revision()
        .is_none());
    drop(first_right);
    drop(first);
    assert!(second
        .right(&runtime)
        .unwrap()
        .placement_revision()
        .is_some());
}
