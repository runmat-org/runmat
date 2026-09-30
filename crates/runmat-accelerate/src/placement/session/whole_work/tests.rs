use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, PlacementFeedback, PlacementResourceSnapshot, PlacementRevision,
    PlacementSignature, ProviderResourceSnapshot, WholeWorkObservation,
    WholeWorkObservationOutcome,
};
use runmat_runtime::context::RuntimePlacementService;

use super::*;

#[test]
fn session_preserves_native_send_affinity_with_local_probe_lease() {
    fn assert_send<T: Send>() {}
    assert_send::<PlacementSession>();
    assert_send::<ProbeLease>();
}

fn signature() -> PlacementSignature {
    PlacementSignature {
        region: None,
        operation: "network".into(),
        runtime_facts: Digest::sha256(b"same-semantic-work"),
        revision: PlacementRevision {
            program: None,
            catalog: Digest::sha256(b"catalog"),
            compiler: Digest::sha256(b"compiler"),
            provider: Digest::sha256(b"provider"),
            policy: Digest::sha256(b"policy"),
        },
    }
}

fn candidate(kind: ExecutionCandidateKind) -> WholeWorkCandidate {
    let provider = kind.is_provider();
    WholeWorkCandidate {
        kind,
        execution_location: if provider {
            CandidateExecutionLocation::Provider { device_id: 7 }
        } else {
            CandidateExecutionLocation::Host
        },
        preparation: CandidatePreparationState::Warm,
        output_residency: if provider {
            CandidateOutputResidency::Provider { device_id: 7 }
        } else {
            CandidateOutputResidency::Host
        },
        route_facts: Digest::sha256(if provider { b"gpu" } else { b"cpu" }),
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoObservations),
        resources: CandidateResourceDemand {
            cpu_millicores: if provider { 0 } else { 1_000 },
            retained_bytes: 16.into(),
            scratch_bytes: 8.into(),
            queue_slots: if provider { 1 } else { 0 },
        },
        transactional_results: true,
    }
}

fn request(candidates: Vec<WholeWorkCandidate>) -> WholeWorkPlacementRequest {
    WholeWorkPlacementRequest {
        signature: signature(),
        constraint: WholeWorkConstraint::AnyLegal,
        candidates,
        resources: PlacementResourceSnapshot {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(1_024),
            cancellation_requested: false,
            providers: vec![ProviderResourceSnapshot {
                device_id: 7,
                capacity_bytes: Some(1_024),
                live_bytes: 0,
                reclaimable_bytes: 0,
                scratch_available_bytes: Some(1_024),
                queue_depth: Some(0),
                queue_limit: Some(1),
                lost: false,
                epoch: 1,
            }],
            epoch: 1,
        },
        deterministic: false,
        require_transactional_results: true,
    }
}

fn success(session: &PlacementSession, selected: &SelectedExecutionCandidate, elapsed: u64) {
    session
        .observe(PlacementFeedback {
            signature: signature(),
            candidate: selected.candidate.clone(),
            kind: selected.kind,
            total_elapsed_ns: elapsed,
            succeeded: true,
        })
        .expect("valid observation");
}

fn failure(session: &PlacementSession, selected: &SelectedExecutionCandidate) {
    session
        .observe(PlacementFeedback {
            signature: signature(),
            candidate: selected.candidate.clone(),
            kind: selected.kind,
            total_elapsed_ns: 1,
            succeeded: false,
        })
        .expect("valid failure observation");
}

#[test]
fn one_real_host_probe_then_provider_probe_after_host_cohort() {
    let session = PlacementSession::default();
    let routes = request(vec![
        candidate(ExecutionCandidateKind::SharedRuntime),
        candidate(ExecutionCandidateKind::ProviderGraph),
    ]);
    for _ in 0..2 {
        let selection = session.select_whole_work(routes.clone()).unwrap();
        assert_eq!(selection.decision.kind, WholeWorkDecisionKind::Exploratory);
        assert_eq!(
            selection.decision.selected.kind,
            ExecutionCandidateKind::SharedRuntime
        );
        assert_eq!(selection.decision.predicted_total_ns, None);
        success(&session, &selection.decision.selected, 100);
    }
    let provider = session.select_whole_work(routes).unwrap();
    assert_eq!(provider.decision.kind, WholeWorkDecisionKind::Exploratory);
    assert_eq!(
        provider.decision.selected.kind,
        ExecutionCandidateKind::ProviderGraph
    );
}

#[test]
fn one_shot_exploration_can_use_a_nontransactional_route_without_replay() {
    let session = PlacementSession::default();
    let mut provider = candidate(ExecutionCandidateKind::ProviderGraph);
    provider.transactional_results = false;
    let mut routes = request(vec![
        candidate(ExecutionCandidateKind::SharedRuntime),
        provider,
    ]);
    routes.require_transactional_results = false;
    for _ in 0..2 {
        let selected = session.select_whole_work(routes.clone()).unwrap();
        assert_eq!(
            selected.decision.selected.kind,
            ExecutionCandidateKind::SharedRuntime
        );
        success(&session, &selected.decision.selected, 100);
    }
    let selected = session.select_whole_work(routes).unwrap();
    assert_eq!(selected.decision.kind, WholeWorkDecisionKind::Exploratory);
    assert_eq!(
        selected.decision.selected.kind,
        ExecutionCandidateKind::ProviderGraph
    );
}

#[test]
fn unknown_preparation_observations_never_certify_a_warm_route() {
    let session = PlacementSession::default();
    let mut unknown = candidate(ExecutionCandidateKind::ProviderGraph);
    unknown.preparation = CandidatePreparationState::Unknown;
    let unknown_id = unknown.selected(&signature().operation);
    success(&session, &unknown_id, 20);
    success(&session, &unknown_id, 22);
    assert!(matches!(
        session.observed_candidate_duration(&signature(), &unknown_id.candidate, unknown_id.kind),
        CandidateObservationEvidence::Observed(_)
    ));
    let warm_id = candidate(ExecutionCandidateKind::ProviderGraph).selected(&signature().operation);
    assert_eq!(
        session.observed_candidate_duration(&signature(), &warm_id.candidate, warm_id.kind),
        CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
    );
}

#[test]
fn abandoned_probe_releases_reservation_without_recording_failure() {
    let session = PlacementSession::default();
    let routes = request(vec![candidate(ExecutionCandidateKind::SharedRuntime)]);
    let first = session.select_whole_work(routes.clone()).unwrap();
    assert!(first.probe.is_some());
    drop(first);
    let again = session.select_whole_work(routes).unwrap();
    assert_eq!(again.decision.kind, WholeWorkDecisionKind::Exploratory);
}

#[test]
fn simultaneous_live_probe_does_not_make_legal_host_work_unavailable() {
    let session = PlacementSession::default();
    let routes = request(vec![candidate(ExecutionCandidateKind::SharedRuntime)]);
    let first = session.select_whole_work(routes.clone()).unwrap();
    assert_eq!(first.decision.kind, WholeWorkDecisionKind::Exploratory);
    let second = session.select_whole_work(routes).unwrap();
    assert_eq!(second.decision.kind, WholeWorkDecisionKind::Unmeasured);
    assert_eq!(second.decision.predicted_total_ns, None);
    assert_eq!(second.decision.selected, first.decision.selected);
    assert!(second.probe.is_none());
    success(&session, &second.decision.selected, 100);
    drop(first);
}

#[test]
fn failed_host_baseline_does_not_starve_a_legal_provider_probe() {
    let session = PlacementSession::default();
    let routes = request(vec![
        candidate(ExecutionCandidateKind::SharedRuntime),
        candidate(ExecutionCandidateKind::ProviderGraph),
    ]);
    for _ in 0..2 {
        let selected = session.select_whole_work(routes.clone()).unwrap();
        assert_eq!(
            selected.decision.selected.kind,
            ExecutionCandidateKind::SharedRuntime
        );
        failure(&session, &selected.decision.selected);
    }
    let selected = session.select_whole_work(routes).unwrap();
    assert_eq!(
        selected.decision.selected.kind,
        ExecutionCandidateKind::ProviderGraph
    );
    assert_eq!(selected.decision.kind, WholeWorkDecisionKind::Exploratory);
}

#[test]
fn explicit_provider_can_execute_without_invented_host_cost() {
    let session = PlacementSession::default();
    let mut routes = request(vec![candidate(ExecutionCandidateKind::ProviderGraph)]);
    routes.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    let selected = session.select_whole_work(routes).unwrap();
    assert_eq!(
        selected.decision.kind,
        WholeWorkDecisionKind::ProviderRequired
    );
    assert_eq!(selected.decision.predicted_total_ns, None);
}

#[test]
fn provider_only_legal_work_does_not_claim_a_cpu_comparison() {
    let session = PlacementSession::default();
    let routes = request(vec![candidate(ExecutionCandidateKind::ProviderGraph)]);
    let selected = session.select_whole_work(routes).unwrap();
    assert_eq!(selected.decision.kind, WholeWorkDecisionKind::Uncompared);
    assert_eq!(selected.decision.predicted_total_ns, None);
}

#[test]
fn unknown_resource_bound_is_not_an_exploration_candidate() {
    let session = PlacementSession::default();
    let mut provider = candidate(ExecutionCandidateKind::ProviderGraph);
    provider.resources.scratch_bytes =
        ResourceByteBound::Unavailable(runmat_execution::ResourceBoundUnavailable::NoScratchBound);
    let routes = request(vec![
        candidate(ExecutionCandidateKind::SharedRuntime),
        provider,
    ]);
    for _ in 0..2 {
        let selected = session.select_whole_work(routes.clone()).unwrap();
        success(&session, &selected.decision.selected, 100);
    }
    let selected = session.select_whole_work(routes).unwrap();
    assert_eq!(
        selected.decision.selected.kind,
        ExecutionCandidateKind::SharedRuntime
    );
}

#[test]
fn unknown_scratch_remains_legal_when_no_matching_budget_exists() {
    let session = PlacementSession::default();
    let mut provider = candidate(ExecutionCandidateKind::ProviderGraph);
    provider.resources.scratch_bytes =
        ResourceByteBound::Unavailable(runmat_execution::ResourceBoundUnavailable::NoScratchBound);
    let mut routes = request(vec![
        candidate(ExecutionCandidateKind::SharedRuntime),
        provider,
    ]);
    routes.resources.providers[0].capacity_bytes = None;
    routes.resources.providers[0].scratch_available_bytes = None;
    for _ in 0..2 {
        let selected = session.select_whole_work(routes.clone()).unwrap();
        success(&session, &selected.decision.selected, 100_000);
    }
    for _ in 0..2 {
        let selected = session.select_whole_work(routes.clone()).unwrap();
        assert_eq!(selected.decision.kind, WholeWorkDecisionKind::Exploratory);
        assert_eq!(
            selected.decision.selected.kind,
            ExecutionCandidateKind::ProviderGraph
        );
        success(&session, &selected.decision.selected, 10_000);
    }
    let selected = session.select_whole_work(routes).unwrap();
    assert_eq!(selected.decision.kind, WholeWorkDecisionKind::Compared);
    assert_eq!(
        selected.decision.selected.kind,
        ExecutionCandidateKind::ProviderGraph
    );
}

#[test]
fn matching_budget_requires_a_real_bound_and_reports_insufficient_evidence() {
    let session = PlacementSession::default();
    let mut provider = candidate(ExecutionCandidateKind::ProviderGraph);
    provider.resources.scratch_bytes = ResourceByteBound::Unavailable(
        runmat_execution::ResourceBoundUnavailable::NoPeakMemoryBound,
    );
    let mut routes = request(vec![provider]);
    routes.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    let error = session
        .select_whole_work(routes)
        .err()
        .expect("insufficient evidence");
    assert_eq!(error.code, "RunMat:Placement:InsufficientResourceEvidence");
}

#[test]
fn known_bound_enforces_budget_and_known_sum_overflow_rejects() {
    let session = PlacementSession::default();
    let provider = candidate(ExecutionCandidateKind::ProviderGraph);
    let mut routes = request(vec![provider.clone()]);
    routes.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    routes.resources.providers[0].scratch_available_bytes = Some(7);
    assert_eq!(
        session
            .select_whole_work(routes)
            .err()
            .expect("over budget")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );

    let mut overflowing = provider;
    overflowing.resources.scratch_bytes = u64::MAX.into();
    overflowing.resources.retained_bytes = 1.into();
    let mut routes = request(vec![overflowing]);
    routes.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    routes.resources.providers[0].capacity_bytes = None;
    routes.resources.providers[0].scratch_available_bytes = None;
    assert_eq!(
        session
            .select_whole_work(routes)
            .err()
            .expect("overflow")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
}

#[test]
fn cancellation_and_abandonment_never_poison_candidate_feedback() {
    let session = PlacementSession::default();
    let routes = request(vec![candidate(ExecutionCandidateKind::ProviderGraph)]);
    let selected = session.select_whole_work(routes).unwrap().decision.selected;
    for outcome in [
        WholeWorkObservationOutcome::Cancelled,
        WholeWorkObservationOutcome::Abandoned,
    ] {
        RuntimePlacementService::observe_whole_work(
            &session,
            WholeWorkObservation {
                signature: signature(),
                selected: selected.clone(),
                elapsed_ns: 10,
                outcome,
            },
        )
        .unwrap();
    }
    assert_eq!(
        session.observed_candidate_duration(&signature(), &selected.candidate, selected.kind),
        CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
    );
}
