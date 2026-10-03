use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, PlacementFeedback, PlacementResourceSnapshot, PlacementRevision,
    PlacementSignature, ProviderReadbackDemand, ProviderReadbackDemands, ProviderResourceSnapshot,
    WholeWorkObservation, WholeWorkObservationOutcome,
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
        presentation: runmat_execution::CandidateOutputPresentation::Direct,
        route_facts: Digest::sha256(if provider { b"gpu" } else { b"cpu" }),
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoObservations),
        resources: CandidateResourceDemand {
            cpu_millicores: if provider { 0 } else { 1_000 },
            retained_bytes: 16.into(),
            scratch_bytes: 8.into(),
            queue_slots: if provider { 1 } else { 0 },
            peak_live: None,
        },
        transactional_results: true,
    }
}

fn host_with_provider_presentation() -> WholeWorkCandidate {
    let mut host = candidate(ExecutionCandidateKind::SharedRuntime);
    host.output_residency = CandidateOutputResidency::Provider { device_id: 7 };
    host.presentation = runmat_execution::CandidateOutputPresentation::ProviderUpload {
        device_id: 7,
        transfer_bytes: Some(16),
        host_result_bytes: None,
        host_scratch_bytes: 0.into(),
        provider_scratch_bytes: 8.into(),
        queue_slots: 1,
        preparation: CandidatePreparationState::Ready,
    };
    host
}

fn host_with_cross_owner_readback(scratch: ResourceByteBound) -> WholeWorkCandidate {
    let mut host = host_with_provider_presentation();
    host.presentation = runmat_execution::CandidateOutputPresentation::ProviderReadbackUpload {
        device_id: 7,
        transfer_bytes: Some(16),
        host_result_bytes: Some(ResourceByteBound::Known(16)),
        host_scratch_bytes: ResourceByteBound::Known(0),
        provider_scratch_bytes: ResourceByteBound::Known(8),
        queue_slots: 1,
        preparation: CandidatePreparationState::Ready,
        readbacks: ProviderReadbackDemands::sequential([ProviderReadbackDemand {
            device_id: 8,
            provider_scratch_bytes: scratch,
            queue_slots: 1,
        }])
        .unwrap(),
    };
    host
}

#[test]
fn cross_owner_readback_source_staging_and_queue_are_admitted_before_upload() {
    let session = PlacementSession::default();
    let mut route = request(vec![host_with_cross_owner_readback(
        ResourceByteBound::Known(8),
    )]);
    route.resources.providers.push(ProviderResourceSnapshot {
        device_id: 8,
        scratch_available_bytes: Some(8),
        queue_depth: Some(0),
        queue_limit: Some(1),
        ..Default::default()
    });
    assert!(session.select_whole_work(route.clone()).is_ok());

    route.resources.providers[1].scratch_available_bytes = Some(7);
    assert_eq!(
        session.select_whole_work(route.clone()).unwrap_err().code,
        "RunMat:Placement:NoLegalCandidate"
    );
    route.resources.providers[1].scratch_available_bytes = Some(8);
    route.resources.providers[1].queue_depth = Some(1);
    assert_eq!(
        session.select_whole_work(route.clone()).unwrap_err().code,
        "RunMat:Placement:NoLegalCandidate"
    );
    route.resources.providers[1].queue_depth = Some(0);
    route.candidates = vec![host_with_cross_owner_readback(
        ResourceByteBound::Unavailable(runmat_execution::ResourceBoundUnavailable::NoScratchBound),
    )];
    assert_eq!(
        session.select_whole_work(route).unwrap_err().code,
        "RunMat:Placement:NoLegalCandidate"
    );
}

#[test]
fn physical_peak_admits_only_incremental_live_bytes_and_does_not_relabel_scratch() {
    let session = PlacementSession::default();
    let mut provider = candidate(ExecutionCandidateKind::ProviderOperation);
    provider.resources.scratch_bytes =
        ResourceByteBound::Unavailable(runmat_execution::ResourceBoundUnavailable::NoScratchBound);
    provider.resources.peak_live = Some(runmat_execution::CandidatePeakLiveDemand {
        peak_bytes: 112,
        already_live_bytes: 80,
    });
    let mut enough = request(vec![provider.clone()]);
    enough.resources.providers[0].live_bytes = 80;
    enough.resources.providers[0].capacity_bytes = Some(112);
    enough.resources.providers[0].scratch_available_bytes = Some(32);
    assert!(session.select_whole_work(enough.clone()).is_ok());

    enough.resources.providers[0].capacity_bytes = Some(111);
    enough.resources.providers[0].scratch_available_bytes = Some(31);
    assert_eq!(
        session
            .select_whole_work(enough)
            .err()
            .expect("over budget")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
    provider.resources.peak_live = Some(runmat_execution::CandidatePeakLiveDemand {
        peak_bytes: 79,
        already_live_bytes: 80,
    });
    assert_eq!(
        session
            .select_whole_work(request(vec![provider]))
            .err()
            .expect("invalid peak")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
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
fn admitted_host_upload_is_a_distinct_complete_route() {
    let session = PlacementSession::default();
    let host_upload = host_with_provider_presentation();
    let direct_host = candidate(ExecutionCandidateKind::SharedRuntime);
    assert_ne!(
        host_upload.selected("network"),
        direct_host.selected("network")
    );
    let selection = session
        .select_whole_work(request(vec![host_upload.clone()]))
        .expect("both execution and presentation resources are admitted");
    assert_eq!(
        selection.decision.selected.kind,
        ExecutionCandidateKind::SharedRuntime
    );
    assert_eq!(selection.decision.selected, host_upload.selected("network"));
    drop(selection);

    let mut alternate_stage = host_upload;
    alternate_stage.presentation = runmat_execution::CandidateOutputPresentation::ProviderUpload {
        device_id: 7,
        transfer_bytes: Some(32),
        host_result_bytes: None,
        host_scratch_bytes: 0.into(),
        provider_scratch_bytes: 8.into(),
        queue_slots: 1,
        preparation: CandidatePreparationState::Ready,
    };
    assert_ne!(
        alternate_stage.selected("network"),
        direct_host.selected("network")
    );
    assert_ne!(
        alternate_stage.selected("network"),
        host_with_provider_presentation().selected("network")
    );
}

#[test]
fn host_upload_requires_both_host_and_target_provider_resources() {
    let session = PlacementSession::default();
    let host_upload = host_with_provider_presentation();
    let mut cases = Vec::new();
    let mut no_host_cpu = request(vec![host_upload.clone()]);
    no_host_cpu.resources.cpu_millicores_available = 999;
    cases.push(no_host_cpu);
    let mut no_host_staging_memory = request(vec![host_upload.clone()]);
    no_host_staging_memory.resources.memory_available_bytes = Some(23);
    cases.push(no_host_staging_memory);
    let mut host_scratch = host_upload.clone();
    host_scratch.presentation = runmat_execution::CandidateOutputPresentation::ProviderUpload {
        device_id: 7,
        transfer_bytes: Some(16),
        host_result_bytes: None,
        host_scratch_bytes: 1.into(),
        provider_scratch_bytes: 8.into(),
        queue_slots: 1,
        preparation: CandidatePreparationState::Ready,
    };
    let mut no_host_upload_scratch = request(vec![host_scratch]);
    no_host_upload_scratch.resources.memory_available_bytes = Some(24);
    cases.push(no_host_upload_scratch);
    let mut no_provider_output_memory = request(vec![host_upload.clone()]);
    no_provider_output_memory.resources.providers[0].capacity_bytes = Some(23);
    no_provider_output_memory.resources.providers[0].scratch_available_bytes = Some(23);
    cases.push(no_provider_output_memory);
    let mut no_provider_scratch = request(vec![host_upload.clone()]);
    no_provider_scratch.resources.providers[0].scratch_available_bytes = Some(7);
    cases.push(no_provider_scratch);
    let mut no_provider_queue = request(vec![host_upload.clone()]);
    no_provider_queue.resources.providers[0].queue_depth = Some(1);
    cases.push(no_provider_queue);
    let mut lost_provider = request(vec![host_upload]);
    lost_provider.resources.providers[0].lost = true;
    cases.push(lost_provider);
    for rejected in cases {
        assert_eq!(
            session
                .select_whole_work(rejected)
                .err()
                .expect("rejected route")
                .code,
            "RunMat:Placement:NoLegalCandidate"
        );
    }
}

#[test]
fn upload_admission_distinguishes_host_payload_from_provider_retention() {
    let session = PlacementSession::default();
    let mut host = host_with_provider_presentation();
    host.resources.scratch_bytes = 0.into();
    host.presentation = runmat_execution::CandidateOutputPresentation::ProviderUpload {
        device_id: 7,
        transfer_bytes: Some(16),
        host_result_bytes: Some(32.into()),
        host_scratch_bytes: 0.into(),
        provider_scratch_bytes: 0.into(),
        queue_slots: 1,
        preparation: CandidatePreparationState::Ready,
    };
    let mut enough = request(vec![host.clone()]);
    enough.resources.memory_available_bytes = Some(32);
    enough.resources.providers[0].capacity_bytes = Some(16);
    enough.resources.providers[0].scratch_available_bytes = Some(16);
    assert!(session.select_whole_work(enough.clone()).is_ok());
    enough.resources.memory_available_bytes = Some(31);
    assert_eq!(
        session
            .select_whole_work(enough)
            .err()
            .expect("host payload exceeds memory")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
}

#[test]
fn mirrored_upload_uses_distinct_host_and_provider_retention_once() {
    let session = PlacementSession::default();
    let mut host = host_with_provider_presentation();
    host.output_residency = CandidateOutputResidency::Mirrored { device_id: 7 };
    host.resources.scratch_bytes = 0.into();
    host.presentation = runmat_execution::CandidateOutputPresentation::ProviderUpload {
        device_id: 7,
        transfer_bytes: Some(16),
        host_result_bytes: Some(32.into()),
        host_scratch_bytes: 0.into(),
        provider_scratch_bytes: 0.into(),
        queue_slots: 1,
        preparation: CandidatePreparationState::Ready,
    };
    let mut enough = request(vec![host]);
    enough.resources.memory_available_bytes = Some(32);
    enough.resources.providers[0].capacity_bytes = Some(16);
    enough.resources.providers[0].scratch_available_bytes = Some(16);
    assert!(session.select_whole_work(enough.clone()).is_ok());
    enough.resources.memory_available_bytes = Some(31);
    assert_eq!(
        session
            .select_whole_work(enough)
            .err()
            .expect("host under budget")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
}

#[test]
fn bounded_host_compute_and_upload_admit_under_finite_memory() {
    let session = PlacementSession::default();
    let mut host = host_with_provider_presentation();
    host.resources.scratch_bytes = 80.into();
    host.presentation = runmat_execution::CandidateOutputPresentation::ProviderUpload {
        device_id: 7,
        transfer_bytes: Some(16),
        host_result_bytes: Some(16.into()),
        host_scratch_bytes: 16.into(),
        provider_scratch_bytes: 8.into(),
        queue_slots: 1,
        preparation: CandidatePreparationState::Ready,
    };
    let mut bounded = request(vec![host]);
    bounded.resources.memory_available_bytes = Some(112);
    assert!(session.select_whole_work(bounded.clone()).is_ok());
    bounded.resources.memory_available_bytes = Some(111);
    assert_eq!(
        session
            .select_whole_work(bounded)
            .err()
            .expect("one byte short")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
}

#[test]
fn host_sink_download_stage_obeys_provider_staging_and_host_payload_budgets() {
    let session = PlacementSession::default();
    let mut host = candidate(ExecutionCandidateKind::SharedRuntime);
    host.presentation = runmat_execution::CandidateOutputPresentation::ProviderDownload {
        device_id: 7,
        provider_scratch_bytes: 8.into(),
        queue_slots: 1,
    };
    let mut bounded = request(vec![host.clone()]);
    bounded.resources.providers[0].scratch_available_bytes = Some(8);
    bounded.resources.memory_available_bytes = Some(24);
    assert!(session.select_whole_work(bounded.clone()).is_ok());
    bounded.resources.providers[0].scratch_available_bytes = Some(7);
    assert_eq!(
        session.select_whole_work(bounded.clone()).unwrap_err().code,
        "RunMat:Placement:NoLegalCandidate"
    );
    bounded.resources.providers[0].scratch_available_bytes = Some(8);
    bounded.resources.memory_available_bytes = Some(23);
    assert_eq!(
        session.select_whole_work(bounded).unwrap_err().code,
        "RunMat:Placement:NoLegalCandidate"
    );
    host.presentation = runmat_execution::CandidateOutputPresentation::ProviderDownload {
        device_id: 7,
        provider_scratch_bytes: ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        ),
        queue_slots: 1,
    };
    assert_eq!(
        session
            .select_whole_work(request(vec![host]))
            .unwrap_err()
            .code,
        "RunMat:Placement:InsufficientResourceEvidence"
    );
}

#[test]
fn provider_required_does_not_admit_a_host_upload_route() {
    let session = PlacementSession::default();
    let mut only_host_upload = request(vec![host_with_provider_presentation()]);
    only_host_upload.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    assert_eq!(
        session
            .select_whole_work(only_host_upload)
            .err()
            .expect("explicit provider required")
            .code,
        "RunMat:Placement:InvalidRequest"
    );
}

#[test]
fn host_upload_needs_resource_evidence_and_transactional_publication() {
    let session = PlacementSession::default();
    let mut host = host_with_provider_presentation();
    host.presentation = runmat_execution::CandidateOutputPresentation::ProviderUpload {
        device_id: 7,
        transfer_bytes: None,
        host_result_bytes: None,
        host_scratch_bytes: 0.into(),
        provider_scratch_bytes: ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        ),
        queue_slots: 1,
        preparation: CandidatePreparationState::Ready,
    };
    let error = session
        .select_whole_work(request(vec![host.clone()]))
        .err()
        .expect("unknown scratch cannot satisfy a known provider budget");
    assert_eq!(error.code, "RunMat:Placement:InsufficientResourceEvidence");

    host.presentation = host_with_provider_presentation().presentation;
    host.transactional_results = false;
    let error = session
        .select_whole_work(request(vec![host]))
        .err()
        .expect("upload publication must be staged when required");
    assert_eq!(error.code, "RunMat:Placement:NoLegalCandidate");
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
