use crate::placement::PlacementSessionConfig;
use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, PlacementFeedback, PlacementResourceSnapshot, PlacementRevision,
    PlacementSignature, ProviderInputStageDemand, ProviderInputStageDemands,
    ProviderReadbackDemand, ProviderReadbackDemands, ProviderResourceSnapshot,
    SelectedExecutionCandidate, WholeWorkObservation, WholeWorkObservationOutcome,
};
use runmat_runtime::context::RuntimePlacementFeedbackSource;
use runmat_runtime::context::RuntimePlacementService;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use super::*;

#[test]
fn unobservable_native_source_does_not_starve_observable_host_routes() {
    let mut source = candidate(ExecutionCandidateKind::OriginalNativeRegion);
    source.execution_location = CandidateExecutionLocation::NativeSource;
    source.preparation = CandidatePreparationState::Ready;
    source.output_residency = CandidateOutputResidency::Unknown;
    source.transactional_results = false;
    let unknown = ResourceByteBound::Unavailable(
        runmat_execution::ResourceBoundUnavailable::NoPeakMemoryBound,
    );
    source.resources = CandidateResourceDemand {
        retained_bytes: unknown,
        scratch_bytes: unknown,
        ..Default::default()
    };
    let routes = |candidates| {
        let mut routes = request(candidates);
        routes.resources.memory_available_bytes = None;
        routes.resources.providers.clear();
        routes.require_transactional_results = false;
        routes
    };

    // Its per-operation owners still provide a legal ordinary fallback, but
    // there is no complete-call completion boundary from which to learn.
    let fallback = PlacementSession::default()
        .select_whole_work(routes(vec![source.clone()]))
        .expect("ordinary native source remains legal");
    assert_eq!(
        fallback.decision.selected.kind,
        ExecutionCandidateKind::OriginalNativeRegion
    );
    assert_ne!(fallback.decision.kind, WholeWorkDecisionKind::Exploratory);
    assert!(fallback.probe.is_none());

    // An observable unmeasured host still receives the normal real-call probe,
    // even when the unobservable source fallback appears first.
    let session = PlacementSession::default();
    let host = candidate(ExecutionCandidateKind::SharedRuntime);
    let selected = session
        .select_whole_work(routes(vec![source.clone(), host]))
        .expect("observable host can be probed");
    assert_eq!(
        selected.decision.selected.kind,
        ExecutionCandidateKind::SharedRuntime
    );
    assert_eq!(selected.decision.kind, WholeWorkDecisionKind::Exploratory);
    assert!(selected.probe.is_some());

    let mut vectorized = candidate(ExecutionCandidateKind::VectorizedNativeCpu);
    vectorized.cost = CandidateCostEvidence::Estimated(runmat_execution::ExecutionCostEstimate {
        components: runmat_execution::ExecutionCostComponents {
            execution_ns: 100,
            ..Default::default()
        },
        scratch_bytes: Some(8),
        confidence: runmat_execution::EstimateConfidence::Prior,
        source: runmat_execution::EstimateSource::Compiler,
    });
    let session = PlacementSession::default();
    for _ in 0..3 {
        let selected = session
            .select_whole_work(routes(vec![source.clone(), vectorized.clone()]))
            .expect("existing priced route is not starved by an unobservable probe");
        assert_eq!(
            selected.decision.selected.kind,
            ExecutionCandidateKind::VectorizedNativeCpu
        );
        assert_ne!(selected.decision.kind, WholeWorkDecisionKind::Exploratory);
        RuntimePlacementService::observe_selected_whole_work(
            &session,
            &selected,
            100,
            WholeWorkObservationOutcome::Success,
        )
        .expect("real selected host completion retains its feedback right");
    }
    let feedback = session.profile_snapshot().feedback;
    assert!(!feedback.is_empty());
    assert!(feedback
        .iter()
        .all(|series| series.kind == ExecutionCandidateKind::VectorizedNativeCpu));
}

#[test]
fn fixed_size_host_route_uses_the_same_placement_policy_without_candidate_vec() {
    let multi = request(vec![candidate(ExecutionCandidateKind::SharedRuntime)]);
    let single = WholeWorkPlacementRequestRef {
        signature: (&multi.signature).into(),
        constraint: multi.constraint,
        candidates: std::slice::from_ref(&multi.candidates[0]),
        resources: &multi.resources,
        deterministic: multi.deterministic,
        require_transactional_results: multi.require_transactional_results,
    };
    let owned = PlacementSession::default()
        .select_whole_work(multi.clone())
        .expect("owned route");
    let fixed = PlacementSession::default()
        .select_whole_work_borrowed(single)
        .expect("fixed-size route");
    assert_eq!(fixed.decision, owned.decision);
}

#[test]
fn managed_selection_uses_one_precharged_slot_and_fences_late_invalidation() {
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;

    let ledger = ManagedHostEffectLedger::new(16 * 1024);
    let session = PlacementSession::new_with_managed_observations(
        PlacementSessionConfig::default(),
        &ledger,
        super::super::ManagedPlacementCapacity {
            in_flight_slots: 1,
            feedback_series: 1,
            signature_bytes_per_series: 1024,
        },
    )
    .expect("managed slot backing");
    let other = PlacementSession::new_with_managed_observations(
        PlacementSessionConfig::default(),
        &ManagedHostEffectLedger::new(16 * 1024),
        super::super::ManagedPlacementCapacity {
            in_flight_slots: 1,
            feedback_series: 1,
            signature_bytes_per_series: 1024,
        },
    )
    .expect("independent session owner");
    let route = request(vec![candidate(ExecutionCandidateKind::SharedRuntime)]);
    let first = session
        .select_whole_work_borrowed_managed(route.as_ref())
        .expect("first managed call");
    assert_eq!(first.kind, WholeWorkDecisionKind::Exploratory);
    assert!(!first
        .observation_lease
        .same_owner(other.managed_observations.as_ref().unwrap()));
    assert!(first.observation_is_current());
    let exhausted = session.select_whole_work_borrowed_managed(route.as_ref());
    assert_eq!(
        exhausted.err().expect("one slot is exhausted").code,
        "RunMat:Placement:InsufficientResourceEvidence"
    );
    session.invalidate(runmat_execution::PlacementInvalidation::All);
    assert!(!first.observation_is_current());
    drop(first);
    let replacement = session
        .select_whole_work_borrowed_managed(route.as_ref())
        .expect("retired slot reusable with next generation");
    assert!(replacement.observation_is_current());
    drop(replacement);
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    drop(session);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

struct CountedReadyFeedback {
    signature: PlacementSignature,
    polls: Arc<AtomicUsize>,
    permit: Arc<dyn runmat_runtime::context::RuntimeGraphObservationPermit>,
}

impl RuntimePlacementFeedbackSource for CountedReadyFeedback {
    fn signature(&self) -> &PlacementSignature {
        &self.signature
    }

    fn take_ready(&mut self) -> Option<runmat_runtime::context::RuntimePlacementFeedbackDelivery> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        Some(
            runmat_runtime::context::RuntimePlacementFeedbackDelivery::Graph {
                feedback: PlacementFeedback {
                    signature: self.signature.clone(),
                    candidate: "unrelated-provider".into(),
                    kind: ExecutionCandidateKind::ProviderOperation,
                    total_elapsed_ns: 11,
                    succeeded: true,
                },
                permit: Arc::clone(&self.permit),
            },
        )
    }
}

#[test]
fn managed_selection_does_not_drain_unrelated_dynamic_feedback() {
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;

    let ledger = ManagedHostEffectLedger::new(16 * 1024);
    let session = PlacementSession::new_with_managed_observations(
        PlacementSessionConfig::default(),
        &ledger,
        super::super::ManagedPlacementCapacity {
            in_flight_slots: 1,
            feedback_series: 1,
            signature_bytes_per_series: 1024,
        },
    )
    .expect("managed backing");
    let route = request(vec![candidate(ExecutionCandidateKind::SharedRuntime)]);
    let polls = Arc::new(AtomicUsize::new(0));
    assert!(RuntimePlacementService::defer_feedback(
        &session,
        Box::new(CountedReadyFeedback {
            signature: route.signature.clone(),
            polls: polls.clone(),
            permit: session.test_graph_observation_permit(
                route.signature.clone(),
                "unrelated-provider",
                ExecutionCandidateKind::ProviderOperation
            ),
        }),
    )
    .is_ok());
    let error = session
        .select_whole_work_borrowed_managed(route.as_ref())
        .err()
        .expect("pending dynamic work is not accounted by managed route");
    assert_eq!(error.code, "RunMat:Placement:InsufficientResourceEvidence");
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert!(session.pending.has_sources());

    RuntimePlacementService::drain_feedback(&session);
    assert_eq!(polls.load(Ordering::SeqCst), 1);
    let selected = session
        .select_whole_work_borrowed_managed(route.as_ref())
        .expect("managed selection after explicit ordinary drain");
    drop(selected);
    assert_eq!(session.state.borrow().feedback.len(), 1);
}

#[test]
fn managed_completion_uses_preallocated_series_and_preserves_portable_profile() {
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;

    let ledger = ManagedHostEffectLedger::new(16 * 1024);
    let session = PlacementSession::new_with_managed_observations(
        PlacementSessionConfig::default(),
        &ledger,
        super::super::ManagedPlacementCapacity {
            in_flight_slots: 1,
            feedback_series: 1,
            signature_bytes_per_series: 1024,
        },
    )
    .unwrap();
    let route = request(vec![candidate(ExecutionCandidateKind::SharedRuntime)]);
    let candidate_digest = route.candidates[0].identity_digest();
    for elapsed in [17, 19] {
        let selection = session
            .select_whole_work_borrowed_managed(route.as_ref())
            .expect("managed admission before effects");
        if elapsed == 17 {
            RuntimePlacementService::observe_managed_whole_work(
                &session,
                &selection,
                elapsed,
                WholeWorkObservationOutcome::Success,
            )
            .expect("inline managed feedback");
        } else {
            let mut receipt = runmat_accelerate_api::observation::ProviderOperationReceipt::new();
            let submission = receipt.recorder().submission().unwrap();
            receipt.publish().unwrap();
            let mut source = crate::placement::completion::managed_whole_work_source_for_test(
                receipt,
                selection,
                route.signature.clone(),
            );
            assert!(source.take_ready().is_none(), "no sample before completion");
            submission.complete();
            let delivery = source.take_ready().expect("managed terminal completion");
            assert!(matches!(
                delivery,
                runmat_runtime::context::RuntimePlacementFeedbackDelivery::ManagedWholeWork { .. }
            ));
            assert!(delivery.snapshot().is_none(), "no portable metadata clone");
            RuntimePlacementService::observe_deferred_feedback(&session, delivery)
                .expect("original managed observation owner");
            assert!(source.take_ready().is_none(), "one terminal delivery only");
        }
    }
    let observed = session.observed_candidate_duration_digest(
        (&route.signature).into(),
        candidate_digest,
        ExecutionCandidateKind::SharedRuntime,
        true,
    );
    assert!(matches!(
        observed,
        CandidateObservationEvidence::Observed(_)
    ));
    let profile = session.profile_snapshot();
    assert_eq!(profile.feedback.len(), 1);
    assert_eq!(profile.feedback[0].signature, route.signature);
    assert_eq!(profile.feedback[0].candidate, candidate_digest.to_string());
    assert_eq!(profile.feedback[0].samples, 2);
    assert!(ledger.snapshot().live_reserved_bytes > 0);
}

#[test]
fn managed_selection_seeds_from_explicit_profile_import_without_duplicate_export() {
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;

    let route = request(vec![candidate(ExecutionCandidateKind::SharedRuntime)]);
    let source = PlacementSession::default();
    // The shared confidence policy needs two completed observations before
    // an imported duration can replace exploratory execution.
    for total_elapsed_ns in [23, 25] {
        source
            .observe(runmat_execution::PlacementFeedback {
                signature: route.signature.clone(),
                candidate: route.candidates[0].identity_digest().to_string(),
                kind: route.candidates[0].kind,
                total_elapsed_ns,
                succeeded: true,
            })
            .expect("portable source profile");
    }
    let imported = source.profile_snapshot();
    let managed = PlacementSession::new_with_managed_observations(
        PlacementSessionConfig::default(),
        &ManagedHostEffectLedger::new(16 * 1024),
        super::super::ManagedPlacementCapacity {
            in_flight_slots: 1,
            feedback_series: 1,
            signature_bytes_per_series: 1024,
        },
    )
    .expect("precharged backing");
    managed.restore_profile(imported).expect("explicit import");
    let selected = managed
        .select_whole_work_borrowed_managed(route.as_ref())
        .expect("matching finite selection");
    assert_eq!(selected.kind, WholeWorkDecisionKind::Uncompared);
    assert_eq!(selected.predicted_total_ns, Some(24));
    RuntimePlacementService::observe_managed_whole_work(
        &managed,
        &selected,
        29,
        WholeWorkObservationOutcome::Success,
    )
    .expect("managed completion");
    let exported = managed.profile_snapshot();
    assert_eq!(exported.feedback.len(), 1);
    assert_eq!(exported.feedback[0].samples, 3);
    assert_eq!(exported.feedback[0].total_elapsed_ns, 77);
}

#[test]
fn compact_selection_retains_unicode_operation_and_portable_identity() {
    let route = candidate(ExecutionCandidateKind::SharedRuntime);
    let operation = "foreign.λ";
    let compact = CompactSelectedExecutionCandidate {
        node: PlacementOperationId::new(operation).expect("valid operation"),
        candidate: route.identity_digest(),
        kind: route.kind,
    };
    assert_eq!(compact.node.as_str(), operation);
    assert_eq!(compact.portable(), route.selected(operation));
    assert!(PlacementOperationId::new("").is_err());
    assert!(PlacementOperationId::new("\n").is_err());
    assert!(PlacementOperationId::new(&"x".repeat(129)).is_err());
}

#[test]
fn owned_and_borrowed_compared_routes_use_the_same_graph_policy() {
    let estimate = |execution_ns| {
        CandidateCostEvidence::Estimated(runmat_execution::ExecutionCostEstimate {
            components: runmat_execution::ExecutionCostComponents {
                execution_ns,
                ..Default::default()
            },
            scratch_bytes: Some(8),
            confidence: runmat_execution::EstimateConfidence::Exact,
            source: runmat_execution::EstimateSource::Compiler,
        })
    };
    let mut host = candidate(ExecutionCandidateKind::SharedRuntime);
    host.cost = estimate(100);
    let mut provider = candidate(ExecutionCandidateKind::ProviderGraph);
    provider.cost = estimate(20);
    let (_, kind, prediction) = select_unobserved(
        "network",
        std::slice::from_ref(&host),
        WholeWorkConstraint::AnyLegal,
        &request(vec![host.clone()]).resources,
        true,
    )
    .expect("a context-free host quote needs no synthetic session");
    assert_eq!(kind, WholeWorkDecisionKind::Uncompared);
    assert_eq!(prediction, Some(100));
    let single = PlacementSession::default()
        .select_whole_work(request(vec![host.clone()]))
        .expect("one valid host price remains executable");
    assert_eq!(single.decision.kind, WholeWorkDecisionKind::Uncompared);
    assert_eq!(single.decision.predicted_total_ns, Some(100));
    assert!(single.observation_is_current());

    let mut unquoted = provider.clone();
    unquoted.cost = CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate);
    let unpriced = request(vec![host.clone(), unquoted]);
    let session = PlacementSession::default();
    let probe = session
        .select_whole_work(unpriced.clone())
        .expect("real bounded provider probe");
    assert_eq!(probe.decision.kind, WholeWorkDecisionKind::Exploratory);
    let ordinary = session
        .select_whole_work(unpriced)
        .expect("live probe does not block host work");
    assert_eq!(ordinary.decision.kind, WholeWorkDecisionKind::Unmeasured);
    assert_eq!(ordinary.decision.predicted_total_ns, Some(100));
    assert_eq!(
        ordinary.decision.selected.kind,
        ExecutionCandidateKind::SharedRuntime
    );
    drop(probe);
    let routes = request(vec![host, provider]);
    let owned = PlacementSession::default()
        .select_whole_work(routes.clone())
        .expect("owned compared route");
    let borrowed = PlacementSession::default()
        .select_whole_work_borrowed(routes.as_ref())
        .expect("borrowed compared route");
    assert_eq!(owned.decision.kind, WholeWorkDecisionKind::Compared);
    assert_eq!(owned.decision, borrowed.decision);
    assert_eq!(owned.decision.signature, routes.signature);
}

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
        provider_provenance: None,
        graph_output_authority: None,
        revision: PlacementRevision {
            program: None,
            catalog: Digest::sha256(b"catalog"),
            compiler: Digest::sha256(b"compiler"),
            authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(
                Digest::sha256(b"provider"),
            ),
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
            provider_host_workspace: None,
            provider_input_stages: None,
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
        readbacks: Box::new(
            ProviderReadbackDemands::sequential([ProviderReadbackDemand {
                device_id: 8,
                provider_scratch_bytes: scratch,
                queue_slots: 1,
            }])
            .unwrap(),
        ),
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
        session
            .select_whole_work(route.clone())
            .err()
            .expect("resource-constrained route must reject")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
    route.resources.providers[1].scratch_available_bytes = Some(8);
    route.resources.providers[1].queue_depth = Some(1);
    assert_eq!(
        session
            .select_whole_work(route.clone())
            .err()
            .expect("resource-constrained route must reject")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
    route.resources.providers[1].queue_depth = Some(0);
    route.candidates = vec![host_with_cross_owner_readback(
        ResourceByteBound::Unavailable(runmat_execution::ResourceBoundUnavailable::NoScratchBound),
    )];
    assert_eq!(
        session
            .select_whole_work(route)
            .err()
            .expect("unknown source staging cannot satisfy a finite provider budget")
            .code,
        "RunMat:Placement:InsufficientResourceEvidence"
    );
}

#[test]
fn same_owner_readback_and_retained_output_use_sequential_physical_peak() {
    let session = PlacementSession::default();
    let mut host = host_with_provider_presentation();
    host.resources.retained_bytes = ResourceByteBound::Known(8);
    host.presentation = runmat_execution::CandidateOutputPresentation::ProviderReadbackUpload {
        device_id: 7,
        transfer_bytes: Some(8),
        host_result_bytes: Some(ResourceByteBound::Known(8)),
        host_scratch_bytes: ResourceByteBound::Known(0),
        provider_scratch_bytes: ResourceByteBound::Known(0),
        queue_slots: 1,
        preparation: CandidatePreparationState::Ready,
        readbacks: Box::new(
            ProviderReadbackDemands::sequential([ProviderReadbackDemand {
                device_id: 7,
                provider_scratch_bytes: ResourceByteBound::Known(8),
                queue_slots: 1,
            }])
            .unwrap(),
        ),
    };
    let mut route = request(vec![host]);
    route.resources.providers[0].live_bytes = 8;
    route.resources.providers[0].capacity_bytes = Some(16);
    route.resources.providers[0].scratch_available_bytes = Some(8);
    assert!(session.select_whole_work(route.clone()).is_ok());
    route.resources.providers[0].capacity_bytes = Some(15);
    assert_eq!(
        session
            .select_whole_work(route)
            .err()
            .expect("resource-constrained route must reject")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
}

#[test]
fn empty_host_sink_needs_no_staging_or_queue_slot_under_finite_zero_quotas() {
    let session = PlacementSession::default();
    let mut host = candidate(ExecutionCandidateKind::SharedRuntime);
    host.resources.retained_bytes = ResourceByteBound::Known(0);
    host.resources.scratch_bytes = ResourceByteBound::Known(0);
    host.presentation = runmat_execution::CandidateOutputPresentation::ProviderDownload {
        device_id: 7,
        provider_scratch_bytes: ResourceByteBound::Known(0),
        queue_slots: 0,
    };
    let mut route = request(vec![host]);
    route.resources.memory_available_bytes = Some(0);
    route.resources.providers[0].scratch_available_bytes = Some(0);
    route.resources.providers[0].queue_depth = Some(0);
    route.resources.providers[0].queue_limit = Some(0);
    assert!(session.select_whole_work(route.clone()).is_ok());
    if let runmat_execution::CandidateOutputPresentation::ProviderDownload { queue_slots, .. } =
        &mut route.candidates[0].presentation
    {
        *queue_slots = 1;
    }
    assert_eq!(
        session
            .select_whole_work(route)
            .err()
            .expect("resource-constrained route must reject")
            .code,
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
fn deferred_whole_work_delivery_keeps_exact_probe_until_terminal_fence() {
    use runmat_accelerate_api::observation::{
        ProviderObservationFailure, ProviderOperationReceipt,
    };
    // Receipt/authority mechanism fixture, not an executed provider workload
    // or a qualification of a numerical FullCall route.
    for terminal in ["complete", "cancel", "invalidate", "drop"] {
        let session = PlacementSession::default();
        let mut host = candidate(ExecutionCandidateKind::SharedRuntime);
        host.cost = CandidateCostEvidence::Estimated(runmat_execution::ExecutionCostEstimate {
            components: runmat_execution::ExecutionCostComponents {
                execution_ns: 100,
                ..Default::default()
            },
            scratch_bytes: Some(8),
            confidence: runmat_execution::EstimateConfidence::Exact,
            source: runmat_execution::EstimateSource::Compiler,
        });
        let routes = request(vec![host, candidate(ExecutionCandidateKind::ProviderGraph)]);
        let selected = session.select_whole_work(routes.clone()).unwrap();
        assert_eq!(selected.decision.kind, WholeWorkDecisionKind::Exploratory);
        let mut receipt = ProviderOperationReceipt::new();
        let recorder = receipt.recorder();
        let submission = recorder.submission().unwrap();
        receipt.publish().unwrap();
        let mut source =
            crate::placement::completion::whole_work_source_for_test(receipt, selected);
        assert!(source.take_ready().is_none());
        assert!(!source.terminal_without_feedback());
        let while_pending = session.select_whole_work(routes.clone()).unwrap();
        assert_eq!(
            while_pending.decision.kind,
            WholeWorkDecisionKind::Unmeasured
        );
        drop(while_pending);
        match terminal {
            "complete" => {
                submission.complete();
                let delivery = source.take_ready().expect("terminal provider fence");
                assert!(matches!(
                    &delivery,
                    runmat_runtime::context::RuntimePlacementFeedbackDelivery::WholeWork { .. }
                ));
                RuntimePlacementService::observe_deferred_feedback(&session, delivery).unwrap();
                assert_eq!(session.profile_snapshot().feedback.len(), 1);
                assert_eq!(
                    session.profile_snapshot().feedback[0].kind,
                    ExecutionCandidateKind::ProviderGraph
                );
                assert!(source.take_ready().is_none(), "one terminal delivery only");
            }
            "cancel" => {
                recorder.fail(ProviderObservationFailure::Cancelled);
                submission.complete();
                assert!(source.take_ready().is_none());
                assert!(source.terminal_without_feedback());
            }
            "invalidate" => {
                session.invalidate(runmat_execution::PlacementInvalidation::All);
                submission.complete();
                assert!(source.take_ready().is_none());
                assert!(source.terminal_without_feedback());
            }
            "drop" => drop(submission),
            _ => unreachable!(),
        }
        drop(source);
        if terminal != "complete" {
            assert!(session.profile_snapshot().feedback.is_empty());
        }
        let fresh = session.select_whole_work(routes).unwrap();
        assert_eq!(
            fresh.decision.kind,
            WholeWorkDecisionKind::Exploratory,
            "terminal route releases its original probe"
        );
    }
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
    let route = candidate(ExecutionCandidateKind::ProviderGraph);
    let (direct, kind, predicted) = select_unobserved(
        "network",
        std::slice::from_ref(&route),
        WholeWorkConstraint::ProviderRequired { device_id: 7 },
        &request(vec![route.clone()]).resources,
        true,
    )
    .expect("same context-free provider-only policy");
    assert_eq!(direct, route.selected("network"));
    assert_eq!(kind, WholeWorkDecisionKind::ProviderRequired);
    assert_eq!(predicted, None);

    // Missing observation rights do not remove physical resource admission.
    // Keep the same unquoted route/identity while varying actual constraints.
    for failure in ["lost", "queue", "quota", "unknown-scratch", "cancel"] {
        let mut provider = route.clone();
        let mut resources = request(vec![provider.clone()]).resources;
        match failure {
            "lost" => resources.providers[0].lost = true,
            "queue" => resources.providers[0].queue_depth = resources.providers[0].queue_limit,
            "quota" => {
                resources.providers[0].capacity_bytes = Some(16);
                resources.providers[0].scratch_available_bytes = Some(16);
            }
            "unknown-scratch" => {
                provider.resources.scratch_bytes = ResourceByteBound::Unavailable(
                    runmat_execution::ResourceBoundUnavailable::NoScratchBound,
                )
            }
            "cancel" => resources.cancellation_requested = true,
            _ => unreachable!(),
        }
        let error = select_unobserved(
            "network",
            std::slice::from_ref(&provider),
            WholeWorkConstraint::ProviderRequired { device_id: 7 },
            &resources,
            true,
        )
        .expect_err("provider intent cannot bypass physical admission");
        assert_eq!(
            error.code,
            match failure {
                "unknown-scratch" => "RunMat:Placement:InsufficientResourceEvidence",
                "cancel" => "RunMat:Placement:Cancelled",
                _ => "RunMat:Placement:NoLegalCandidate",
            }
        );
        let host = candidate(ExecutionCandidateKind::SharedRuntime);
        let ordinary = select_unobserved(
            "network",
            &[host, provider],
            WholeWorkConstraint::AnyLegal,
            &resources,
            true,
        );
        if failure == "cancel" {
            assert_eq!(ordinary.unwrap_err().code, "RunMat:Placement:Cancelled");
        } else {
            let (selected, _, _) = ordinary.expect("independently legal host remains available");
            assert_eq!(selected.kind, ExecutionCandidateKind::SharedRuntime);
        }
    }

    let host = candidate(ExecutionCandidateKind::SharedRuntime);
    let resources = request(vec![host.clone(), route.clone()]).resources;
    let (_, kind, predicted) = select_unobserved(
        "network",
        &[host, route],
        WholeWorkConstraint::AnyLegal,
        &resources,
        true,
    )
    .expect("ordinary host needs no invented session or learning permit");
    assert_eq!(kind, WholeWorkDecisionKind::Unmeasured);
    assert_eq!(predicted, None);
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
fn provider_hook_host_workspace_coexists_with_physical_output_peak() {
    let session = PlacementSession::default();
    let mut provider = candidate(ExecutionCandidateKind::ProviderOperation);
    provider.resources.retained_bytes = 16.into();
    provider.resources.scratch_bytes =
        ResourceByteBound::Unavailable(runmat_execution::ResourceBoundUnavailable::NoScratchBound);
    provider.resources.peak_live = Some(runmat_execution::CandidatePeakLiveDemand {
        peak_bytes: 24,
        already_live_bytes: 0,
    });
    provider.resources.provider_host_workspace = Some(64.into());
    let mut bounded = request(vec![provider.clone()]);
    bounded.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    bounded.resources.memory_available_bytes = Some(64);
    bounded.resources.providers[0].scratch_available_bytes = Some(24);
    assert!(session.select_whole_work(bounded.clone()).is_ok());

    bounded.resources.memory_available_bytes = Some(63);
    assert_eq!(
        session
            .select_whole_work(bounded.clone())
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
    bounded.resources.memory_available_bytes = Some(64);
    bounded.resources.providers[0].scratch_available_bytes = Some(23);
    assert_eq!(
        session
            .select_whole_work(bounded.clone())
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );

    bounded.resources.providers[0].scratch_available_bytes = Some(24);
    bounded.candidates[0].resources.cpu_millicores = 1_000;
    bounded.resources.cpu_millicores_available = 999;
    assert_eq!(
        session
            .select_whole_work(bounded.clone())
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );

    provider.resources.provider_host_workspace = Some(ResourceByteBound::Unavailable(
        runmat_execution::ResourceBoundUnavailable::NoScratchBound,
    ));
    let mut unknown = request(vec![provider]);
    unknown.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    assert_eq!(
        session
            .select_whole_work(unknown)
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:InsufficientResourceEvidence"
    );
}

#[test]
fn conditional_host_or_provider_output_does_not_double_count_same_host_payload() {
    let session = PlacementSession::default();
    let mut provider = candidate(ExecutionCandidateKind::ProviderOperation);
    provider.output_residency = CandidateOutputResidency::ConditionalHostOrProvider {
        device_id: 7,
        host_bytes: ResourceByteBound::Known(32),
    };
    provider.resources.retained_bytes = 16.into();
    provider.resources.scratch_bytes =
        ResourceByteBound::Unavailable(runmat_execution::ResourceBoundUnavailable::NoScratchBound);
    provider.resources.peak_live = Some(runmat_execution::CandidatePeakLiveDemand {
        peak_bytes: 24,
        already_live_bytes: 0,
    });
    provider.resources.provider_host_workspace = Some(32.into());
    let mut bounded = request(vec![provider]);
    bounded.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    bounded.resources.memory_available_bytes = Some(32);
    bounded.resources.providers[0].scratch_available_bytes = Some(24);
    assert!(session.select_whole_work(bounded.clone()).is_ok());

    bounded.resources.memory_available_bytes = Some(31);
    assert_eq!(
        session
            .select_whole_work(bounded.clone())
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
    bounded.resources.memory_available_bytes = Some(32);
    bounded.resources.providers[0].scratch_available_bytes = Some(23);
    assert_eq!(
        session
            .select_whole_work(bounded)
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
}

#[test]
fn conditional_provider_upload_peaks_sequential_host_and_device_stages() {
    let session = PlacementSession::default();
    let mut provider = candidate(ExecutionCandidateKind::ProviderOperation);
    provider.preparation = CandidatePreparationState::Ready;
    provider.resources.retained_bytes = 16.into();
    provider.resources.scratch_bytes =
        ResourceByteBound::Unavailable(runmat_execution::ResourceBoundUnavailable::NoScratchBound);
    provider.resources.peak_live = Some(runmat_execution::CandidatePeakLiveDemand {
        peak_bytes: 24,
        already_live_bytes: 0,
    });
    provider.resources.provider_host_workspace = Some(8.into());
    provider.presentation =
        runmat_execution::CandidateOutputPresentation::ConditionalProviderUpload {
            device_id: 7,
            transfer_bytes: Some(16),
            host_result_bytes: Some(16.into()),
            host_scratch_bytes: 8.into(),
            provider_scratch_bytes: 16.into(),
            queue_slots: 1,
        };
    let mut route = request(vec![provider]);
    route.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    route.resources.memory_available_bytes = Some(24);
    route.resources.providers[0].scratch_available_bytes = Some(32);
    assert!(session.select_whole_work(route.clone()).is_ok());

    route.resources.memory_available_bytes = Some(23);
    assert_eq!(
        session
            .select_whole_work(route.clone())
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
    route.resources.memory_available_bytes = Some(24);
    route.resources.providers[0].scratch_available_bytes = Some(31);
    assert_eq!(
        session
            .select_whole_work(route.clone())
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
    route.resources.providers[0].scratch_available_bytes = Some(32);
    route.resources.providers[0].queue_depth = Some(1);
    route.resources.providers[0].queue_limit = Some(1);
    assert_eq!(
        session
            .select_whole_work(route.clone())
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
    route.resources.providers[0].queue_depth = Some(0);
    if let runmat_execution::CandidateOutputPresentation::ConditionalProviderUpload {
        host_scratch_bytes,
        ..
    } = &mut route.candidates[0].presentation
    {
        *host_scratch_bytes = ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        );
    }
    assert_eq!(
        session
            .select_whole_work(route)
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:InsufficientResourceEvidence"
    );
}

#[test]
fn conditional_upload_retains_prepared_input_during_fallback_publication() {
    let session = PlacementSession::default();
    let mut provider = candidate(ExecutionCandidateKind::ProviderOperation);
    provider.preparation = CandidatePreparationState::Ready;
    provider.resources.retained_bytes = 16.into();
    provider.resources.peak_live = Some(runmat_execution::CandidatePeakLiveDemand {
        peak_bytes: 32,
        already_live_bytes: 0,
    });
    provider.resources.provider_host_workspace = Some(0.into());
    provider.resources.provider_input_stages = Some(
        ProviderInputStageDemands::sequential([ProviderInputStageDemand {
            device_id: 7,
            incremental_peak_bytes: 24.into(),
            queue_slots: 1,
        }])
        .unwrap(),
    );
    provider.presentation =
        runmat_execution::CandidateOutputPresentation::ConditionalProviderUpload {
            device_id: 7,
            transfer_bytes: Some(16),
            host_result_bytes: Some(16.into()),
            host_scratch_bytes: 0.into(),
            // The prepared 16-byte H remains live with the 8-byte completion
            // fence; the partition adds the new 16-byte output separately.
            provider_scratch_bytes: 24.into(),
            queue_slots: 1,
        };
    let mut route = request(vec![provider]);
    route.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    route.resources.providers[0].scratch_available_bytes = Some(40);
    assert!(session.select_whole_work(route.clone()).is_ok());
    route.resources.providers[0].scratch_available_bytes = Some(39);
    assert_eq!(
        session
            .select_whole_work(route)
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
}

#[test]
fn provider_input_stages_include_live_preparation_and_peak_against_execution() {
    let session = PlacementSession::default();
    let mut provider = candidate(ExecutionCandidateKind::ProviderOperation);
    provider.resources.retained_bytes = 24.into();
    provider.resources.peak_live = Some(runmat_execution::CandidatePeakLiveDemand {
        peak_bytes: 40,
        already_live_bytes: 0,
    });
    provider.resources.provider_input_stages = Some(
        ProviderInputStageDemands::sequential([
            ProviderInputStageDemand {
                device_id: 7,
                incremental_peak_bytes: 16.into(),
                queue_slots: 1,
            },
            // An uploaded 16-byte H remains live during its second 16-byte
            // readback, so this stage is 32 rather than just 16 staging.
            ProviderInputStageDemand {
                device_id: 7,
                incremental_peak_bytes: 32.into(),
                queue_slots: 1,
            },
        ])
        .unwrap(),
    );
    let mut bounded = request(vec![provider]);
    bounded.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    bounded.resources.providers[0].scratch_available_bytes = Some(40);
    assert!(session.select_whole_work(bounded.clone()).is_ok());
    bounded.resources.providers[0].scratch_available_bytes = Some(39);
    assert_eq!(
        session
            .select_whole_work(bounded.clone())
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );

    bounded.resources.providers[0].scratch_available_bytes = Some(40);
    bounded.resources.providers[0].queue_depth = Some(1);
    assert_eq!(
        session
            .select_whole_work(bounded)
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
}

#[test]
fn typed_prepared_inputs_and_final_kernel_share_exact_memory_boundary() {
    let session = PlacementSession::default();
    let mut provider = candidate(ExecutionCandidateKind::ProviderOperation);
    provider.resources.retained_bytes = 16.into();
    // Two preparation uploads retain 32 bytes; the selected kernel adds an
    // 80-byte output-plus-partial peak while both prepared inputs are live.
    provider.resources.peak_live = Some(runmat_execution::CandidatePeakLiveDemand {
        peak_bytes: 112,
        already_live_bytes: 0,
    });
    provider.resources.provider_input_stages = Some(
        ProviderInputStageDemands::sequential([
            ProviderInputStageDemand {
                device_id: 7,
                incremental_peak_bytes: 16.into(),
                queue_slots: 1,
            },
            ProviderInputStageDemand {
                device_id: 7,
                incremental_peak_bytes: 32.into(),
                queue_slots: 1,
            },
        ])
        .unwrap(),
    );
    let mut route = request(vec![provider]);
    route.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    route.resources.providers[0].scratch_available_bytes = Some(112);
    assert!(session.select_whole_work(route.clone()).is_ok());
    route.resources.providers[0].scratch_available_bytes = Some(111);
    assert_eq!(
        session
            .select_whole_work(route)
            .err()
            .expect("over budget")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
}

#[test]
fn cross_owner_provider_input_stage_needs_its_own_finite_quota() {
    let session = PlacementSession::default();
    let mut provider = candidate(ExecutionCandidateKind::ProviderOperation);
    provider.resources.provider_input_stages = Some(
        ProviderInputStageDemands::sequential([ProviderInputStageDemand {
            device_id: 8,
            incremental_peak_bytes: 16.into(),
            queue_slots: 1,
        }])
        .unwrap(),
    );
    let mut bounded = request(vec![provider]);
    bounded.constraint = WholeWorkConstraint::ProviderRequired { device_id: 7 };
    bounded.resources.providers.push(ProviderResourceSnapshot {
        device_id: 8,
        scratch_available_bytes: Some(16),
        queue_depth: Some(0),
        queue_limit: Some(1),
        ..Default::default()
    });
    assert!(session.select_whole_work(bounded.clone()).is_ok());
    bounded.resources.providers[1].scratch_available_bytes = Some(15);
    assert_eq!(
        session
            .select_whole_work(bounded)
            .err()
            .expect("whole-work route must be rejected")
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
        session
            .select_whole_work(bounded.clone())
            .err()
            .expect("resource-constrained route must reject")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
    bounded.resources.providers[0].scratch_available_bytes = Some(8);
    bounded.resources.memory_available_bytes = Some(23);
    assert_eq!(
        session
            .select_whole_work(bounded)
            .err()
            .expect("resource-constrained route must reject")
            .code,
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
            .err()
            .expect("resource-constrained route must reject")
            .code,
        "RunMat:Placement:InsufficientResourceEvidence"
    );
}

#[test]
fn multi_owner_host_sink_admits_each_readback_before_host_publication() {
    let session = PlacementSession::default();
    let mut host = candidate(ExecutionCandidateKind::SharedRuntime);
    host.presentation = runmat_execution::CandidateOutputPresentation::ProviderReadbacksHost {
        readbacks: Box::new(
            ProviderReadbackDemands::sequential([
                ProviderReadbackDemand {
                    device_id: 7,
                    provider_scratch_bytes: 4.into(),
                    queue_slots: 1,
                },
                ProviderReadbackDemand {
                    device_id: 8,
                    provider_scratch_bytes: 8.into(),
                    queue_slots: 1,
                },
            ])
            .unwrap(),
        ),
    };
    let mut bounded = request(vec![host]);
    bounded.resources.providers[0].scratch_available_bytes = Some(4);
    bounded.resources.providers[0].queue_depth = Some(0);
    bounded.resources.providers[0].queue_limit = Some(1);
    bounded.resources.providers.push(ProviderResourceSnapshot {
        device_id: 8,
        scratch_available_bytes: Some(8),
        queue_depth: Some(0),
        queue_limit: Some(1),
        ..Default::default()
    });
    assert!(session.select_whole_work(bounded.clone()).is_ok());
    bounded.resources.providers[1].scratch_available_bytes = Some(7);
    assert_eq!(
        session
            .select_whole_work(bounded.clone())
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
    bounded.resources.providers[1].scratch_available_bytes = Some(8);
    bounded.resources.providers[1].queue_depth = Some(1);
    assert_eq!(
        session
            .select_whole_work(bounded)
            .err()
            .expect("whole-work route must be rejected")
            .code,
        "RunMat:Placement:NoLegalCandidate"
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
