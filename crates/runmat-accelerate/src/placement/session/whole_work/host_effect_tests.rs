use super::*;
use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputPresentation, CandidateOutputResidency,
    CandidatePreparationState, CandidateResourceDemand, PlacementExecutionAuthorityRevision,
    PlacementResourceSnapshot, PlacementSignature, ResourceBoundUnavailable, ResourceByteBound,
};
use runmat_runtime::context::RuntimePlacementService;

fn request(
    session: &PlacementSession,
    retained: ResourceByteBound,
    scratch: ResourceByteBound,
) -> WholeWorkPlacementRequest {
    let host_service = Digest::sha256(b"installed-archive-service-and-codec");
    let revision = session
        .whole_work_host_service_revision(None, host_service)
        .expect("placement owns host-service revision");
    assert_eq!(
        revision.authority,
        PlacementExecutionAuthorityRevision::HostService(host_service)
    );
    WholeWorkPlacementRequest {
        signature: PlacementSignature {
            region: None,
            operation: "unzip".into(),
            runtime_facts: Digest::sha256(b"specific-admitted-archive"),
            revision,
        },
        constraint: WholeWorkConstraint::AnyLegal,
        candidates: vec![WholeWorkCandidate {
            kind: ExecutionCandidateKind::SharedRuntime,
            execution_location: CandidateExecutionLocation::Host,
            preparation: CandidatePreparationState::Ready,
            output_residency: CandidateOutputResidency::Host,
            presentation: CandidateOutputPresentation::Direct,
            route_facts: Digest::sha256(b"direct-host-effect"),
            cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
            resources: CandidateResourceDemand {
                cpu_millicores: 1_000,
                retained_bytes: retained,
                scratch_bytes: scratch,
                provider_host_workspace: None,
                provider_input_stages: None,
                queue_slots: 0,
                peak_live: None,
            },
            transactional_results: false,
        }],
        resources: PlacementResourceSnapshot {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(36),
            cancellation_requested: false,
            providers: Vec::new(),
            epoch: 1,
        },
        deterministic: true,
        require_transactional_results: false,
    }
}

#[test]
fn host_service_direct_effect_obeys_real_whole_work_memory_account() {
    let session = PlacementSession::default();
    let mut bounded = request(
        &session,
        ResourceByteBound::Known(12),
        ResourceByteBound::Known(24),
    );
    let selected = session
        .select_whole_work(bounded.clone())
        .expect("exact bound fits");
    assert_eq!(
        selected.decision.selected.kind,
        ExecutionCandidateKind::SharedRuntime
    );
    drop(selected);

    bounded.resources.memory_available_bytes = Some(35);
    assert_eq!(
        session
            .select_whole_work(bounded.clone())
            .err()
            .unwrap()
            .code,
        "RunMat:Placement:NoLegalCandidate"
    );
    bounded.resources.memory_available_bytes = Some(36);
    bounded.candidates[0].resources.scratch_bytes =
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound);
    assert_eq!(
        session.select_whole_work(bounded).err().unwrap().code,
        "RunMat:Placement:InsufficientResourceEvidence"
    );
}
