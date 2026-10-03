use super::*;
use runmat_execution::{PlacementExecutionAuthorityRevision, PlacementRevision};

fn signature(token: &str) -> PlacementSignature {
    let digest = Digest::sha256(token.as_bytes());
    PlacementSignature {
        region: None,
        operation: token.into(),
        runtime_facts: digest,
        revision: PlacementRevision {
            program: None,
            catalog: Digest::sha256(b"catalog"),
            compiler: Digest::sha256(b"compiler"),
            authority: PlacementExecutionAuthorityRevision::Provider(Digest::sha256(b"provider")),
            policy: Digest::sha256(b"policy"),
        },
    }
}

#[test]
fn provider_and_host_service_samples_and_invalidation_remain_separate() {
    let session = PlacementSession::default();
    let provider = signature("same-route-authority");
    let digest = match provider.revision.authority {
        PlacementExecutionAuthorityRevision::Provider(digest) => digest,
        PlacementExecutionAuthorityRevision::HostService(_) => unreachable!("provider fixture"),
    };
    let mut host = provider.clone();
    host.revision.authority = PlacementExecutionAuthorityRevision::HostService(digest);
    for signature in [&provider, &host] {
        // The shared estimator requires two successful observations; namespace
        // isolation must not bypass that confidence requirement.
        for _ in 0..2 {
            session
                .observe(PlacementFeedback {
                    signature: signature.clone(),
                    candidate: "same-route".into(),
                    kind: ExecutionCandidateKind::SharedRuntime,
                    total_elapsed_ns: 12,
                    succeeded: true,
                })
                .unwrap();
        }
    }
    assert_eq!(session.profile_snapshot().feedback.len(), 2);
    session.invalidate(PlacementInvalidation::Provider { digest });
    assert!(matches!(
        session.observed_candidate_duration(
            &provider,
            "same-route",
            ExecutionCandidateKind::SharedRuntime,
        ),
        CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
    ));
    assert!(matches!(
        session.observed_candidate_duration(
            &host,
            "same-route",
            ExecutionCandidateKind::SharedRuntime
        ),
        CandidateObservationEvidence::Observed(_)
    ));
    session.invalidate(PlacementInvalidation::HostService { digest });
    assert!(session.profile_snapshot().feedback.is_empty());
}

#[test]
fn old_provider_field_and_v2_profile_cannot_restore() {
    let session = PlacementSession::default();
    let provider = signature("schema-authority");
    session
        .observe(PlacementFeedback {
            signature: provider,
            candidate: "same-route".into(),
            kind: ExecutionCandidateKind::SharedRuntime,
            total_elapsed_ns: 12,
            succeeded: true,
        })
        .unwrap();
    let mut snapshot = session.profile_snapshot();
    snapshot.schema_version = 2;
    assert_eq!(
        PlacementSession::default()
            .restore_profile(snapshot.clone())
            .unwrap_err()
            .code,
        "RunMat:Placement:ProfileVersion"
    );

    let mut old_json = serde_json::to_value(snapshot).unwrap();
    let revision = old_json["feedback"][0]["signature"]["revision"]
        .as_object_mut()
        .unwrap();
    let authority = revision.remove("authority").unwrap();
    revision.insert("provider".into(), authority["digest"].clone());
    assert!(serde_json::from_value::<PlacementProfileSnapshot>(old_json).is_err());
}
