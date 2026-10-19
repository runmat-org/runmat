use super::*;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity};
use runmat_execution::{PlacementExecutionAuthorityRevision, PlacementRevision};

fn signature(token: &str) -> PlacementSignature {
    let digest = Digest::sha256(token.as_bytes());
    PlacementSignature {
        region: None,
        operation: token.into(),
        runtime_facts: digest,
        provider_provenance: None,
        graph_output_authority: None,
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
        PlacementExecutionAuthorityRevision::HostService(_)
        | PlacementExecutionAuthorityRevision::RuntimeBindingSession(_) => {
            unreachable!("provider fixture")
        }
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
fn explicit_provider_invalidation_is_exact_for_session_bound_decisions_and_samples() {
    let session = PlacementSession::default();
    let provider_a = Digest::sha256(b"capability A");
    let provider_b = Digest::sha256(b"capability B");
    let mut a = signature("fusion.reduction");
    a.revision.authority = PlacementExecutionAuthorityRevision::RuntimeBindingSession(
        Digest::sha256(b"same VM semantic revision"),
    );
    a.provider_provenance = Some(provider_a);
    let mut b = a.clone();
    b.provider_provenance = Some(provider_b);
    for selected in [&a, &b] {
        for _ in 0..2 {
            session
                .observe(PlacementFeedback {
                    signature: selected.clone(),
                    candidate: "physical".into(),
                    kind: ExecutionCandidateKind::ProviderFusion,
                    total_elapsed_ns: 13,
                    succeeded: true,
                })
                .unwrap();
        }
    }
    for selected in [&a, &b] {
        session.state.borrow_mut().decisions.insert(
            selected.cache_key(),
            CachedDecision {
                decision: PlacementDecision {
                    signature: selected.clone(),
                    selections: Vec::new(),
                    predicted_total_ns: 1,
                    from_cache: false,
                    used_local_fallback: false,
                    explored_states: 0,
                    pruned_states: 0,
                },
                resources: Default::default(),
                last_used: 0,
            },
        );
    }
    assert_eq!(session.state.borrow().decisions.len(), 2);
    assert_eq!(session.state.borrow().feedback.len(), 2);
    assert!(session.profile_snapshot().feedback.is_empty());
    session.invalidate(PlacementInvalidation::Provider { digest: provider_a });
    assert!(!session
        .state
        .borrow()
        .decisions
        .contains_key(&a.cache_key()));
    assert!(session
        .state
        .borrow()
        .decisions
        .contains_key(&b.cache_key()));
    assert!(matches!(
        session.observed_candidate_duration(&a, "physical", ExecutionCandidateKind::ProviderFusion),
        CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
    ));
    assert!(matches!(
        session.observed_candidate_duration(&b, "physical", ExecutionCandidateKind::ProviderFusion),
        CandidateObservationEvidence::Observed(_)
    ));
}

#[test]
fn portable_profile_rejects_forged_provider_provenance() {
    let session = PlacementSession::default();
    let mut portable = signature("portable-provider-series");
    portable.provider_provenance = Some(Digest::sha256(b"forged provider namespace"));
    session
        .observe(PlacementFeedback {
            signature: portable,
            candidate: "portable".into(),
            kind: ExecutionCandidateKind::ProviderOperation,
            total_elapsed_ns: 11,
            succeeded: true,
        })
        .unwrap();
    let snapshot = session.profile_snapshot();
    assert_eq!(snapshot.feedback.len(), 1);
    assert_eq!(
        PlacementSession::default()
            .restore_profile(snapshot)
            .unwrap_err()
            .code,
        "RunMat:Placement:ProfileEntry"
    );
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

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn runtime_binding_authority_is_session_local_and_cannot_cross_profile_boundary() {
    use runmat_runtime::context::RuntimePlacementService;

    let binding = BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "strcmp" },
        variant: "default",
    };
    let generation = Digest::sha256(b"scoped registration generation one");
    let session = PlacementSession::default();
    let revision = session
        .whole_work_runtime_binding_session_revision(None, binding, generation)
        .expect("session owns a local binding revision");
    let other_session = PlacementSession::default();
    assert_ne!(
        revision,
        other_session
            .whole_work_runtime_binding_session_revision(None, binding, generation)
            .unwrap()
    );
    assert_ne!(
        revision,
        session
            .whole_work_runtime_binding_session_revision(
                None,
                BuiltinBindingIdentity {
                    variant: "other",
                    ..binding
                },
                generation,
            )
            .unwrap()
    );
    assert_ne!(
        revision,
        session
            .whole_work_runtime_binding_session_revision(
                None,
                binding,
                Digest::sha256(b"scoped registration generation two"),
            )
            .unwrap()
    );

    let mut local = signature("runtime-binding-local");
    local.revision = revision;
    let permit = session.observation_permits.register(&local).unwrap();
    for _ in 0..2 {
        session
            .observe(PlacementFeedback {
                signature: local.clone(),
                candidate: "selected-host-route".into(),
                kind: ExecutionCandidateKind::SharedRuntime,
                total_elapsed_ns: 17,
                succeeded: true,
            })
            .unwrap();
    }
    assert!(matches!(
        session.observed_candidate_duration(
            &local,
            "selected-host-route",
            ExecutionCandidateKind::SharedRuntime
        ),
        CandidateObservationEvidence::Observed(_)
    ));
    assert!(session.profile_snapshot().feedback.is_empty());

    let existing = signature("retained-on-rejected-import");
    let existing_permit = other_session
        .observation_permits
        .register(&existing)
        .unwrap();
    other_session
        .observe(PlacementFeedback {
            signature: existing.clone(),
            candidate: "preexisting-provider".into(),
            kind: ExecutionCandidateKind::SharedRuntime,
            total_elapsed_ns: 23,
            succeeded: true,
        })
        .unwrap();
    let before = other_session.profile_snapshot();
    assert_eq!(before.feedback.len(), 1);
    assert!(existing_permit.is_current());

    let series = session
        .state
        .borrow()
        .feedback
        .values()
        .next()
        .unwrap()
        .clone();
    let forged = PlacementProfileSnapshot {
        schema_version: PLACEMENT_PROFILE_SCHEMA_VERSION,
        feedback: vec![series],
    };
    assert_eq!(
        other_session.restore_profile(forged).unwrap_err().code,
        "RunMat:Placement:ProfileEntry"
    );
    assert_eq!(other_session.profile_snapshot(), before);
    assert!(existing_permit.is_current());
    let digest = match local.revision.authority {
        PlacementExecutionAuthorityRevision::RuntimeBindingSession(digest) => digest,
        _ => unreachable!("local runtime binding revision"),
    };
    session.invalidate(PlacementInvalidation::RuntimeBindingSession { digest });
    assert!(!permit.is_current());
    assert!(matches!(
        session.observed_candidate_duration(
            &local,
            "selected-host-route",
            ExecutionCandidateKind::SharedRuntime
        ),
        CandidateObservationEvidence::Unavailable(CandidateCostUnavailable::NoObservations)
    ));
}
