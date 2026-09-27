use super::*;
use crate::tiering::CompilationMode;

fn site(name: &str) -> TierSiteId {
    TierSiteId {
        entry: name.into(),
        function: ProgramFunctionId(3),
        loop_header: None,
    }
}

fn session() -> TieringSession {
    TieringSession::new(TieringConfig {
        generic_hot_threshold: 2,
        specialized_hot_threshold: 2,
        deterministic: true,
        max_sites: 2,
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn unprofiled_samples_reach_generic_tier_without_fabricated_profile_samples() {
    let session = session();
    let target = site("unprofiled");
    for elapsed in [30, 40] {
        session
            .observe_unprofiled_invocation(target.clone(), elapsed)
            .unwrap();
    }
    assert_eq!(
        session.decide_generic(&target, &TierAvailability::default()),
        TierDecision::CompileGeneric {
            mode: CompilationMode::DeterministicSynchronous
        }
    );
    let available = TierAvailability {
        generic_ready: true,
        ..Default::default()
    };
    assert_eq!(
        session.decide_generic(&target, &available),
        TierDecision::ExecuteGeneric
    );
    let snapshot = session.snapshot();
    assert_eq!(snapshot.sites[0].invocations, 2);
    assert_eq!(snapshot.sites[0].total_elapsed_ns, 70);
    assert!(snapshot.sites[0].profiles.is_empty());
}

#[test]
fn retained_specialization_cannot_capture_unprofiled_invocations() {
    let session = session();
    let target = site("mixed");
    let profile = RepresentationProfile::from_facts(vec![], 64).unwrap();
    session
        .observe_invocation(target.clone(), &profile, 10)
        .unwrap();
    session
        .observe_invocation(target.clone(), &profile, 20)
        .unwrap();
    session
        .observe_unprofiled_invocation(target.clone(), 30)
        .unwrap();
    let mut available = TierAvailability {
        generic_ready: true,
        specialized_profiles: vec![profile.digest],
        ..Default::default()
    };
    assert_eq!(
        session.decide(&target, &available),
        TierDecision::ExecuteSpecialized {
            profile: profile.digest
        }
    );
    assert_eq!(
        session.decide_generic(&target, &available),
        TierDecision::ExecuteGeneric
    );
    available.generic_ready = false;
    assert_eq!(
        session.decide_generic(&target, &available),
        TierDecision::CompileGeneric {
            mode: CompilationMode::DeterministicSynchronous
        }
    );
    available.pending_compilations = session.config().max_pending_compilations;
    assert_eq!(
        session.decide_generic(&target, &available),
        TierDecision::Interpret
    );
    let snapshot = session.snapshot();
    assert_eq!(snapshot.sites[0].invocations, 3);
    assert_eq!(snapshot.sites[0].profiles[0].samples, 2);
}

#[test]
fn unprofiled_feedback_obeys_site_bounds_and_backedges_still_apply() {
    let session = session();
    for name in ["old", "middle", "latest"] {
        session
            .observe_unprofiled_invocation(site(name), 1)
            .unwrap();
    }
    let snapshot = session.snapshot();
    assert_eq!(snapshot.sites.len(), 2);
    assert!(!snapshot.sites.iter().any(|value| value.site.entry == "old"));
    session
        .observe_backedge(site("latest"), session.config().loop_hot_threshold)
        .unwrap();
    assert_eq!(
        session.decide_generic(&site("latest"), &TierAvailability::default()),
        TierDecision::CompileGeneric {
            mode: CompilationMode::DeterministicSynchronous
        }
    );
    assert!(session.observe_unprofiled_invocation(site(""), 1).is_err());
    assert_eq!(session.snapshot().sites.len(), 2);
}
