use super::*;
use runmat_execution::{ProgramEnvironment, ProgramRevision};

fn program() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"study graph"),
        Digest::sha256(b"source"),
        ProgramEnvironment::new(
            2,
            2,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"builtins"),
            "runmat",
        )
        .unwrap(),
    )
    .unwrap()
}
fn key(profile: &FeaExecutionTimingProfile, cores: u32) -> FeaExecutionTimingKey {
    profile
        .for_revision(
            program(),
            [1; 32],
            FeaTimingRoute::NativeCheckedSource,
            cores,
        )
        .unwrap()
}
fn record(key: &FeaExecutionTimingKey, millis: u64) {
    key.profile
        .placement
        .observe(PlacementFeedback {
            signature: key.signature.clone(),
            candidate: CANDIDATE.into(),
            kind: key.kind,
            total_elapsed_ns: millis * 1_000_000,
            succeeded: true,
        })
        .unwrap();
}
fn session() -> RunMatSession {
    RunMatSession::with_options(false, false).unwrap()
}

#[test]
fn no_number_for_missing_sparse_or_unstable_measurements() {
    let mut untimed = session();
    untimed.runtime_context = untimed.runtime_context.clone().with_service_ports(
        untimed
            .runtime_context
            .service_ports()
            .clone()
            .without_timing(),
    );
    assert!(untimed
        .fea_execution_timing_profile()
        .for_revision(program(), [1; 32], FeaTimingRoute::NativeCheckedSource, 1)
        .is_err());
    let session = session();
    let profile = session.fea_execution_timing_profile();
    let key = key(&profile, 1);
    assert!(key.estimate().is_none());
    record(&key, 1000);
    record(&key, 1000);
    assert!(key.estimate().is_none());
    record(&key, 1000);
    let (middle, evidence) = key.estimate().unwrap();
    assert_eq!(middle, 1000);
    assert_eq!(evidence.sample_count, 3);
    assert_eq!((evidence.lower_millis, evidence.upper_millis), (1000, 1000));
    record(&key, 10000);
    assert!(key.estimate().is_none());
}

#[test]
fn variation_is_measured_not_invented_or_silently_discarded() {
    let session = session();
    let profile = session.fea_execution_timing_profile();
    let key = key(&profile, 1);
    for millis in [900, 1000, 1100] {
        record(&key, millis);
    }
    let (middle, evidence) = key.estimate().unwrap();
    assert_eq!(middle, 1000);
    assert!(evidence.lower_millis <= 800 && evidence.upper_millis >= 1200);
    assert_eq!(evidence.sample_count, 3);
}

#[test]
fn serial_parallel_browser_source_and_machine_are_separate_evidence() {
    let first = session();
    let profile = first.fea_execution_timing_profile();
    let original = key(&profile, 1);
    for millis in [1000, 1000, 1000] {
        record(&original, millis);
    }
    assert!(original.estimate().is_some());
    assert!(key(&profile, 8).estimate().is_none());
    assert!(profile
        .for_revision(program(), [1; 32], FeaTimingRoute::BrowserCheckedSource, 1)
        .unwrap()
        .estimate()
        .is_none());
    assert!(profile
        .for_revision(program(), [2; 32], FeaTimingRoute::NativeCheckedSource, 1)
        .unwrap()
        .estimate()
        .is_none());
    let second = session();
    second
        .restore_placement_profile(first.placement_profile_snapshot())
        .unwrap();
    assert!(key(&second.fea_execution_timing_profile(), 1)
        .estimate()
        .is_none());
    // Actual multi-core observations, rather than serial/cores, are required.
    let parallel = key(&profile, 8);
    for millis in [700, 710, 720] {
        record(&parallel, millis);
    }
    assert_eq!(parallel.estimate().unwrap().0, 710);
    assert_eq!(original.estimate().unwrap().0, 1000);
}

#[test]
fn failure_retires_evidence_and_cancellation_never_adds_a_sample() {
    let session = session();
    let profile = session.fea_execution_timing_profile();
    let original = key(&profile, 1);
    for millis in [1000, 1000, 1000] {
        record(&original, millis);
    }
    drop(key(&profile, 1).start());
    assert_eq!(original.estimate().unwrap().1.sample_count, 3);
    key(&profile, 1).start().complete(false);
    assert!(original.estimate().is_none());
}

#[test]
fn profile_invalidation_and_foreign_facts_do_not_reuse_a_number() {
    let session = session();
    let profile = session.fea_execution_timing_profile();
    let original = key(&profile, 1);
    for millis in [1000, 1000, 1000] {
        record(&original, millis);
    }
    let mut stale = key(&profile, 1);
    stale.signature.runtime_facts = Digest::sha256(b"other source or allocation");
    assert!(stale.estimate().is_none());
    profile.placement.invalidate(PlacementInvalidation::All);
    assert!(original.estimate().is_none());
    assert!(profile
        .for_revision(program(), [1; 32], FeaTimingRoute::NativeCheckedSource, 0)
        .is_err());
}

#[test]
fn rolled_back_monotonic_clock_never_becomes_a_sample() {
    let session = session();
    let profile = session.fea_execution_timing_profile();
    let mut observation = key(&profile, 1).start();
    observation.started = Duration::MAX;
    observation.complete(true);
    assert!(session.placement_profile_snapshot().feedback.is_empty());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn successful_completion_uses_the_actual_monotonic_clock() {
    let session = session();
    let profile = session.fea_execution_timing_profile();
    let observation = key(&profile, 1).start();
    std::thread::sleep(Duration::from_millis(2));
    observation.complete(true);
    let snapshot = session.placement_profile_snapshot();
    assert_eq!(snapshot.feedback.len(), 1);
    assert_eq!(snapshot.feedback[0].samples, 1);
    assert!(snapshot.feedback[0].total_elapsed_ns >= 1_000_000);
    // A completed measurement alone still does not become a prediction.
    assert!(key(&profile, 1).estimate().is_none());
}

#[test]
fn late_completions_cannot_repopulate_a_replaced_profile() {
    let session = session();
    let profile = session.fea_execution_timing_profile();
    let old = key(&profile, 1).start();
    profile.placement.invalidate(PlacementInvalidation::All);
    old.complete(true);
    assert!(session.placement_profile_snapshot().feedback.is_empty());
    let old = key(&profile, 1).start();
    session
        .restore_placement_profile(session.placement_profile_snapshot())
        .unwrap();
    old.complete(true);
    assert!(session.placement_profile_snapshot().feedback.is_empty());
}
