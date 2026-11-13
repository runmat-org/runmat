#![cfg(not(target_arch = "wasm32"))]
//! Profile-plumbing contract tests. Controlled profile moments are fixtures,
//! not FEA performance benchmarks; no numerical solve is started here.
use runmat_core::{
    CheckedFeaSessionSource, FeaExecutionTimingProfile, FeaTimingRoute, RunMatSession,
};
use std::time::Duration;

const STUDY: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh volume on geometry.part {
    tetrahedra { order = linear; target_size = 0.1 m; maximum_deviation = 0.01 m }
}
model structure {}
study static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve { structural static {} }
    output displacement = analysis.solve.field.node.displacement
}
"#;

fn fixture() -> (tempfile::TempDir, RunMatSession, CheckedFeaSessionSource) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("study.fea");
    std::fs::write(&path, STUDY).unwrap();
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let source =
        futures::executor::block_on(session.load_and_check_fea_source_path(&path)).unwrap();
    assert!(source.check_result().status.is_valid());
    (directory, session, source)
}

fn seed(session: &RunMatSession, source: &CheckedFeaSessionSource, cores: u32, millis: &[u64]) {
    let before = session.placement_profile_snapshot();
    let observation = session
        .fea_execution_timing_profile()
        .for_source(source, FeaTimingRoute::NativeCheckedSource, cores)
        .unwrap()
        .start();
    std::thread::sleep(Duration::from_millis(2));
    observation.complete(true);
    let mut snapshot = session.placement_profile_snapshot();
    let series = snapshot
        .feedback
        .iter_mut()
        .find(|series| {
            !before.feedback.iter().any(|old| {
                old.signature == series.signature
                    && old.samples == series.samples
                    && old.total_elapsed_ns == series.total_elapsed_ns
            })
        })
        .expect("actual timer must create/update its existing profile series");
    series.samples = millis.len() as u64;
    series.failures = 0;
    series.total_elapsed_ns = millis.iter().map(|n| u128::from(*n) * 1_000_000).sum();
    series.total_squared_elapsed_ns = millis
        .iter()
        .map(|n| (u128::from(*n) * 1_000_000).pow(2))
        .sum();
    session.restore_placement_profile(snapshot).unwrap();
}
fn estimate(
    profile: &FeaExecutionTimingProfile,
    source: &CheckedFeaSessionSource,
    route: FeaTimingRoute,
    cores: u32,
) -> Option<u64> {
    profile
        .for_source(source, route, cores)
        .unwrap()
        .estimate()
        .map(|value| value.0)
}

#[test]
fn exact_host_route_and_allocation_require_their_own_measurements() {
    let (_directory, session, source) = fixture();
    let profile = session.fea_execution_timing_profile();
    assert_eq!(
        estimate(&profile, &source, FeaTimingRoute::NativeCheckedSource, 1),
        None
    );
    seed(&session, &source, 1, &[1000, 1000, 1000]);
    assert_eq!(
        estimate(&profile, &source, FeaTimingRoute::NativeCheckedSource, 1),
        Some(1000)
    );
    assert_eq!(
        estimate(&profile, &source, FeaTimingRoute::NativeCheckedSource, 4),
        None
    );
    assert_eq!(
        estimate(&profile, &source, FeaTimingRoute::BrowserCheckedSource, 1),
        None
    );
    seed(&session, &source, 4, &[700, 710, 720]);
    assert_eq!(
        estimate(&profile, &source, FeaTimingRoute::NativeCheckedSource, 4),
        Some(710)
    );
    let other = RunMatSession::with_options(false, false).unwrap();
    other
        .restore_placement_profile(session.placement_profile_snapshot())
        .unwrap();
    assert_eq!(
        estimate(
            &other.fea_execution_timing_profile(),
            &source,
            FeaTimingRoute::NativeCheckedSource,
            1
        ),
        None
    );
}

#[test]
fn changed_source_and_replaced_profiles_reject_old_evidence_and_late_receipts() {
    let (directory, mut session, source) = fixture();
    let profile = session.fea_execution_timing_profile();
    seed(&session, &source, 1, &[1000, 1000, 1000]);
    std::fs::write(
        directory.path().join("study.fea"),
        STUDY.replace("0.1 m", "0.2 m"),
    )
    .unwrap();
    let changed = futures::executor::block_on(
        session.load_and_check_fea_source_path(&directory.path().join("study.fea")),
    )
    .unwrap();
    assert!(changed.check_result().status.is_valid());
    assert_eq!(
        estimate(&profile, &changed, FeaTimingRoute::NativeCheckedSource, 1),
        None
    );
    let late = profile
        .for_source(&source, FeaTimingRoute::NativeCheckedSource, 1)
        .unwrap()
        .start();
    let mut empty = session.placement_profile_snapshot();
    empty.feedback.clear();
    session.restore_placement_profile(empty).unwrap();
    late.complete(true);
    assert!(session.placement_profile_snapshot().feedback.is_empty());
}

#[test]
fn actual_monotonic_measurement_cancellation_and_failure_use_the_existing_profile() {
    let (_directory, session, source) = fixture();
    let profile = session.fea_execution_timing_profile();
    let observation = profile
        .for_source(&source, FeaTimingRoute::NativeCheckedSource, 1)
        .unwrap()
        .start();
    std::thread::sleep(Duration::from_millis(2));
    observation.complete(true);
    let snapshot = session.placement_profile_snapshot();
    assert_eq!(snapshot.feedback.len(), 1);
    assert_eq!(snapshot.feedback[0].samples, 1);
    assert!(snapshot.feedback[0].total_elapsed_ns >= 1_000_000);
    drop(
        profile
            .for_source(&source, FeaTimingRoute::NativeCheckedSource, 1)
            .unwrap()
            .start(),
    );
    assert_eq!(session.placement_profile_snapshot().feedback[0].samples, 1);
    seed(&session, &source, 1, &[1000, 1000, 1000]);
    assert_eq!(
        estimate(&profile, &source, FeaTimingRoute::NativeCheckedSource, 1),
        Some(1000)
    );
    profile
        .for_source(&source, FeaTimingRoute::NativeCheckedSource, 1)
        .unwrap()
        .start()
        .complete(false);
    assert_eq!(
        estimate(&profile, &source, FeaTimingRoute::NativeCheckedSource, 1),
        None
    );
}

#[test]
fn sparse_or_unstable_measurements_stay_unavailable() {
    let (_directory, session, source) = fixture();
    let profile = session.fea_execution_timing_profile();
    seed(&session, &source, 1, &[1000, 1000]);
    assert_eq!(
        estimate(&profile, &source, FeaTimingRoute::NativeCheckedSource, 1),
        None
    );
    seed(&session, &source, 1, &[700, 1000, 1300]);
    assert_eq!(
        estimate(&profile, &source, FeaTimingRoute::NativeCheckedSource, 1),
        None
    );
    seed(&session, &source, 1, &[900, 1000, 1100]);
    let (mean, evidence) = profile
        .for_source(&source, FeaTimingRoute::NativeCheckedSource, 1)
        .unwrap()
        .estimate()
        .unwrap();
    assert_eq!(mean, 1000);
    assert_eq!(evidence.sample_count, 3);
    assert!(evidence.lower_millis <= 800 && evidence.upper_millis >= 1200);
}
