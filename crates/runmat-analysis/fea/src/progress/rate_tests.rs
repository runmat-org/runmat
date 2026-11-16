use super::*;

fn advanced(phase: FeaProgressPhase, unit: &str, current: u64, total: Option<u64>) -> FeaProgressEvent {
    FeaProgressEvent {
        operation: "test".into(),
        phase,
        status: FeaProgressStatus::Advanced,
        message: String::new(),
        current: Some(current),
        total,
        fraction: total.map(|total| current as f64 / total as f64),
        metadata: BTreeMap::from([(UNIT_METADATA.to_string(), unit.to_string())]),
    }
}

fn at(origin: Instant, millis: u64) -> Instant {
    origin + Duration::from_millis(millis)
}

#[test]
fn counts_pass_at_most_ten_per_second_and_only_when_they_move() {
    let origin = Instant::now();
    let mut limiter = ProgressRateLimiter::new();
    let phase = FeaProgressPhase::ModelAssembly;
    assert!(limiter.admit(&advanced(phase, "elements", 0, Some(1000)), at(origin, 0)));
    // Too soon, although the count moved 10%.
    assert!(!limiter.admit(&advanced(phase, "elements", 100, Some(1000)), at(origin, 50)));
    // Interval passed but the fraction moved only 0.4%.
    assert!(!limiter.admit(&advanced(phase, "elements", 4, Some(1000)), at(origin, 150)));
    assert!(limiter.admit(&advanced(phase, "elements", 5, Some(1000)), at(origin, 160)));
    // Two seconds without an event admits even an unchanged count.
    assert!(limiter.admit(&advanced(phase, "elements", 5, Some(1000)), at(origin, 2200)));
    // The final count always passes.
    assert!(limiter.admit(&advanced(phase, "elements", 1000, Some(1000)), at(origin, 2201)));
    assert!(!limiter.admit(&advanced(phase, "elements", 1000, Some(1000)), at(origin, 2202)));
}

#[test]
fn a_flood_of_counts_is_thinned_to_the_event_interval() {
    let origin = Instant::now();
    let mut limiter = ProgressRateLimiter::new();
    let admitted = (0..=10_000_u64)
        .filter(|step| {
            limiter.admit(
                &advanced(FeaProgressPhase::Solve, "x", *step, Some(10_000)),
                at(origin, *step),
            )
        })
        .count();
    // 10 s of events every millisecond: about 10 per second plus the final one.
    assert!((95..=102).contains(&admitted), "{admitted}");
}

#[test]
fn streams_are_independent_and_phase_boundaries_always_pass() {
    let origin = Instant::now();
    let mut limiter = ProgressRateLimiter::new();
    let mesh = FeaProgressPhase::MeshPrep;
    assert!(limiter.admit(&advanced(mesh, "surface_faces", 1, Some(10)), at(origin, 0)));
    assert!(limiter.admit(&advanced(mesh, "volume_points", 1, None), at(origin, 1)));
    let mut completed = advanced(mesh, "surface_faces", 2, Some(10));
    completed.status = FeaProgressStatus::Completed;
    assert!(limiter.admit(&completed, at(origin, 2)));
    // The boundary restarted the phase's streams.
    assert!(limiter.admit(&advanced(mesh, "surface_faces", 1, Some(10)), at(origin, 3)));
}

#[test]
fn count_only_streams_need_relative_growth_and_restarts_pass() {
    let origin = Instant::now();
    let mut limiter = ProgressRateLimiter::new();
    let phase = FeaProgressPhase::Solve;
    assert!(limiter.admit(&advanced(phase, "iterations", 1000, None), at(origin, 0)));
    assert!(!limiter.admit(&advanced(phase, "iterations", 1004, None), at(origin, 200)));
    assert!(limiter.admit(&advanced(phase, "iterations", 1005, None), at(origin, 300)));
    // A new solve restarts its iteration count.
    assert!(limiter.admit(&advanced(phase, "iterations", 1, None), at(origin, 400)));
}
