use super::*;
use crate::solve::fixed_point::AndersonOnePolicy;

#[test]
fn shared_plan_is_derived_once_and_matches_a_fresh_preparation() {
    let f = case(1, ReferenceCase::Couette, false);
    let plans = CanonicalFluidPlanCache::default();
    let sample = crate::excitation::ProjectionSample::Static;
    let fresh = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let first = CanonicalFluidProblem::prepare_shared(f.input(), sample.clone(), &plans).unwrap();
    let hit = CanonicalFluidProblem::prepare_shared(f.input(), sample.clone(), &plans).unwrap();
    assert!(std::sync::Arc::ptr_eq(&first.plan, &hit.plan));
    let boundaries = identity("plan-cache-test", &fresh.boundaries).unwrap();
    let solved = identity("plan-cache-test", &fresh.solve().unwrap()).unwrap();
    for p in [&first, &hit] {
        assert_eq!(p.numerical_input_identity, fresh.numerical_input_identity);
        assert_eq!(p.reservation(), fresh.reservation());
        assert_eq!(
            identity("plan-cache-test", &p.boundaries).unwrap(),
            boundaries
        );
        assert_eq!(
            identity("plan-cache-test", &p.solve().unwrap()).unwrap(),
            solved
        );
    }

    // A different policy is a different plan, never the cached one.
    let mut g = case(1, ReferenceCase::Couette, false);
    g.policy.fixed_point_acceleration = Some(AndersonOnePolicy::default());
    let other = CanonicalFluidProblem::prepare_shared(g.input(), sample.clone(), &plans).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&other.plan, &first.plan));
    assert_eq!(
        other.numerical_input_identity,
        CanonicalFluidProblem::prepare(g.input())
            .unwrap()
            .numerical_input_identity
    );
}
