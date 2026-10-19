//! Fresh equilibrium trajectories, not reintegration of a frozen displacement.
//! Original exact-polynomial gates remain mandatory, including every point's
//! monotone axial strain, original equilibrium and independent reactions/energies.
use super::*;
use crate::topology::CanonicalSolidIntegrationRule;

#[test]
fn fresh_tet10_integration_trajectories_meet_original_eight_step_exactness() {
    let mut failures = Vec::new();
    for rule in [
        CanonicalSolidIntegrationRule::Keast15,
        CanonicalSolidIntegrationRule::Duffy5,
        CanonicalSolidIntegrationRule::Duffy8,
    ] {
        // Start from zero original point history for each rule. No history
        // interpolation, mid-trajectory rule switch or old-state replay occurs.
        let run = execution::run_with_rule(1, true, 8, rule);
        let e = run.errors;
        if !(e.displacement < 1e-7
            && e.stress < 1e-7
            && e.alpha < 1e-7
            && e.energies.iter().all(|e| *e < 1e-7)
            && e.lateral_moment < 1e-7)
        {
            failures.push((rule, e));
        }
    }
    assert!(
        failures.is_empty(),
        "original fresh trajectory gates: {failures:?}"
    );
}
