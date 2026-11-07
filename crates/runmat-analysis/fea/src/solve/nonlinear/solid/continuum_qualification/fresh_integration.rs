//! Fresh equilibrium trajectories, not reintegration of a frozen displacement.
//! Original exact-polynomial gates remain mandatory, including every point's
//! monotone axial strain, original equilibrium and independent reactions/energies.
use super::*;
use crate::topology::CanonicalSolidIntegrationRule;

// witness: FEA-NONLINEAR-OPERATOR-001/references @ small_strain_j2/tet10/numerical_kernel;
//   coverage: partial;
//   check: j2-manufactured-cube; boundary: fresh trajectories under Keast15, Duffy5 and Duffy8 are exact in one increment and
//   their eight-increment path error (the force-controlled yield front crossing elements mid-step,
//   see the cube test) converges at order >= 1.5 under every rule.
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
        let e = execution::run_with_rule(1, true, 1, rule).errors;
        if !(e.displacement < 1e-7
            && e.stress < 1e-7
            && e.alpha < 1e-7
            && e.energies.iter().all(|e| *e < 1e-7)
            && e.lateral_moment < 1e-7)
        {
            failures.push((rule, e));
        }
        let stepped = [1, 2, 4].map(|cells| {
            let run = execution::run_with_rule(cells, true, 8, rule);
            (run.errors.displacement, run.plastic_off_axis)
        });
        assert_stepped_path_error_converges(&stepped, &format!("fresh {rule:?}"));
    }
    assert!(
        failures.is_empty(),
        "original fresh trajectory gates: {failures:?}"
    );
}
