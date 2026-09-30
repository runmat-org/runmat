use super::*;
use crate::fixtures::{fixture_model, FixtureId};

#[test]
fn nonconvergence_retains_finite_original_equilibrium_diagnostics() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let excitations = crate::excitation::realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        crate::excitation::ExcitationSamplePoint::Static,
    )
    .unwrap();
    let summary = crate::assembly::assemble_structural_system(
        &model,
        &excitations,
        None,
        crate::assembly::StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .unwrap();
    let imaginary = (0..summary.dof_count)
        .map(|index| (index % 7 + 1) as f64)
        .collect::<Vec<_>>();
    let error = solve_direct_harmonic(
        &summary,
        &imaginary,
        0.5,
        DirectHarmonicPolicy {
            relative_tolerance: 1e-12,
            restart: 1,
            maximum_iterations: 1,
        },
    )
    .unwrap_err();
    let DirectHarmonicError::NotConverged {
        iterations,
        maximum_iterations,
        restart,
        reduced_dof_count,
        residual_norm,
        relative_residual,
    } = error
    else {
        panic!("unexpected error: {error}");
    };
    assert_eq!(iterations, 1);
    assert_eq!(maximum_iterations, 1);
    assert_eq!(restart, 1);
    assert!(reduced_dof_count > 0);
    assert!(residual_norm.is_finite() && residual_norm > 0.0);
    assert!(relative_residual.is_finite() && relative_residual > 1e-12);
}
