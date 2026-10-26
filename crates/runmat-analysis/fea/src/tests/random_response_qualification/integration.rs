use super::{
    dense_reference::{integrate_dense_response, maximum_relative_error},
    fixture::{
        channel, complete_physical_basis, constant_spectrum, converged_policy,
        full_frequency_bounds, hermitian_density,
    },
};
use crate::solve::dynamics::solve_modal_random_response;

#[test]
fn physical_mdof_response_matches_dense_dynamic_stiffness_reference() {
    let (summary, basis) = complete_physical_basis();
    let bounds = full_frequency_bounds(&basis);
    let excitation = constant_spectrum(
        bounds,
        vec![channel("oblique", [0.73, -0.41, 0.29])],
        runmat_analysis_core::CrossPowerSpectralDensity::diagonal(
            &runmat_types::physical::Quantity::new([2.5e-4]),
        ),
    );
    let policy = converged_policy(0.999_999_999);
    let result = solve_modal_random_response(&summary, &basis, &excitation, policy).unwrap();
    assert_eq!(result.selected_mode_count(), basis.modes().len());
    let reference = integrate_dense_response(
        &summary,
        &basis,
        &excitation,
        policy.modal_damping_ratio,
        120_000,
    );
    let error = maximum_relative_error(result.dof_variance(), &reference);
    assert!(
        error < 3.0e-5,
        "dense MDOF relative variance error={error:e}"
    );
}

#[test]
fn complex_cpsd_phase_and_correlation_match_dense_reference() {
    let (summary, basis) = complete_physical_basis();
    let bounds = full_frequency_bounds(&basis);
    let channels = vec![
        channel("primary", [0.81, 0.34, -0.17]),
        channel("cross", [-0.22, 0.69, 0.61]),
    ];
    let policy = converged_policy(0.999_999_999);
    let positive = constant_spectrum(
        bounds,
        channels.clone(),
        hermitian_density([3.0e-4, 1.7e-4], 0.72, 0.83),
    );
    let zero_phase = constant_spectrum(
        bounds,
        channels,
        hermitian_density([3.0e-4, 1.7e-4], 0.72, 0.0),
    );
    let positive_result = solve_modal_random_response(&summary, &basis, &positive, policy).unwrap();
    let zero_phase_result =
        solve_modal_random_response(&summary, &basis, &zero_phase, policy).unwrap();
    for (excitation, result) in [
        (&positive, &positive_result),
        (&zero_phase, &zero_phase_result),
    ] {
        let reference = integrate_dense_response(
            &summary,
            &basis,
            excitation,
            policy.modal_damping_ratio,
            120_000,
        );
        let error = maximum_relative_error(result.dof_variance(), &reference);
        assert!(
            error < 3.0e-5,
            "complex CPSD relative variance error={error:e}"
        );
    }
    let phase_effect = positive_result
        .dof_variance()
        .iter()
        .zip(zero_phase_result.dof_variance())
        .filter(|(positive, zero_phase)| positive.abs().max(zero_phase.abs()) > 1.0e-24)
        .map(|(positive, zero_phase)| {
            (positive - zero_phase).abs() / positive.abs().max(zero_phase.abs())
        })
        .fold(0.0, f64::max);
    assert!(
        phase_effect > 1.0e-6,
        "CPSD phase must remain observable, relative effect={phase_effect:e}"
    );
}

#[test]
fn deliberately_under_resolved_resonance_is_rejected() {
    let (summary, basis) = complete_physical_basis();
    let frequency = basis.modes()[0].frequency_hz();
    let excitation = constant_spectrum(
        [frequency * 0.7, frequency * 1.3],
        vec![channel("resonant", [0.73, -0.41, 0.29])],
        runmat_analysis_core::CrossPowerSpectralDensity::diagonal(
            &runmat_types::physical::Quantity::new([1.0]),
        ),
    );
    let mut policy = converged_policy(0.0);
    policy.relative_variance_tolerance = 1.0e-14;
    policy.maximum_refinement_levels = 1;
    let error = solve_modal_random_response(&summary, &basis, &excitation, policy).unwrap_err();
    assert!(error.to_string().contains("did not converge"), "{error}");
}
